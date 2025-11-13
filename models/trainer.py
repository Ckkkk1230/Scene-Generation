from typing import Dict
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from utils.logger import get_logger
from utils.checkpoint import save_checkpoint
from utils.visualize import save_image_grid
from data.loader import create_dataloaders
from models.generator import UNetGenerator
from models.discriminator import PatchDiscriminator


class DepthPredictionHead(nn.Module):
    """深度预测头部，用于辅助深度预测损失"""
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid()  # 输出0-1范围的深度图
        )
    
    def forward(self, x):
        return self.conv(x)


class Trainer:
    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = get_logger()

        # 数据
        self.train_loader, self.val_loader = create_dataloaders(cfg)

        # 模型
        g_args = cfg['model']['generator']
        d_args = cfg['model']['discriminator']
        self.netG = UNetGenerator(
            in_channels=g_args['in_channels'],
            out_channels=g_args['out_channels'],
            base_channels=g_args['base_channels'],
        ).to(self.device)
        self.netD = PatchDiscriminator(
            in_channels=d_args['in_channels'],
            base_channels=d_args['base_channels'],
        ).to(self.device)
        
        # 深度预测头部（新增）
        self.depth_head = DepthPredictionHead(
            in_channels=g_args['base_channels'] * 8,  # U-Net瓶颈层通道数
            out_channels=1
        ).to(self.device)

        # 损失函数
        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.l1_criterion = nn.L1Loss()
        self.depth_criterion = nn.L1Loss()  # 深度预测损失（新增）
        
        # 超参数
        t = cfg['train']
        self.optG = Adam(
            list(self.netG.parameters()) + list(self.depth_head.parameters()),  # 联合优化
            lr=t['lr'], 
            betas=(t['beta1'], t['beta2'])
        )
        self.optD = Adam(self.netD.parameters(), lr=t['lr'], betas=(t['beta1'], t['beta2']))
        
        self.lambda_recon = t['lambda_recon']
        self.lambda_depth = t.get('lambda_depth', 0.5)  # 深度损失权重（新增）

        self.checkpoint_dir = cfg['paths']['checkpoints']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _compute_depth_loss(self, fake_images, real_images, conditions):
        """计算深度预测损失（新增）"""
        # 使用真实图像和生成图像分别预测深度
        real_depth = self.depth_head(real_images)
        fake_depth = self.depth_head(fake_images)
        
        # 深度一致性损失：生成图像预测的深度应与真实图像预测的深度一致
        depth_loss = self.depth_criterion(fake_depth, real_depth.detach())
        
        return depth_loss

    def _train_epoch(self, epoch: int) -> None:
        self.netG.train()
        self.netD.train()
        self.depth_head.train()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for cond, real in pbar:
            cond = cond.to(self.device)
            real = real.to(self.device)

            # 更新判别器
            self.optD.zero_grad()
            fake = self.netG(cond)
            d_real = self.netD(torch.cat([cond, real], dim=1))
            d_fake = self.netD(torch.cat([cond, fake.detach()], dim=1))
            loss_d_real = self.adv_criterion(d_real, torch.ones_like(d_real))
            loss_d_fake = self.adv_criterion(d_fake, torch.zeros_like(d_fake))
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            self.optD.step()

            # 更新生成器
            self.optG.zero_grad()
            d_fake_for_g = self.netD(torch.cat([cond, fake], dim=1))
            loss_g_adv = self.adv_criterion(d_fake_for_g, torch.ones_like(d_fake_for_g))
            loss_l1 = self.l1_criterion(fake, real)
            
            # 深度预测损失（新增）
            loss_depth = self._compute_depth_loss(fake, real, cond)
            
            # 总生成器损失
            loss_g = (loss_g_adv + 
                     self.lambda_recon * loss_l1 + 
                     self.lambda_depth * loss_depth)
            
            loss_g.backward()
            self.optG.step()

            pbar.set_postfix({
                'D': f"{loss_d.item():.3f}",
                'G_adv': f"{loss_g_adv.item():.3f}",
                'L1': f"{loss_l1.item():.3f}",
                'Depth': f"{loss_depth.item():.3f}",  # 新增深度损失显示
            })

    def _validate_and_vis(self, epoch: int) -> None:
        self.netG.eval()
        self.depth_head.eval()
        
        samples = []
        depth_samples = []  # 新增深度可视化
        
        with torch.no_grad():
            for i, (cond, real) in enumerate(self.val_loader):
                cond = cond.to(self.device)
                real = real.to(self.device)
                fake = self.netG(cond)
                
                # 预测深度图（新增）
                fake_depth = self.depth_head(fake)
                real_depth = self.depth_head(real)
                
                samples.append(fake[:2].cpu())
                depth_samples.append({
                    'fake_depth': fake_depth[:2].cpu(),
                    'real_depth': real_depth[:2].cpu()
                })
                
                if i >= 1:
                    break
        
        # 保存生成图像
        if samples:
            img_path = os.path.join(self.cfg['paths']['outputs'], f"val_samples_epoch_{epoch}.png")
            save_image_grid(samples, img_path, nrow=2)
            self.logger.info(f"保存验证可视化: {img_path}")
            
            # 保存深度图对比（新增）
            if depth_samples:
                depth_path = os.path.join(self.cfg['paths']['outputs'], f"depth_samples_epoch_{epoch}.png")
                # 这里可以添加深度图的可视化保存逻辑
                self.logger.info(f"保存深度可视化: {depth_path}")

    def train(self) -> None:
        epochs = self.cfg['train']['epochs']
        for epoch in range(1, epochs + 1):
            self._train_epoch(epoch)
            self._validate_and_vis(epoch)
            
            if epoch % self.cfg['train']['save_every'] == 0:
                ckpt_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pt")
                save_checkpoint({
                    'netG': self.netG.state_dict(),
                    'netD': self.netD.state_dict(),
                    'depth_head': self.depth_head.state_dict(),  # 新增
                    'optG': self.optG.state_dict(),
                    'optD': self.optD.state_dict(),
                    'epoch': epoch,
                }, ckpt_path)
                self.logger.info(f"保存检查点: {ckpt_path}")