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

        # 损失与优化器
        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.l1_criterion = nn.L1Loss()
        t = cfg['train']
        self.optG = Adam(self.netG.parameters(), lr=t['lr'], betas=(t['beta1'], t['beta2']))
        self.optD = Adam(self.netD.parameters(), lr=t['lr'], betas=(t['beta1'], t['beta2']))
        self.lambda_recon = t['lambda_recon']

        self.checkpoint_dir = cfg['paths']['checkpoints']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _train_epoch(self, epoch: int) -> None:
        self.netG.train(); self.netD.train()
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
            loss_g = loss_g_adv + self.lambda_recon * loss_l1
            loss_g.backward()
            self.optG.step()

            pbar.set_postfix({
                'D': f"{loss_d.item():.3f}",
                'G_adv': f"{loss_g_adv.item():.3f}",
                'L1': f"{loss_l1.item():.3f}",
            })

    def _validate_and_vis(self, epoch: int) -> None:
        self.netG.eval()
        samples = []
        with torch.no_grad():
            for i, (cond, real) in enumerate(self.val_loader):
                cond = cond.to(self.device)
                real = real.to(self.device)
                fake = self.netG(cond)
                samples.append(fake[:2].cpu())
                if i >= 1:
                    break
        if samples:
            path = os.path.join(self.cfg['paths']['outputs'], f"val_samples_epoch_{epoch}.png")
            save_image_grid(samples, path, nrow=2)
            self.logger.info(f"保存验证可视化: {path}")

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
                    'optG': self.optG.state_dict(),
                    'optD': self.optD.state_dict(),
                    'epoch': epoch,
                }, ckpt_path)
                self.logger.info(f"保存检查点: {ckpt_path}")