import torch
import torch.nn as nn
import numpy as np
from typing import NamedTuple, List
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import random
from blocks import Quantizer, ResBlock


class VQVAEResult(NamedTuple):
    x_hat: torch.Tensor
    recon_loss: torch.Tensor
    commit_loss: torch.Tensor
    vq_loss: torch.Tensor


class VQVAE(nn.Module):
    """ Discrete AutoEncoder to tokenize Image Tensors.
    - source: https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/auto_encoder.py#L277
    """

    def __init__(self, emb_dim=512, num_embs=200) -> None:
        super().__init__()
        self.quantizer = Quantizer(num_embs,
                                   emb_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=emb_dim,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(emb_dim),
            nn.ReLU(),
            nn.Conv2d(emb_dim,
                      emb_dim,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(emb_dim),
            nn.ReLU(),
            ResBlock(emb_dim, emb_dim, batch_norm=True),
            nn.BatchNorm2d(emb_dim),
            ResBlock(emb_dim, emb_dim, batch_norm=True),
            nn.BatchNorm2d(emb_dim),
        )

        self.decoder = nn.Sequential(
            ResBlock(emb_dim, emb_dim),
            nn.BatchNorm2d(emb_dim),
            ResBlock(emb_dim, emb_dim),
            nn.ConvTranspose2d(
                emb_dim, emb_dim,
                kernel_size=4,
                stride=2,
                padding=1),
            nn.BatchNorm2d(emb_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(
                emb_dim, 3,
                kernel_size=4,
                stride=2,
                padding=1
            ),
        )

    def forward(self, x: torch.Tensor) -> VQVAEResult:
        z = self.encoder(x)
        z_quantized = self.quantizer(z)
        x_hat = self.decoder(z_quantized)

        recon_loss = ((x_hat-x)**2).mean()
        commit_loss = ((z.detach()
                       - z_quantized)**2).mean()
        vq_loss = ((z-z_quantized.detach())**2).mean()
        return VQVAEResult(x_hat,
                           recon_loss,
                           commit_loss,
                           vq_loss)

    @torch.no_grad()
    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z_quantized = self.quantizer(z)
        return z_quantized

    @torch.no_grad()
    def viz_recon(self,
                  x: torch.Tensor):
        """Visualizing the reconstructed result 
        to evaluate 
        how well the model reconstructs the original input"""
        assert x.ndim == 3
        assert x.shape[0] == 3
        x = x.unsqueeze(0)
        x_hat = self.forward(x).x_hat

        x_hat = x_hat.detach().cpu()[0].permute(1, 2, 0)
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.imshow(x[0].permute(1, 2, 0).cpu())
        ax2.imshow(x_hat)
        plt.show()
