import torch
from vqvae import VQVAE
from utils import Trajectory
from pathlib import Path
import random
import numpy as np


def main():
    # args
    emb_dim = 64
    vocab_size = 1000
    lr = 0.005
    batch_size = 64
    epoch = 50
    device = ("cuda"
              if torch.cuda.is_available()
              else "cpu")

    __dirname = Path(__file__).parent
    vqvae = VQVAE(emb_dim,
                  num_embs=vocab_size,
                  ).to(device=device)
    trajectories = Trajectory\
        .from_pickle(__dirname / "traj.pkl")
    NUM_TOTAL_DATA = len(trajectories.obs)
    optimzer = torch.optim.Adam(vqvae.parameters(),
                                lr=lr)

    for j in range(epoch):
        observations = random.sample(trajectories.obs,
                                     NUM_TOTAL_DATA)
        observations = np.stack(observations)
        observations = torch.tensor(observations,
                                    dtype=torch.float32)
        observations = observations.permute(0, 3, 1, 2)  # (N, C, H, W)
        vq_losses = 0
        recon_losses = 0
        commit_losses = 0
        for i in range(0,
                       len(observations),
                       batch_size):

            obs = observations[i: i + batch_size]
            obs = obs.to(device=device)
            res = vqvae(obs)
            loss = (res.vq_loss
                    + 1.5*res.recon_loss
                    + 0.4*res.commit_loss)
            vq_losses += res.vq_loss.item()
            recon_losses += res.recon_loss.item()
            commit_losses += res.commit_loss.item()
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
        print(f"{recon_losses:.5f}")
        print(f"{commit_losses:.5f}")
        print(f"{vq_losses:.5f}")
        print("=-"*25)
        if j % 5 == 4:
            vqvae.viz_recon(obs[0])
            torch.save(vqvae,
                       __dirname/f"epoch_{j}.chpt")


if __name__ == "__main__":
    main()
