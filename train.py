import os
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from model import Generator, Discriminator
from config import EPOCHS, LAMBDA_L1, SAVE_DIR, LR

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(train_loader, name="model"):
    G = Generator().to(device)
    D = Discriminator().to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()

    scaler = GradScaler()

    G_losses, D_losses = [], []

    for epoch in range(EPOCHS):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            real = torch.ones_like(D(x, y))
            fake = torch.zeros_like(D(x, y))

            # Generator
            opt_G.zero_grad()
            with autocast(device):
                y_fake = G(x)
                loss_GAN = criterion_GAN(D(x, y_fake), real)
                loss_L1 = criterion_L1(y_fake, y)
                loss_G = loss_GAN + LAMBDA_L1 * loss_L1

            scaler.scale(loss_G).backward()
            scaler.step(opt_G)

            # Discriminator
            opt_D.zero_grad()
            with autocast(device):
                loss_real = criterion_GAN(D(x, y), real)
                loss_fake = criterion_GAN(D(x, y_fake.detach()), fake)
                loss_D = (loss_real + loss_fake) / 2.0

            scaler.scale(loss_D).backward()
            scaler.step(opt_D)
            scaler.update()

        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())

        print(f"{name} | Epoch {epoch} | G: {loss_G.item():.4f} | D: {loss_D.item():.4f}")

    # Save model
    os.makedirs(SAVE_DIR, exist_ok=True)

    torch.save(G.state_dict(), f"{SAVE_DIR}/{name}_G.pth")
    torch.save(D.state_dict(), f"{SAVE_DIR}/{name}_D.pth")

    return G, G_losses, D_losses