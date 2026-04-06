import torch
import torch.nn as nn
import torch.nn.functional as F

def down_block(in_c, out_c, normalize=True):
    layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
    if normalize:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*layers)

def up_block(in_c, out_c, dropout=False):
    layers = [
        nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU()
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = down_block(3, 64, False)
        self.d2 = down_block(64, 128)
        self.d3 = down_block(128, 256)
        self.d4 = down_block(256, 512)

        self.u1 = up_block(512, 256)
        self.u2 = up_block(512, 128)
        self.u3 = up_block(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)

        u1 = self.u1(d4)
        if u1.shape[2:] != d3.shape[2:]:
            u1 = F.interpolate(u1, size=d3.shape[2:], mode="bilinear", align_corners=False)
        u2 = self.u2(torch.cat([u1, d3], dim=1))
        if u2.shape[2:] != d2.shape[2:]:
            u2 = F.interpolate(u2, size=d2.shape[2:], mode="bilinear", align_corners=False)
        u3 = self.u3(torch.cat([u2, d2], dim=1))
        if u3.shape[2:] != d1.shape[2:]:
            u3 = F.interpolate(u3, size=d1.shape[2:], mode="bilinear", align_corners=False)

        out = self.final(torch.cat([u3, d1], dim=1))
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 1, 4, 1, 1)
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))