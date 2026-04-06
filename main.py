from torch.utils.data import DataLoader, ConcatDataset

from dataset import Pix2PixDataset
from config import BATCH_SIZE

from train import train_model
from results import plot_losses, show_samples

cuhk_path = "data\\cuhk"
anime_path = "data\\anime"

cuhk_ds = Pix2PixDataset(cuhk_path, "cuhk")
anime_ds = Pix2PixDataset(anime_path, "anime")

combined_ds = ConcatDataset([cuhk_ds, anime_ds])

cuhk_loader = DataLoader(cuhk_ds, batch_size=BATCH_SIZE, shuffle=True)
anime_loader = DataLoader(anime_ds, batch_size=BATCH_SIZE, shuffle=True)
combined_loader = DataLoader(combined_ds, batch_size=BATCH_SIZE, shuffle=True)


G1, G1_loss, D1_loss = train_model(cuhk_loader, "sketchy")
plot_losses(G1_loss, D1_loss, "Sketchy Model")
show_samples(G1, cuhk_loader)

# G2, G2_loss, D2_loss = train_model(anime_loader, "weeb")
# plot_losses(G2_loss, D2_loss, "Weeb Model")
# show_samples(G2, anime_loader)

# G3, G3_loss, D3_loss = train_model(combined_loader, "sketchyweeb")
# plot_losses(G3_loss, D3_loss, "Sketchy Weeb")
# show_samples(G3, combined_loader)