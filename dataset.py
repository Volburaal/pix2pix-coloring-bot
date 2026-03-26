import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from config import IMG_SIZE

class Pix2PixDataset(Dataset):
    def __init__(self, root_dir, mode="cuhk"):
        self.root_dir = root_dir
        self.mode = mode

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        if mode == "cuhk":
            self.files = sorted(os.listdir(os.path.join(root_dir, "photos")))
        else:
            self.files = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.mode == "cuhk":
            name = self.files[idx]

            photo = Image.open(os.path.join(self.root_dir, "photos", name)).convert("RGB")
            sketch = Image.open(os.path.join(self.root_dir, "sketches", name)).convert("RGB")

            return self.transform(sketch), self.transform(photo)

        else:
            img = Image.open(os.path.join(self.root_dir, self.files[idx])).convert("RGB")
            # img = self.transform(img)

            w = img.shape[2]
            color = img[:, :, :w//2]
            sketch = img[:, :, w//2:]

            return self.transform(sketch), self.transform(color)