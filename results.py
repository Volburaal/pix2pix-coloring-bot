import torch
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def show_samples(G, loader):
    G.eval()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    with torch.no_grad():
        y_fake = G(x)

    x = x.cpu()
    y_fake = y_fake.cpu()
    y = y.cpu()

    for i in range(5):
        plt.figure(figsize=(10,3))

        plt.subplot(1,3,1)
        plt.imshow(x[i].permute(1,2,0)*0.5+0.5)
        plt.title("Input")

        plt.subplot(1,3,2)
        plt.imshow(y_fake[i].permute(1,2,0)*0.5+0.5)
        plt.title("Generated")

        plt.subplot(1,3,3)
        plt.imshow(y[i].permute(1,2,0)*0.5+0.5)
        plt.title("Real")

        plt.show()

def plot_losses(G_losses, D_losses, title="Training Loss"):
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.legend()
    plt.title(title)
    plt.show()

cuhk_path = "/content/data/cuhk"
anime_path = "/content/data/anime"