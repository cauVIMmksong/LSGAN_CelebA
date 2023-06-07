import torch
import random
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import ImageFile, Image

from CelebDataset import CelebDataset
from MyModels import GeneratorModel, DiscriminatorModel

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device,"is available!")



transform = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
<<<<<<< HEAD
dataset = CelebDataset("D:\workspace\DCGAN\LSGAN_CelebA\LSGAN_CelebA\images\img_align_celeba", transform=transform)
=======
dataset = CelebDataset("/home/work/LSGAN_CelebA/LSGAN_CelebA-main/FFHQ", transform=transform)
>>>>>>> master
dataloader = DataLoader(dataset, shuffle=True, batch_size=64, num_workers=0)

def view_samples(images):
    img = torchvision.utils.make_grid(images, padding=2, normalize=True)
    img = img.cpu().numpy()
    plt.figure(figsize = (8, 8))
    plt.imshow(np.transpose(img, (1,2,0)))
    plt.show()
    
test = iter(dataloader)
sample = next(test)
print(sample.size())
view_samples(sample)

def save_progress(images, epoch, step):
    img = torchvision.utils.make_grid(images, padding=2, normalize=True)
    img = img.cpu().numpy()
    img = np.transpose(img, (1,2,0))
    img = np.uint8(img*255)
    imageio.imwrite(f"progress_pics/{epoch}-{step}.jpg", img)

def save_model_state(model, optimizer, loss, epoch, name):
    model_path = f"saved_models/{name}{epoch}.pt"
    state_dict = {
        'epoch' : epoch,
        'model_state_dict' : model.state_dict(),
        'opt_state_dict' : optimizer.state_dict(),
        'training_loss' : loss,
    }
    torch.save(state_dict, model_path)

def load_model_state(model, filename):
    model_info = torch.load(f"saved_models/{filename}.pt")
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(model_info["opt_state_dict"])
    model.load_state_dict(model_info["model_state_dict"])
    return model, optimizer

def plot_losses(gen, dis):
    fig = plt.figure(figsize=(8, 4))
    ax = plt.subplot(111)
    ax.plot(gen, label="Generator")
    ax.plot(dis, label="Discriminator")
    plt.title("Gen/Dis losses")
    ax.legend()
    plt.show()

n_z = 128
n_c = 3
n_feature_maps_g = 64
n_feature_maps_d = 64
epochs = 100

fixed_noise = torch.randn(64, n_z, 1, 1).to(device)
torch.save(fixed_noise, "fixed_noise.pt")

g_losses = []
d_losses = []

Dis = DiscriminatorModel(n_c, n_feature_maps_d).to(device)
Gen = GeneratorModel(n_z, n_feature_maps_g, n_c).to(device)

lr_g = 2e-4
lr_d = 2e-4
Dis_opt = optim.Adam(Dis.parameters(), lr=lr_d, betas=(0.5, 0.999))
Gen_opt = optim.Adam(Gen.parameters(), lr=lr_g, betas=(0.5, 0.999))
criterion = nn.MSELoss()

len(dataloader)
checkpoint = int(len(dataloader)/20)

d_running_loss = 0.
g_running_loss = 0.

for e in range(1, epochs+1):
    print(f"Epoch {e} started.")
    for i, batch in enumerate(dataloader, 1):
        
        batch_size = batch.size(0)
        real_image = batch.to(device)

        zero_labels = torch.zeros(batch_size, 1, 1, 1).to(device)
        one_labels = torch.ones(batch_size, 1, 1, 1).to(device)

        Dis.zero_grad()
        D_x = Dis(real_image)
        D_x_loss = criterion(D_x, one_labels)
        d_running_loss += D_x_loss.item()
        D_x_loss.backward()
        
        z = torch.randn(batch_size, n_z, 1, 1).to(device)  
        G_z = Gen(z)
        D_G_z = Dis(G_z.detach())
        D_G_z_loss = criterion(D_G_z, zero_labels)
        d_running_loss += D_G_z_loss.item()
        D_G_z_loss.backward()
        
        Dis_opt.step()
        
        Gen.zero_grad()
        G_z = Gen(z)
        D_G_z = Dis(G_z)
        G_z_loss = criterion(D_G_z, one_labels)
        g_running_loss += G_z_loss.item()
        
        G_z_loss.backward()    
        Gen_opt.step()
        
        if i % checkpoint == 0:
            
            g_current_loss = g_running_loss/checkpoint
            d_current_loss = d_running_loss/checkpoint
            g_losses.append(g_current_loss)
            d_losses.append(d_current_loss)
            print(f"[Generator loss: {g_current_loss}, Discriminator loss: {d_current_loss}]")
            fixed_z_images = Gen(fixed_noise).detach()
            save_progress(fixed_z_images, e, i//checkpoint)
            g_running_loss = 0.
            d_running_loss = 0.
            
    
    plot_losses(g_losses, d_losses)
    view_samples(fixed_z_images)
    save_model_state(Gen, Gen_opt, g_losses, e, "Gen")
    save_model_state(Dis, Dis_opt, d_losses, e, "Dis")