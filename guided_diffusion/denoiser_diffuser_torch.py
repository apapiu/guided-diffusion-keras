
#os.environ["WANDB_API_KEY"]='you_key'
#!wandb login

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch.nn as nn
import torch
import torchvision

from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, IterableDataset
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def diffusion(n_iter=20, class_guidance=3, plot_all_images=True, save_img_name="img"):

    noise_levels = 1 - np.power(np.arange(0.0001, 0.99, 1 / n_iter), 1 / 3)
    noise_levels[-1] = 0.01
    num_imgs = 100

    seeds = torch.randn(num_imgs,3,32,32).to(device)
    new_img = seeds

    n_repetitions = 10  
    digits = torch.arange(0, 10) 
    labels = torch.repeat_interleave(digits, n_repetitions)

    empty_labels = torch.zeros_like(labels)

    labels = torch.cat([labels, empty_labels])

    for i in range(len(noise_levels) - 1):

        curr_noise, next_noise = noise_levels[i], noise_levels[i + 1]

        print(curr_noise)

        noises = torch.full((num_imgs,1), curr_noise)
        noises = torch.cat([noises, noises])

        with torch.no_grad():
            x0_pred = model(torch.cat([new_img, new_img]), 
                            noises.to(device), 
                            labels.to(device)
                            )
            
        x0_pred_label = x0_pred[:num_imgs]
        x0_pred_no_label = x0_pred[num_imgs:]

        # classifier free guidance:
        x0_pred = class_guidance * x0_pred_label + (1 - class_guidance) * x0_pred_no_label

        # new image at next_noise level is a weighted average of old image and predicted x0:
        new_img = ((curr_noise - next_noise) * x0_pred + next_noise * new_img) / curr_noise

        if False:
            plot_img(x0_pred[0].detach().cpu())
            plt.show()

    new_imgs = new_img.cpu().numpy()
     
    if plot_all_images:
        plot_images(new_imgs.transpose(0, 2, 3, 1), nrows=10, save_name=save_img_name)

    return new_imgs

class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_min_frequency=1.0, embedding_max_frequency=1000.0, embedding_dims=32):
        super(SinusoidalEmbedding, self).__init__()

        frequencies = torch.exp(
            torch.linspace(
                torch.log(torch.tensor(embedding_min_frequency)),
                torch.log(torch.tensor(embedding_max_frequency)),
                embedding_dims // 2
            )
        )

        self.register_buffer('angular_speeds', 2.0 * torch.pi * frequencies)

    def forward(self, x):
        angular_speeds = self.angular_speeds
                
        embeddings = torch.cat([torch.sin(angular_speeds * x), 
                                torch.cos(angular_speeds * x)], dim=-1)
        return embeddings

class SpatialAttention(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.proj = nn.Conv2d(c_in, 3*c_in, 1, padding='same') #just projection.
        self.attn = nn.MultiheadAttention(embed_dim = c_in, num_heads = 4, dropout=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        bs, c, h, w = x.size()
        #h*w, c -> attend to various spatial dimensions
        q,k,v = self.proj(x).view(bs, 3*c, h*w).transpose(1,2).chunk(3, dim=2)
        out, _ = self.attn(q,k,v)
        out = out.transpose(1,2).view(bs, c, h, w)

        return self.relu(out) + x

#     def forward(self, x):
#         bs, c, h, w = x.shape()
#         #c, h*w -> attend to various channels
#         q,k,v = self.proj(x).view(bs, c, h*w).chunk(3)
#         out = self.attn(q,k,v)
#         out = out.view(bs, c, h, w)
#         retnr out + x


#residual block:
class ConvBlock(nn.Module):
     def __init__(self, c_in, c_mid, c_out):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(c_in, c_mid, 3, padding='same'), 
                                  nn.ReLU(),
                                  nn.Conv2d(c_mid, c_out, 3, padding='same'), 
                                  nn.ReLU())
        self.resid = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        

     def forward(self, x):
        return self.conv(x) + self.resid(x)


class Denoiser(nn.Module):
    def __init__(self, n_channels, image_size, 
                 noise_embed_dims,
                 class_emb_dims,
                 n_classes):
        super().__init__()

        n = n_channels

        self.scaler = GradScaler()
        self.img_size = image_size
        self.class_emb_dims = class_emb_dims
        self.noise_embed_dims = noise_embed_dims

        self.fourier_feats = SinusoidalEmbedding(noise_embed_dims)

        self.proj = nn.Conv2d(3, n, 1, padding='same')

        self.block1 = nn.Sequential(ConvBlock(noise_embed_dims+class_emb_dims+n, 
                                                   1*n, 1*n),
                                    ConvBlock(1*n, 1*n, 1*n)) #skip+down 32->16

        self.block2 = nn.Sequential(ConvBlock(1*n, 2*n, 2*n), 
                                    ConvBlock(2*n, 2*n, 2*n)) #skip+down 16->8
        
        self.block3 = nn.Sequential(ConvBlock(2*n, 4*n, 4*n), 
                                    ConvBlock(4*n, 4*n, 2*n)) #up+skip
        
        self.block4 = nn.Sequential(ConvBlock(4*n, 2*n, 2*n), 
                                    ConvBlock(2*n, 2*n, 1*n))

        self.block5 = nn.Sequential(ConvBlock(2*n, 1*n, 1*n), #up+skip
                                    ConvBlock(1*n, 1*n, 3))
        
        
        self.avgpool = torch.nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.class_embed = nn.Embedding(n_classes, class_emb_dims)
        self.global_step = 0


        self.loss_fn = nn.MSELoss() 
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)


        
    def forward(self, x, noise_level, label):

        label = self.class_embed(label).view(-1, self.class_emb_dims, 1, 1)
        label = nn.Upsample(scale_factor=self.img_size)(label)

        noise_level = self.fourier_feats(noise_level).view(-1, self.noise_embed_dims, 1, 1)
        noise_level = nn.Upsample(scale_factor=self.img_size)(noise_level)
        
        x = self.proj(x)

        x = torch.cat([x, noise_level, label], dim=1)

        skip1 = self.block1(x)

        skip2 = self.block2(self.avgpool(skip1))

        x = self.block3(self.avgpool(skip2))

        x = self.upsample(x)
        x = self.block4(torch.cat([x, skip2], dim=1))

        x = self.upsample(x)
        x = torch.cat([x, skip1], dim=1) 

        x = self.block5(x)

        return x

    def train_step(self, x, noise_level, label, y):

        with autocast():
                pred = self.forward(x, noise_level, label)
                loss = self.loss_fn(pred, y)

        self.optimizer.zero_grad()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.global_step += 1

        return loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_img(x):
    plt.figure(figsize=(4, 4))
    plt.imshow((x.permute(1,2,0).numpy()+1)/2)

def count_parameters_per_layer(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")

if __name__ == '__main__':

    transform = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)

  
    batch_size = 64
    class_guidance = 3
    n_epochs = 20
    
    n_channels = 64
    image_size = 32
    noise_embed_dims = 32
    class_emb_dims = 64
    n_classes = 11
    
    
    train_loader = DataLoader(trainset, batch_size=batch_size)
    
    model = Denoiser(n_channels, image_size, 
                     noise_embed_dims,
                     class_emb_dims,
                     n_classes)
    
    model.to(device)
    print(count_parameters(model))
    print(count_parameters_per_layer(model))
    
    
    config = {k: v for k, v in locals().items() if k in ['n_epochs', 'n_channels', 'class_guidance', 'n_epochs',
                                                         'image_size', 'noise_embed_dims', 'class_emb_dims',
                                                         'n_classes']}
    
    wandb.init(
        project="cifar_diffusion",
        config = config)
    
    n_epoch = 10
    for i in range(n_epoch):
    
        if i % 2 == 0:
            imgs = diffusion()
            wandb.log({f"epoch {i}": wandb.Image("img.png")})
            checkpoint_path = f"model_checkpoint_{model.global_step}.pth"
            torch.save(model, checkpoint_path)
            wandb.save(checkpoint_path)
    
        for x, y in tqdm(train_loader):
            noise_level = torch.tensor(np.random.beta(1, 2.7, len(x)))
            signal_level = 1 - noise_level
    
            noise = torch.randn_like(x)
    
            x_noisy = noise_level.view(-1,1,1,1)*noise + signal_level.view(-1,1,1,1)*x
    
            x = x.to(device)
            x_noisy = x_noisy.float().to(device)
            noise_level = noise_level.float().to(device)
            y = y.to(device)
    
            #to learn the unconditional prediction:
            if np.random.uniform() > 0.15:
                #leave zero label for unconditional
                label = (y+1)
            else:
                label = torch.zeros_like(y)
    
            loss = model.train_step(x_noisy,  noise_level.view(-1,1), label.view(-1,1), x)
            wandb.log({"train_loss":loss}, step=model.global_step)
    wandb.finish()
