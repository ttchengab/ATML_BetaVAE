"""
Import necessary libraries to create a variational autoencoder
The code is mainly developed using the PyTorch library
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from torch.utils.data import Dataset, DataLoader

"""
The following part takes a random image from test loader to feed into the VAE.
Both the original image and generated image from the distribution are shown.
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import glob

avgpool = nn.AdaptiveAvgPool2d((128, 128))
class threeDChairs(Dataset):
    def __init__(self, train, transform=None, target_transform=None):
        self.img_files = glob.glob('rendered_chairs/*/renders/*.png')
        self.add = 0
        if not train:
            self.add = 1000

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        image = cv2.imread(self.img_files[idx+self.add])/255
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1)
        image = avgpool(image)
        return image


print("chairs20")
"""
Determine if any GPUs are available
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
Initialize Hyperparameters
"""
batch_size = 128
learning_rate = 1e-3
num_epochs = 100


"""
Create dataloaders to feed data into the neural network
Default MNIST dataset is used and standard train/test split is performed
"""

train_set = threeDChairs(train=True)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = threeDChairs(train=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

"""
A Convolutional Variational Autoencoder
"""
class VAE(nn.Module):
    def __init__(self, imgChannels=3, featureDim=32*120*120, hiddenDim=256, zDim=20):
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 16, 5)
        self.encConv2 = nn.Conv2d(16, 32, 5)
        self.encFC = nn.Linear(featureDim, hiddenDim)
        self.encFC1 = nn.Linear(hiddenDim, zDim)
        self.encFC2 = nn.Linear(hiddenDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, hiddenDim)
        self.decFC2 = nn.Linear(hiddenDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(32, 16, 5)
        self.decConv2 = nn.ConvTranspose2d(16, imgChannels, 5)
        self.training = True

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        # print("CURRENT SHAPE = {}".format(x.shape))
        x = x.view(-1, 32*120*120)

        x = F.relu(self.encFC(x))
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = F.relu(self.decFC2(x))
        x = x.view(-1, 32, 120, 120)
        x = F.relu(self.decConv1(x))
        x = torch.sigmoid(self.decConv2(x))
        # print("OUT SHAPE = {}".format(x.shape))
        return x

    def forward(self, x, traversal):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        if self.training:
            z = self.reparameterize(mu, logVar)
        else:
            # (1, 256)
            mu = mu.squeeze(0)
            mu[-1] += traversal
            z = mu.unsqueeze(0)
        out = self.decoder(z)
        return out, mu, logVar



"""
Initialize the network and the Adam optimizer
"""
net = VAE().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
beta = 6


"""
Training the network for a given number of epochs
The loss after every epoch is printed
"""
for epoch in range(num_epochs):
    net.training = True
    for idx, data in enumerate(train_loader, 0):
        imgs = data
        imgs = imgs.to(device)

        # Feeding a batch of images into the network to obtain the output image, mu, and logVar
        out, mu, logVar = net(imgs, 0)

        # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
        kl_divergence = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
        loss = F.binary_cross_entropy(out, imgs, size_average=False) + beta * kl_divergence

        # Backpropagation based on the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch {}: Loss {}'.format(epoch, loss))

    if epoch == 0 or ((epoch + 1) % 25 == 0):
        net.eval()
        net.training = False
        torch.save(net.state_dict(), 'beta/MNISTTTT20{}Epoch_{}.pt'.format(beta, epoch))
        with torch.no_grad():
            for data in random.sample(list(test_loader), 1):
                imgs = data
                imgs = imgs.to(device)

                plt.subplot(151)
                out, mu, logVAR = net(imgs[0].unsqueeze(0), -3)
                outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])
                # plt.imshow(np.squeeze(outimg), cmap="gray")
                plt.imshow(cv2.cvtColor(np.squeeze(outimg), cv2.COLOR_BGR2RGB))

                plt.subplot(152)
                out, mu, logVAR = net(imgs[0].unsqueeze(0), -1.5)
                outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])
                # plt.imshow(np.squeeze(outimg), cmap="gray")
                plt.imshow(cv2.cvtColor(np.squeeze(outimg), cv2.COLOR_BGR2RGB))

                plt.subplot(153)
                out, mu, logVAR = net(imgs[0].unsqueeze(0), 0)
                outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])
                # plt.imshow(np.squeeze(outimg), cmap="gray")
                plt.imshow(cv2.cvtColor(np.squeeze(outimg), cv2.COLOR_BGR2RGB))

                plt.subplot(154)
                out, mu, logVAR = net(imgs[0].unsqueeze(0), 1.5)
                outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])
                plt.imshow(np.squeeze(outimg), cmap="gray")
                plt.imshow(cv2.cvtColor(np.squeeze(outimg), cv2.COLOR_BGR2RGB))

                plt.subplot(155)
                out, mu, logVAR = net(imgs[0].unsqueeze(0), 3)
                outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])
                # plt.imshow(np.squeeze(outimg), cmap="gray")
                plt.imshow(cv2.cvtColor(np.squeeze(outimg), cv2.COLOR_BGR2RGB))
                plt.savefig('beta/CHAIRS128{}Epoch_{}.png'.format(beta, epoch))
