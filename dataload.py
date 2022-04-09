from tkinter import image_names
from xml.dom import INDEX_SIZE_ERR
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import sys


batch_size = 62
image_size = 150



class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)



    def get_index(self, idx):
        if idx > 84319:
            return idx
        j, i = divmod(idx, 4216)
        if i > 3843:
            return idx
        else:
            q,r = divmod(i, 62)
            return j*4216 +62*r + q
    

    def __getitem__(self, idx):
        index = self.get_index(idx)
        path = self.imgs[index][0]
        img = self.loader(path)
        img = self.transform(img)
        return img



def return_data(dset):
    # convert images to 150x150 pixels and greyscale as in DC-IGN paper
    dset_dir = 'data'
    root = os.path.join(dset_dir, dset)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1)])
    train_kwargs = {'root':root, 'transform':transform}
    dset = CustomImageFolder
    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=False,
                              drop_last=False)

    data_loader = train_loader
    print(f'data loader length = {len(data_loader)}') #sanity check
    return data_loader

train_loader = return_data('3DChairs_rotation')

torch.save(train_loader, '3DChairs_loader')



"""
Check it has loaded correctly
"""

train_loader = torch.load('3DChairs_loader')


i = 0 
j = 0
for data in train_loader:
    for img in data:
        print(f'size = {torch.Tensor.size(data)}')
        img = np.transpose(img.cpu().numpy(), [1,2,0])
        plt.subplot(121)
        plt.imshow(np.squeeze(img), cmap='gray')
        plt.show()
        i += 1
        if i > 1:
            i = 0
            break
    j += 1
    if j > 1:
        break







