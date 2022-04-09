import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt

"""
Set up and load dataLoader
"""



batch_size = 62

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
        index = idx
        path = self.imgs[index][0]
        img = self.loader(path)
        img = self.transform(img)
        return img


"""
Loader must have data in correct order
"""
train_loader = torch.load('3DChairs_loader_rotation')



size = torch.Tensor.size

class DC_IGN(nn.Module):
    def __init__(self, img_channels=1, zdim=200):
        super(DC_IGN, self).__init__()

        first_enc_filter = 96
        second_enc_filter = 64
        third_enc_filter = 32

        self.enc_kernel_size = 5
        self.encConv1 = nn.Conv2d(img_channels, first_enc_filter, self.enc_kernel_size)
        self.encConv2 = nn.Conv2d(first_enc_filter, second_enc_filter, self.enc_kernel_size)
        self.encConv3 = nn.Conv2d(second_enc_filter, third_enc_filter, self.enc_kernel_size)
        self.encFC1 = nn.Linear(32*1*1, zdim)
        self.encFC2 = nn.Linear(32*1*1, zdim)

        first_dec_filter = 32
        second_dec_filter = 64
        third_dec_filter = 96
        dec_kernel_size = 7


        # 7200 is just from reading from the diagram (fig. 1) in dc-ign paper
        self.decFC1 = nn.Linear(zdim, 7200)
        self.decConv1 = nn.Conv2d(7200,first_dec_filter, dec_kernel_size)
        self.decConv2 = nn.Conv2d(first_dec_filter, second_dec_filter, dec_kernel_size)
        self.decConv3 = nn.Conv2d(second_enc_filter, third_dec_filter, dec_kernel_size)
        self.decConv4 = nn.Conv2d(third_dec_filter, img_channels, dec_kernel_size)

    def encoder(self, x):
        x= F.max_pool2d(self.encConv1(x), self.enc_kernel_size)
        x = F.max_pool2d(self.encConv2(x), self.enc_kernel_size)
        x = F.relu(self.encConv3(x))
        x = x.view(batch_size,1,32)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparametrize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        x = self.decFC1(z)
        x = x.view(batch_size, 32, 15, 15)
        x = F.interpolate(self.decConv2(x), (30,30), mode='nearest') #equivalent to upsampling with nearest neighbour
        x = F.interpolate(self.decConv3(x), (156,156), mode='nearest')
        x = torch.sigmoid(self.decConv4(x))
        return x
    


    def adjust_encode(self, z, scene_variable):
        # Change input to decoder to force neurons in graphics code to learn correct features
        means = torch.mean(z, axis=0)
        z_new = torch.clone(z)
        if scene_variable == 'rotation':
            for i in range(size(z)[0]):
                z_new[i][0] = means
                z_new[i][0][0] = z[i][0][0]
        elif scene_variable == 'intrinsic':
            for i in range(size(z)[0]):
                z_new[i][0] = z[i][0]
                z_new[i][0][0] = means[0][0]
        else:
            print('This should not be printed during training')
        return z_new

    def forward(self, x, scene_variable):
        mu, logVar = self.encoder(x)
        z = self.reparametrize(mu, logVar)
        z = self.adjust_encode(z, scene_variable)
        out = self.decoder(z)
        return out, mu, logVar


"""
Determine if any GPUs are available and set hyperparameters
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = DC_IGN().to(device)
num_epochs = 1
learning_rate = 5e-4
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)




"""
Adjust gradients of mean and logVar according to the paper
"""


def fix_grads_rotation(param):
    mean = torch.mean(param)
    new_vec = param.grad
    for i in range(1,200):
        new_vec[i] = (param[i] - mean)/100
    new_vec -= param.grad
    return new_vec

def fix_grads_intrinsic(param):
    mean = torch.mean(param)
    new_vec = param.grad
    new_vec[0] = (param[0] - mean)/100
    new_vec -= param.grad
    return new_vec







net.load_state_dict(torch.load('original_DC-IGN_checkpoint'))

num_epochs=0

for epoch in range(num_epochs):
    for idx, data in enumerate(train_loader, 0):
        print(f'index = {idx}')
        if idx >= 0:

            
            # 90% batches change only intrinsic and 10% rotation as in paper
        
            if idx%68 < 62:
                scene_variable = 'intrinsic'
                fix_grads = fix_grads_intrinsic
            else:
                scene_variable = 'rotation'
                fix_grads = fix_grads_rotation

            imgs = data.to(device)
            

            out, mu, logVar = net(imgs, scene_variable)
            


            # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
            kl_divergence = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
            print(f'kl divergence = {kl_divergence}')
            loss = F.binary_cross_entropy(out, imgs, reduction='sum') + kl_divergence
            print(f'loss = {loss}\n')

            # Backpropagation based on the loss
            optimizer.zero_grad()
            
            # adjust gradients as in paper - current implementation is inefficient

            loss.backward(retain_graph=True)
            mean_grad = fix_grads(list(net.parameters())[7])
            logVar_grad = fix_grads(list(net.parameters())[9])
            optimizer.zero_grad()
            list(net.parameters())[7].grad = mean_grad
            list(net.parameters())[9].grad = logVar_grad
            loss.backward(retain_graph=True) 
            optimizer.step()

    print('Epoch {}: Loss {}'.format(epoch, loss))

torch.save(net.state_dict(), 'DC-IGN_checkpoint')





"""
View sample results
"""


net.eval()
i = 0
with torch.no_grad():
    for imgs in train_loader:
        imgs = imgs.to(device)
        img = np.transpose(imgs[2].cpu().numpy(), [1,2,0])
        plt.subplot(121)
        plt.imshow(np.squeeze(img), cmap='gray')
        plt.show()
        out, mu, logVAR = net(imgs, 'test')
        mse_loss = F.mse_loss(out, imgs, reduction='sum')
        print(f'mse loss ={mse_loss}')
        outimg = np.transpose(out[2].cpu().numpy(), [1,2,0])
        plt.subplot(122)
        plt.imshow(np.squeeze(outimg), cmap='gray')
        plt.show()
        i += 1
        # stop viewing after 10 chairs
        if i > 10: 
            break