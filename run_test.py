import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

from torch.autograd import Variable

import os
import numpy as np
import pdb
import warnings
warnings.filterwarnings("ignore")


class arg():
    seed=1
    no_cuda=True

    batch_size=64
    intermediate_size=128 #usual hidden size, linear around z
    hidden_size=30 # latent space z
    test_batch_size=100
    epochs=10
    lr=1e-1 #0.001
    momentum=0.5
    log_interval=10
    save_model=True
    experiment=2
        
    if not os.path.exists(f"./exp{experiment}"):
        os.makedirs(f"./exp{experiment}")
        os.makedirs(f"./exp{experiment}/data")
        open(f'./exp{experiment}/logfile.txt', 'w+').close()
    else:
        open(f'./exp{experiment}/logfile.txt', 'w+').close()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def do_write(string):
    f=open(f'./exp2/logfile.txt','a+')
    f.write(string)
    f.close()

def main4(load_old=False):
    
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            '''old: self.conv1 = nn.Conv2d(1, 20, 5, 1)#in out kernel_sz stride
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4 * 4 * 50, 500) # in out
            self.fc2 = nn.Linear(500, 10)'''
             # Encoder
            #32x32x3
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)#32x32x32
            self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=4, padding=0)#8x8x32
            self.conv3 = nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1)#8x8x96
            self.conv3b = nn.Conv2d(96, 96, kernel_size=2, stride=3, padding=0)#3x3x96
            self.conv4 = nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1)#3x3x256
            self.fc1a = nn.Linear(3 * 3 * 256, 180)#2304
            #FC 32x32/240x240= 0,177
            #F*((I-K+2P)/S+1)
            '''# Latent space
            self.fc21 = nn.Linear(128, 20)
            self.fc22 = nn.Linear(128, 20)'''
            # Decoder
            '''self.fc3 = nn.Linear(20, 128)'''

            self.fc4a = nn.Linear(180, 2304)
            self.deconv1 = nn.ConvTranspose2d(256, 96, kernel_size=3, stride=1, padding=1)
            self.deconv1b = nn.ConvTranspose2d(96, 96, kernel_size=2, stride=3, padding=0)
            self.deconv2 = nn.ConvTranspose2d(96, 32, kernel_size=3, stride=1, padding=1)
            self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=4, padding=0)
            self.conv5 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
            
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            
        def encode(self, x):

            #do_write("Encoding: Convolution...\n")
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv3b(x))
            x = F.relu(self.conv4(x))
       
            x = x.view(x.size(0),-1)
            #pdb.set_trace()

            #do_write("Encoding: FC...\n")
            x = F.relu(self.fc1a(x))
            
            return x
        
        def normalize(self, x):
            #x_normed = x / x.max(0, keepdim=True)[0] 
            #return x_normed
            alpha=(x-x.mean(0,keepdim=True))
            beta=alpha/x.std(0,keepdim=True)
            return beta
            
        def decode(self, x):
            #do_write("De-coding: FC...\n")
            out = self.relu(self.fc4a(x))
            # import pdb; pdb.set_trace()
            out = out.view(out.size(0), 256, 3, 3)

            #do_write("De-coding: Deconvolution...\n")
            out = self.relu(self.deconv1(out))
            out = self.relu(self.deconv1b(out))
            out = self.relu(self.deconv2(out))
            out = self.relu(self.deconv3(out))
            out = self.sigmoid(self.conv5(out))
            return out
            
        def forward(self, x):
            mu = self.encode(x)
            #return F.log_softmax(x, dim=1)
            mu_0=self.normalize(mu)
            #return self.decode(mu_0),mu_0
            return self.decode(mu),mu
        
    '''def loss_function(recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x.view(-1, 32 * 32 * 3),
                                 x.view(-1, 32 * 32 * 3), size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #kullbach-leibler divergence
        return BCE + KLD'''
    
    def loss_function(recon_x, x):
        #pdb.set_trace()
        BCE = F.binary_cross_entropy(recon_x.view(-1, 32 * 32 * 3),
                                 x.view(-1, 32 * 32 * 3), size_average=False)
        return BCE 
    
    def train(args, model, device, train_loader, optimizer, epoch):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)[0]

            #do_write("Calc Loss...\n")
            loss = loss_function(output, data)
            loss.backward()
            
            #loss = F.nll_loss(output, target)
            #loss.backward()
            
            #train_loss += loss.data[0]
            train_loss += loss.item()
            
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                do_write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item() / len(data)))
        
        do_write('====> Epoch: {} Average loss: {:.4f}\n'.format(epoch, train_loss / len(train_loader.dataset)))


        # save the reconstructed images
        reconst_images = model(args.fixed_x)[0]
        
        ##
        #print(my_variable.data.cpu().numpy())
        #x = Variable()
        #print(np.shape(reconst_images.data.cpu().numpy()[0]))
        reconst_images = reconst_images.view(reconst_images.size(0), 3, 32, 32)
        save_image(reconst_images.data.cpu(), f'./exp{args.experiment}/data/CIFAR_reconst_images_%d.png' % (epoch))

    def test(args, model, device, test_loader):
        model.eval()
        test_loss = 0
        #correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)[0]
                #nochma angucken
                #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss, negative log likelihood loss.
                
                test_loss += loss_function(output, data).item()
                #pdb.set_trace()
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                #correct += pred.eq(target.view_as(pred)).sum().item() #accuracy, elementwise equality, and sum

        test_loss /= len(test_loader.dataset)


        do_write('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    
    args=arg()
        

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True,
                      transform=transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    
    
    device = torch.device("cuda" if use_cuda else "cpu")
    
    if load_old:
        model=Net().to(device)
        model.load_state_dict(torch.load(f"exp{args.experiment}/cifar_cnn_5.pt"))
    else:
        model = Net().to(device)
    
    #model = Net()
    #if not args.no_cuda:
    #    model.cuda()
        
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    #gradients tend to vanish or explode
    '''It uses a moving average of squared gradients to normalize the gradient itself. 
    That has an effect of balancing the step size?—?decrease the step for large gradient 
    to avoid exploding, and 
    increase the step for small gradient to avoid vanishing'''
    
    
    #how to save fixed inputs for debugging
    data_iter = iter(train_loader)
    fixed_x, _ = next(data_iter)
    #pdb.set_trace()
    save_image(Variable(fixed_x).data.cpu(), f'./exp{args.experiment}/data/CIFAR_real_images.png')
    args.fixed_x = to_var(fixed_x) 
    #args.fixed_x = to_var(fixed_x.view(fixed_x.size(0), -1)) 
    #args.fixed_x=args.fixed_x.to(device)
    #print(np.shape(args.fixed_x.data.cpu().numpy()))
    
    for epoch in range(1, args.epochs + 1):

        
        do_write("in epoch")
        do_write(f"{epoch}\n")
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

        if (args.save_model):
            torch.save(model.state_dict(), f"exp{args.experiment}/cifar_cnn_{epoch}.pt")

    args.f.close()

if __name__ == '__main__':
    main4(load_old=False)
