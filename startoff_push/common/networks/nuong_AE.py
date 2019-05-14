import torch.nn as nn
import torch.nn.functional as F

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


class Net_wider(nn.Module):
        def __init__(self):
            super(Net_wider, self).__init__()
             # Encoder
            #There are papers that highlight that deep NN with the right architectures achieve better results 
            # than shallow ones that have the same computational power (e.g. number of neurons or connections). 
            #32x32x3
            self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)#32x32x128
            self.conv1b = nn.Conv2d(128, 128, kernel_size=4, stride=4, padding=2)#9x9x128
            self.conv2 = nn.Conv2d(128, 3200, kernel_size=3, stride=4, padding=1)#3x3x3200
            self.fc1a = nn.Linear(3 * 3 * 3200, 1600)
            self.fc1b = nn.Linear(1600, 180)#2304
            #FC 32x32/240x240= 0,177
            #F*((I-K+2P)/S+1)
            '''# Latent space
            self.fc21 = nn.Linear(128, 20)
            self.fc22 = nn.Linear(128, 20)'''
            # Decoder
            '''self.fc3 = nn.Linear(20, 128)'''

            
            self.fc4a = nn.Linear(180, 1600)
            self.fc4b = nn.Linear(1600, 3*3*3200)#3x3x3200
            self.deconv1 = nn.ConvTranspose2d(3200, 128, kernel_size=3, stride=4, padding=1)#9x9x128
            
            self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4, padding=2)#32x32x128
            self.deconv2b = nn.ConvTranspose2d(128, 3, kernel_size=3, stride=1, padding=1)#32x32x3
            
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.8)
            
            self.batchnorm128 = nn.BatchNorm2d(128)
            self.batchnorm3200 = nn.BatchNorm2d(3200)
            
            ''' for m in self.modules():

            if isinstance(m, nn.Conv2d):

                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels

                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):

                m.weight.data.fill_(1)

                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):

                m.bias.data.zero_()
            '''
        def encode(self, x):

            #do_write("Encoding: Convolution...\n")
            x = F.relu(self.conv1(x))
            x = self.batchnorm128(x)
            x = F.relu(self.conv1b(x))
            #x = self.batchnorm128(x)
            x = self.dropout(x)
            x = F.relu(self.conv2(x))
            x = self.batchnorm3200(x)
            x = self.dropout(x)
       
            x = x.view(x.size(0),-1)
            #pdb.set_trace()

            #do_write("Encoding: FC...\n")
            x = F.relu(self.fc1a(x))
            x = F.relu(self.fc1b(x))
            return x
        
        def normalize(self, x):
            #x_normed = x / x.max(0, keepdim=True)[0] 
            #return x_normed
            alpha=(x-x.mean(0,keepdim=True))
            beta=alpha/x.std(0,keepdim=True)
            return beta
            
        def decode(self, x):
            
            out = self.relu(self.fc4a(x))
            out = self.relu(self.fc4b(out))
            # import pdb; pdb.set_trace()
            out = out.view(out.size(0), 3200, 3, 3)
            out = self.batchnorm3200(out)
            out = self.dropout(out)
            out = self.relu(self.deconv1(out))#9x9x128
            out = self.relu(self.deconv2(out))#32x32x128
            out = self.batchnorm128(out)
            out = self.dropout(out)
            out = self.sigmoid( self.relu(self.deconv2b(out)))
            return out
            
        def forward(self, x):
            mu = self.encode(x)
            #return F.log_softmax(x, dim=1)
            mu_0=self.normalize(mu)
            #return self.decode(mu_0),mu_0
            return self.decode(mu),mu


