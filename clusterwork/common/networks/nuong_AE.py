import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)#32x32x32
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=4, padding=0)#8x8x32
        self.conv3 = nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1)#8x8x96
        self.conv3b = nn.Conv2d(96, 96, kernel_size=2, stride=3, padding=0)#3x3x96
        self.conv4 = nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1)#3x3x256
        self.fc1a = nn.Linear(3 * 3 * 256, 180)


        self.fc4a = nn.Linear(180, 2304)
        self.deconv1 = nn.ConvTranspose2d(256, 96, kernel_size=3, stride=1, padding=1)
        self.deconv1b = nn.ConvTranspose2d(96, 96, kernel_size=2, stride=3, padding=0)
        self.deconv2 = nn.ConvTranspose2d(96, 32, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=4, padding=0)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
            
        self.batchnorm32 = nn.BatchNorm2d(32)
        self.batchnorm96 = nn.BatchNorm2d(96)
        
    def encode(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.batchnorm32(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv3b(x))
        x = self.batchnorm96(x)
        x = self.dropout(x)
        x = F.relu(self.conv4(x))
       
        x = x.view(x.size(0),-1)

        x = F.relu(self.fc1a(x))
        
        return x
       
        
    def decode(self, x):
        out = self.relu(self.fc4a(x))
        out = out.view(out.size(0), 256, 3, 3)

        out = self.relu(self.deconv1(out))
        out = self.relu(self.deconv1b(out))
        out = self.batchnorm96(out)
        out = self.dropout(out)
        out = self.relu(self.deconv2(out))
        out = self.relu(self.deconv3(out))
        out = self.batchnorm32(out)
        out = self.dropout(out)
        out = self.sigmoid(self.conv5(out))
        return out
        
    def forward(self, x):
        mu = self.encode(x)
        return self.decode(mu),mu



class Net_deepwide_do(nn.Module):
        def __init__(self):
            super(Net_deepwide_do, self).__init__()

            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)#32x32x32
            self.conv1b = nn.Conv2d(32, 32, kernel_size=4, stride=4, padding=2)#9x9x32
            self.conv1c = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)#9x9x128
            self.conv2 = nn.Conv2d(128, 1024, kernel_size=3, stride=1, padding=1)#9x9x1024
            self.conv2b = nn.Conv2d(1024, 1024, kernel_size=3, stride=4, padding=1)#3x3x1024
            self.conv2c = nn.Conv2d(1024, 4096, kernel_size=3, stride=1, padding=1)#3x3x4096
            self.fc1a = nn.Linear(3 * 3 * 4096, 4608)
            self.fc1b = nn.Linear(4608, 1152)
            self.fc1c = nn.Linear(1152, 144)

            '''# Latent space
            self.fc21 = nn.Linear(128, 20)
            self.fc22 = nn.Linear(128, 20)'''
            # Decoder
            '''self.fc3 = nn.Linear(20, 128)'''

            
            self.fc4a = nn.Linear(144, 1152)
            self.fc4b = nn.Linear(1152,4608)
            self.fc4c = nn.Linear(4608, 3 * 3 * 4096)
            
            self.deconv1 = nn.ConvTranspose2d(4096, 1024, kernel_size=3, stride=1, padding=1)#3x3x1024
            self.deconv1b = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=4, padding=1)#9x9x1024
            self.deconv1c = nn.ConvTranspose2d(1024, 128, kernel_size=3, stride=1, padding=1)#9x9x128
            self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=1, padding=1)#9x9x32
            self.deconv2b = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=4, padding=2)#32x32x32
            self.deconv2c = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)#
            
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.5)
            
            self.batchnorm128 = nn.BatchNorm2d(128)
            self.batchnorm1024 = nn.BatchNorm2d(1024)
            self.batchnorm4096 = nn.BatchNorm2d(4096)
            self.batchnorm32 = nn.BatchNorm2d(32)
            
        def encode(self, x):

            #do_write("Encoding: Convolution...\n")
            x = F.relu(self.conv1(x))
            #x = self.batchnorm32(x)
            x = F.relu(self.conv1b(x))
            #x = self.batchnorm32(x)
            x = F.relu(self.conv1c(x))
            #x = self.batchnorm128(x)
            x = self.dropout(x)
            x = F.relu(self.conv2(x))
            #x = self.batchnorm1024(x)
            x = F.relu(self.conv2b(x))
            #x = self.batchnorm1024(x)
            x = F.relu(self.conv2c(x))
            #x = self.batchnorm4096(x)
            x = self.dropout(x)
       
            x = x.view(x.size(0),-1)
            #pdb.set_trace()

            #do_write("Encoding: FC...\n")
            x = F.relu(self.fc1a(x))
            x = F.relu(self.fc1b(x))

            x = F.relu(self.fc1c(x))
            return x

        def decode(self, x):
            
            out = self.relu(self.fc4a(x))
            out = self.relu(self.fc4b(out))
            out = self.relu(self.fc4c(out))
            # import pdb; pdb.set_trace()
            out = out.view(out.size(0), 4096, 3, 3)
            #out = self.batchnorm4096(out)
            out = self.relu(self.deconv1(out))#3x3x1024
            #out = self.batchnorm1024(out)
            out = self.relu(self.deconv1b(out))
            #out = self.batchnorm1024(out)
            out = self.dropout(out)
            out = self.relu(self.deconv1c(out))#9x9x128
            #out = self.batchnorm128(out)
            out = self.relu(self.deconv2(out))#9x9x32
            #out = self.batchnorm32(out)
            out = self.relu(self.deconv2b(out))#32x32x32
            #out = self.batchnorm32(out)
            out = self.dropout(out)
            out = self.sigmoid( self.relu(self.deconv2c(out)))#32x32x3
            return out
            
        def forward(self, x):
            mu = self.encode(x)

            return self.decode(mu),mu

class Net_deepwide(nn.Module):
        def __init__(self):
            super(Net_deepwide, self).__init__()

            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)#32x32x32
            self.conv1b = nn.Conv2d(32, 32, kernel_size=4, stride=4, padding=2)#9x9x32
            self.conv1c = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)#9x9x128
            self.conv2 = nn.Conv2d(128, 1024, kernel_size=3, stride=1, padding=1)#9x9x1024
            self.conv2b = nn.Conv2d(1024, 1024, kernel_size=3, stride=4, padding=1)#3x3x1024
            self.conv2c = nn.Conv2d(1024, 4096, kernel_size=3, stride=1, padding=1)#3x3x4096
            self.fc1a = nn.Linear(3 * 3 * 4096, 4608)
            self.fc1b = nn.Linear(4608, 1152)
            self.fc1c = nn.Linear(1152, 144)

            '''# Latent space
            self.fc21 = nn.Linear(128, 20)
            self.fc22 = nn.Linear(128, 20)'''
            # Decoder
            '''self.fc3 = nn.Linear(20, 128)'''

            
            self.fc4a = nn.Linear(144, 1152)
            self.fc4b = nn.Linear(1152,4608)
            self.fc4c = nn.Linear(4608, 3 * 3 * 4096)
            
            self.deconv1 = nn.ConvTranspose2d(4096, 1024, kernel_size=3, stride=1, padding=1)#3x3x1024
            self.deconv1b = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=4, padding=1)#9x9x1024
            self.deconv1c = nn.ConvTranspose2d(1024, 128, kernel_size=3, stride=1, padding=1)#9x9x128
            self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=1, padding=1)#9x9x32
            self.deconv2b = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=4, padding=2)#32x32x32
            self.deconv2c = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)#
            
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.5)
            
            #self.batchnorm128 = nn.BatchNorm2d(128)
            #self.batchnorm1024 = nn.BatchNorm2d(1024)
            #self.batchnorm4096 = nn.BatchNorm2d(4096)
            #self.batchnorm32 = nn.BatchNorm2d(32)
            
        def encode(self, x):

            #do_write("Encoding: Convolution...\n")
            x = F.relu(self.conv1(x))
            #x = self.batchnorm32(x)
            x = F.relu(self.conv1b(x))
            #x = self.batchnorm32(x)
            x = F.relu(self.conv1c(x))
            #x = self.batchnorm128(x)
            #x = self.dropout(x)
            x = F.relu(self.conv2(x))
            #x = self.batchnorm1024(x)
            x = F.relu(self.conv2b(x))
            #x = self.batchnorm1024(x)
            x = F.relu(self.conv2c(x))
            #x = self.batchnorm4096(x)
            #x = self.dropout(x)
       
            x = x.view(x.size(0),-1)

            x = F.relu(self.fc1a(x))
            x = F.relu(self.fc1b(x))

            x = F.relu(self.fc1c(x))
            return x

        def decode(self, x):
            
            out = self.relu(self.fc4a(x))
            out = self.relu(self.fc4b(out))
            out = self.relu(self.fc4c(out))
            # import pdb; pdb.set_trace()
            out = out.view(out.size(0), 4096, 3, 3)
            #out = self.batchnorm4096(out)
            out = self.relu(self.deconv1(out))#3x3x1024
            #out = self.batchnorm1024(out)
            out = self.relu(self.deconv1b(out))
            #out = self.batchnorm1024(out)
            #out = self.dropout(out)
            out = self.relu(self.deconv1c(out))#9x9x128
            #out = self.batchnorm128(out)
            out = self.relu(self.deconv2(out))#9x9x32
            #out = self.batchnorm32(out)
            out = self.relu(self.deconv2b(out))#32x32x32
            #out = self.batchnorm32(out)
            #out = self.dropout(out)
            out = self.sigmoid( self.relu(self.deconv2c(out)))#32x32x3
            return out
            
        def forward(self, x):
            mu = self.encode(x)

            return self.decode(mu),mu


class Net_deepwide_enc_do(nn.Module):
        def __init__(self):
            super(Net_deepwide_enc_do, self).__init__()
             # Encoder
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)#32x32x32
            self.conv1b = nn.Conv2d(32, 32, kernel_size=4, stride=4, padding=2)#9x9x32
            self.conv1c = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)#9x9x128
            self.conv2 = nn.Conv2d(128, 1024, kernel_size=3, stride=1, padding=1)#9x9x1024
            self.conv2b = nn.Conv2d(1024, 1024, kernel_size=3, stride=4, padding=1)#3x3x1024
            self.conv2c = nn.Conv2d(1024, 4096, kernel_size=3, stride=1, padding=1)#3x3x4096
            self.fc1a = nn.Linear(3 * 3 * 4096, 4608)
            self.fc1b = nn.Linear(4608, 1152)
            self.fc1c = nn.Linear(1152, 144)

            '''# Latent space
            self.fc21 = nn.Linear(128, 20)
            self.fc22 = nn.Linear(128, 20)'''
            # Decoder
            '''self.fc3 = nn.Linear(20, 128)'''

            
            self.fc4a = nn.Linear(144, 1152)
            self.fc4b = nn.Linear(1152,4608)
            self.fc4c = nn.Linear(4608, 3 * 3 * 4096)
            
            self.deconv1 = nn.ConvTranspose2d(4096, 1024, kernel_size=3, stride=1, padding=1)#3x3x1024
            self.deconv1b = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=4, padding=1)#9x9x1024
            self.deconv1c = nn.ConvTranspose2d(1024, 128, kernel_size=3, stride=1, padding=1)#9x9x128
            self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=1, padding=1)#9x9x32
            self.deconv2b = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=4, padding=2)#32x32x32
            self.deconv2c = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)#
            
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.5)
            
            self.batchnorm128 = nn.BatchNorm2d(128)
            self.batchnorm1024 = nn.BatchNorm2d(1024)
            self.batchnorm4096 = nn.BatchNorm2d(4096)
            self.batchnorm32 = nn.BatchNorm2d(32)
            
        def encode(self, x):

            x = F.relu(self.conv1(x))
            x = self.batchnorm32(x)
            x = F.relu(self.conv1b(x))
            x = self.batchnorm32(x)
            x = F.relu(self.conv1c(x))
            x = self.batchnorm128(x)
            x = self.dropout(x)
            x = F.relu(self.conv2(x))
            x = self.batchnorm1024(x)
            x = F.relu(self.conv2b(x))
            x = self.batchnorm1024(x)
            x = F.relu(self.conv2c(x))
            x = self.batchnorm4096(x)
            x = self.dropout(x)
       
            x = x.view(x.size(0),-1)

            x = F.relu(self.fc1a(x))
            x = F.relu(self.fc1b(x))

            x = F.relu(self.fc1c(x))
            return x

        def decode(self, x):
            
            out = self.relu(self.fc4a(x))
            out = self.relu(self.fc4b(out))
            out = self.relu(self.fc4c(out))
            out = out.view(out.size(0), 4096, 3, 3)
            out = self.batchnorm4096(out)
            out = self.relu(self.deconv1(out))#3x3x1024
            out = self.batchnorm1024(out)
            out = self.relu(self.deconv1b(out))
            out = self.batchnorm1024(out)
            #out = self.dropout(out)
            out = self.relu(self.deconv1c(out))#9x9x128
            out = self.batchnorm128(out)
            out = self.relu(self.deconv2(out))#9x9x32
            out = self.batchnorm32(out)
            out = self.relu(self.deconv2b(out))#32x32x32
            out = self.batchnorm32(out)
            #out = self.dropout(out)
            out = self.sigmoid( self.relu(self.deconv2c(out)))#32x32x3
            return out
            
        def forward(self, x):
            mu = self.encode(x)

            return self.decode(mu),mu



class Net_deepwide_dec_do(nn.Module):
        def __init__(self):
            super(Net_deepwide_dec_do, self).__init__()
             # Encoder

            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)#32x32x32
            self.conv1b = nn.Conv2d(32, 32, kernel_size=4, stride=4, padding=2)#9x9x32
            self.conv1c = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)#9x9x128
            self.conv2 = nn.Conv2d(128, 1024, kernel_size=3, stride=1, padding=1)#9x9x1024
            self.conv2b = nn.Conv2d(1024, 1024, kernel_size=3, stride=4, padding=1)#3x3x1024
            self.conv2c = nn.Conv2d(1024, 4096, kernel_size=3, stride=1, padding=1)#3x3x4096
            self.fc1a = nn.Linear(3 * 3 * 4096, 4608)
            self.fc1b = nn.Linear(4608, 1152)
            self.fc1c = nn.Linear(1152, 144)

            '''# Latent space
            self.fc21 = nn.Linear(128, 20)
            self.fc22 = nn.Linear(128, 20)'''
            # Decoder
            '''self.fc3 = nn.Linear(20, 128)'''

            
            self.fc4a = nn.Linear(144, 1152)
            self.fc4b = nn.Linear(1152,4608)
            self.fc4c = nn.Linear(4608, 3 * 3 * 4096)
            
            self.deconv1 = nn.ConvTranspose2d(4096, 1024, kernel_size=3, stride=1, padding=1)#3x3x1024
            self.deconv1b = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=4, padding=1)#9x9x1024
            self.deconv1c = nn.ConvTranspose2d(1024, 128, kernel_size=3, stride=1, padding=1)#9x9x128
            self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=1, padding=1)#9x9x32
            self.deconv2b = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=4, padding=2)#32x32x32
            self.deconv2c = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)#
            
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.5)
            
            self.batchnorm128 = nn.BatchNorm2d(128)
            self.batchnorm1024 = nn.BatchNorm2d(1024)
            self.batchnorm4096 = nn.BatchNorm2d(4096)
            self.batchnorm32 = nn.BatchNorm2d(32)
            
        def encode(self, x):

            x = F.relu(self.conv1(x))
            x = self.batchnorm32(x)
            x = F.relu(self.conv1b(x))
            x = self.batchnorm32(x)
            x = F.relu(self.conv1c(x))
            x = self.batchnorm128(x)
            x = self.dropout(x)
            x = F.relu(self.conv2(x))
            x = self.batchnorm1024(x)
            x = F.relu(self.conv2b(x))
            x = self.batchnorm1024(x)
            x = F.relu(self.conv2c(x))
            x = self.batchnorm4096(x)
            x = self.dropout(x)
       
            x = x.view(x.size(0),-1)

            x = F.relu(self.fc1a(x))
            x = F.relu(self.fc1b(x))

            x = F.relu(self.fc1c(x))
            return x

        def decode(self, x):
            
            out = self.relu(self.fc4a(x))
            out = self.relu(self.fc4b(out))
            out = self.relu(self.fc4c(out))
            out = out.view(out.size(0), 4096, 3, 3)
            out = self.batchnorm4096(out)
            out = self.relu(self.deconv1(out))#3x3x1024
            out = self.batchnorm1024(out)
            out = self.relu(self.deconv1b(out))
            out = self.batchnorm1024(out)
            out = self.dropout(out)
            out = self.relu(self.deconv1c(out))#9x9x128
            out = self.batchnorm128(out)
            out = self.relu(self.deconv2(out))#9x9x32
            out = self.batchnorm32(out)
            out = self.relu(self.deconv2b(out))#32x32x32
            out = self.batchnorm32(out)
            out = self.dropout(out)
            out = self.sigmoid( self.relu(self.deconv2c(out)))
            return out
            
        def forward(self, x):
            mu = self.encode(x)
            return self.decode(mu),mu

class Net_wide(nn.Module):
        def __init__(self):
            super(Net_wide, self).__init__()
             # Encoder

            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)#32x32x32
            self.conv1b = nn.Conv2d(32, 32, kernel_size=4, stride=4, padding=2)#9x9x32
            self.conv1c = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)#9x9x128
            self.conv2 = nn.Conv2d(128, 1024, kernel_size=3, stride=1, padding=1)#9x9x1024
            self.conv2b = nn.Conv2d(1024, 1024, kernel_size=3, stride=4, padding=1)#3x3x1024
            self.conv2c = nn.Conv2d(1024, 4096, kernel_size=3, stride=1, padding=1)#3x3x4096
            self.fc1a = nn.Linear(3 * 3 * 4096, 4608)
            #self.fc1b = nn.Linear(4608, 1152)
            #self.fc1c = nn.Linear(1152, 144)

            '''# Latent space
            self.fc21 = nn.Linear(128, 20)
            self.fc22 = nn.Linear(128, 20)'''
            # Decoder
            '''self.fc3 = nn.Linear(20, 128)'''

            
            #self.fc4a = nn.Linear(144, 1152)
            #self.fc4b = nn.Linear(1152,4608)
            self.fc4c = nn.Linear(4608, 3 * 3 * 4096)
            
            self.deconv1 = nn.ConvTranspose2d(4096, 1024, kernel_size=3, stride=1, padding=1)#3x3x1024
            self.deconv1b = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=4, padding=1)#9x9x1024
            self.deconv1c = nn.ConvTranspose2d(1024, 128, kernel_size=3, stride=1, padding=1)#9x9x128
            self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=1, padding=1)#9x9x32
            self.deconv2b = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=4, padding=2)#32x32x32
            self.deconv2c = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)#
            
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.5)
            
            self.batchnorm128 = nn.BatchNorm2d(128)
            self.batchnorm1024 = nn.BatchNorm2d(1024)
            self.batchnorm4096 = nn.BatchNorm2d(4096)
            self.batchnorm32 = nn.BatchNorm2d(32)
            
        def encode(self, x):

            x = F.relu(self.conv1(x))
            x = self.batchnorm32(x)
            x = F.relu(self.conv1b(x))
            x = self.batchnorm32(x)
            x = F.relu(self.conv1c(x))
            x = self.batchnorm128(x)
            x = self.dropout(x)
            x = F.relu(self.conv2(x))
            x = self.batchnorm1024(x)
            x = F.relu(self.conv2b(x))
            x = self.batchnorm1024(x)
            x = F.relu(self.conv2c(x))
            x = self.batchnorm4096(x)
            x = self.dropout(x)
       
            x = x.view(x.size(0),-1)

            x = F.relu(self.fc1a(x))
            #x = F.relu(self.fc1b(x))

            #x = F.relu(self.fc1c(x))
            return x

        def decode(self, x):
            
            #out = self.relu(self.fc4a(x))
            #out = self.relu(self.fc4b(out))
            #out = self.relu(self.fc4c(out))
            out = self.relu(self.fc4c(x))
            out = out.view(out.size(0), 4096, 3, 3)
            out = self.batchnorm4096(out)
            out = self.relu(self.deconv1(out))#3x3x1024
            out = self.batchnorm1024(out)
            out = self.relu(self.deconv1b(out))
            out = self.batchnorm1024(out)
            out = self.dropout(out)
            out = self.relu(self.deconv1c(out))#9x9x128
            out = self.batchnorm128(out)
            out = self.relu(self.deconv2(out))#9x9x32
            out = self.batchnorm32(out)
            out = self.relu(self.deconv2b(out))#32x32x32
            out = self.batchnorm32(out)
            out = self.dropout(out)
            out = self.sigmoid( self.relu(self.deconv2c(out)))
            return out
            
        def forward(self, x):
            mu = self.encode(x)
            return self.decode(mu),mu


import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)#32x32x32
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=4, padding=0)#8x8x32
        self.conv3 = nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1)#8x8x96
        self.conv3b = nn.Conv2d(96, 96, kernel_size=2, stride=3, padding=0)#3x3x96
        self.conv4 = nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1)#3x3x256
        self.fc1a = nn.Linear(3 * 3 * 256, 180)


        self.fc4a = nn.Linear(180, 2304)
        self.deconv1 = nn.ConvTranspose2d(256, 96, kernel_size=3, stride=1, padding=1)
        self.deconv1b = nn.ConvTranspose2d(96, 96, kernel_size=2, stride=3, padding=0)
        self.deconv2 = nn.ConvTranspose2d(96, 32, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=4, padding=0)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
            
        self.batchnorm32 = nn.BatchNorm2d(32)
        self.batchnorm96 = nn.BatchNorm2d(96)
        
    def encode(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.batchnorm32(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv3b(x))
        x = self.batchnorm96(x)
        x = self.dropout(x)
        x = F.relu(self.conv4(x))
       
        x = x.view(x.size(0),-1)

        x = F.relu(self.fc1a(x))
        
        return x
       
        
    def decode(self, x):
        out = self.relu(self.fc4a(x))
        out = out.view(out.size(0), 256, 3, 3)

        out = self.relu(self.deconv1(out))
        out = self.relu(self.deconv1b(out))
        out = self.batchnorm96(out)
        out = self.dropout(out)
        out = self.relu(self.deconv2(out))
        out = self.relu(self.deconv3(out))
        out = self.batchnorm32(out)
        out = self.dropout(out)
        out = self.sigmoid(self.conv5(out))
        return out
        
    def forward(self, x):
        mu = self.encode(x)
        return self.decode(mu),mu



class Net_deepwide_do(nn.Module):
        def __init__(self):
            super(Net_deepwide_do, self).__init__()

            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)#32x32x32
            self.conv1b = nn.Conv2d(32, 32, kernel_size=4, stride=4, padding=2)#9x9x32
            self.conv1c = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)#9x9x128
            self.conv2 = nn.Conv2d(128, 1024, kernel_size=3, stride=1, padding=1)#9x9x1024
            self.conv2b = nn.Conv2d(1024, 1024, kernel_size=3, stride=4, padding=1)#3x3x1024
            self.conv2c = nn.Conv2d(1024, 4096, kernel_size=3, stride=1, padding=1)#3x3x4096
            self.fc1a = nn.Linear(3 * 3 * 4096, 4608)
            self.fc1b = nn.Linear(4608, 1152)
            self.fc1c = nn.Linear(1152, 144)

            '''# Latent space
            self.fc21 = nn.Linear(128, 20)
            self.fc22 = nn.Linear(128, 20)'''
            # Decoder
            '''self.fc3 = nn.Linear(20, 128)'''

            
            self.fc4a = nn.Linear(144, 1152)
            self.fc4b = nn.Linear(1152,4608)
            self.fc4c = nn.Linear(4608, 3 * 3 * 4096)
            
            self.deconv1 = nn.ConvTranspose2d(4096, 1024, kernel_size=3, stride=1, padding=1)#3x3x1024
            self.deconv1b = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=4, padding=1)#9x9x1024
            self.deconv1c = nn.ConvTranspose2d(1024, 128, kernel_size=3, stride=1, padding=1)#9x9x128
            self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=1, padding=1)#9x9x32
            self.deconv2b = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=4, padding=2)#32x32x32
            self.deconv2c = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)#
            
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.5)
            
            self.batchnorm128 = nn.BatchNorm2d(128)
            self.batchnorm1024 = nn.BatchNorm2d(1024)
            self.batchnorm4096 = nn.BatchNorm2d(4096)
            self.batchnorm32 = nn.BatchNorm2d(32)
            
        def encode(self, x):

            #do_write("Encoding: Convolution...\n")
            x = F.relu(self.conv1(x))
            #x = self.batchnorm32(x)
            x = F.relu(self.conv1b(x))
            #x = self.batchnorm32(x)
            x = F.relu(self.conv1c(x))
            #x = self.batchnorm128(x)
            x = self.dropout(x)
            x = F.relu(self.conv2(x))
            #x = self.batchnorm1024(x)
            x = F.relu(self.conv2b(x))
            #x = self.batchnorm1024(x)
            x = F.relu(self.conv2c(x))
            #x = self.batchnorm4096(x)
            x = self.dropout(x)
       
            x = x.view(x.size(0),-1)
            #pdb.set_trace()

            #do_write("Encoding: FC...\n")
            x = F.relu(self.fc1a(x))
            x = F.relu(self.fc1b(x))

            x = F.relu(self.fc1c(x))
            return x

        def decode(self, x):
            
            out = self.relu(self.fc4a(x))
            out = self.relu(self.fc4b(out))
            out = self.relu(self.fc4c(out))
            # import pdb; pdb.set_trace()
            out = out.view(out.size(0), 4096, 3, 3)
            #out = self.batchnorm4096(out)
            out = self.relu(self.deconv1(out))#3x3x1024
            #out = self.batchnorm1024(out)
            out = self.relu(self.deconv1b(out))
            #out = self.batchnorm1024(out)
            out = self.dropout(out)
            out = self.relu(self.deconv1c(out))#9x9x128
            #out = self.batchnorm128(out)
            out = self.relu(self.deconv2(out))#9x9x32
            #out = self.batchnorm32(out)
            out = self.relu(self.deconv2b(out))#32x32x32
            #out = self.batchnorm32(out)
            out = self.dropout(out)
            out = self.sigmoid( self.relu(self.deconv2c(out)))#32x32x3
            return out
            
        def forward(self, x):
            mu = self.encode(x)

            return self.decode(mu),mu

class Net_deepwide(nn.Module):
        def __init__(self):
            super(Net_deepwide, self).__init__()

            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)#32x32x32
            self.conv1b = nn.Conv2d(32, 32, kernel_size=4, stride=4, padding=2)#9x9x32
            self.conv1c = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)#9x9x128
            self.conv2 = nn.Conv2d(128, 1024, kernel_size=3, stride=1, padding=1)#9x9x1024
            self.conv2b = nn.Conv2d(1024, 1024, kernel_size=3, stride=4, padding=1)#3x3x1024
            self.conv2c = nn.Conv2d(1024, 4096, kernel_size=3, stride=1, padding=1)#3x3x4096
            self.fc1a = nn.Linear(3 * 3 * 4096, 4608)
            self.fc1b = nn.Linear(4608, 1152)
            self.fc1c = nn.Linear(1152, 144)

            '''# Latent space
            self.fc21 = nn.Linear(128, 20)
            self.fc22 = nn.Linear(128, 20)'''
            # Decoder
            '''self.fc3 = nn.Linear(20, 128)'''

            
            self.fc4a = nn.Linear(144, 1152)
            self.fc4b = nn.Linear(1152,4608)
            self.fc4c = nn.Linear(4608, 3 * 3 * 4096)
            
            self.deconv1 = nn.ConvTranspose2d(4096, 1024, kernel_size=3, stride=1, padding=1)#3x3x1024
            self.deconv1b = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=4, padding=1)#9x9x1024
            self.deconv1c = nn.ConvTranspose2d(1024, 128, kernel_size=3, stride=1, padding=1)#9x9x128
            self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=1, padding=1)#9x9x32
            self.deconv2b = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=4, padding=2)#32x32x32
            self.deconv2c = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)#
            
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.5)
            
            #self.batchnorm128 = nn.BatchNorm2d(128)
            #self.batchnorm1024 = nn.BatchNorm2d(1024)
            #self.batchnorm4096 = nn.BatchNorm2d(4096)
            #self.batchnorm32 = nn.BatchNorm2d(32)
            
        def encode(self, x):

            #do_write("Encoding: Convolution...\n")
            x = F.relu(self.conv1(x))
            #x = self.batchnorm32(x)
            x = F.relu(self.conv1b(x))
            #x = self.batchnorm32(x)
            x = F.relu(self.conv1c(x))
            #x = self.batchnorm128(x)
            #x = self.dropout(x)
            x = F.relu(self.conv2(x))
            #x = self.batchnorm1024(x)
            x = F.relu(self.conv2b(x))
            #x = self.batchnorm1024(x)
            x = F.relu(self.conv2c(x))
            #x = self.batchnorm4096(x)
            #x = self.dropout(x)
       
            x = x.view(x.size(0),-1)

            x = F.relu(self.fc1a(x))
            x = F.relu(self.fc1b(x))

            x = F.relu(self.fc1c(x))
            return x

        def decode(self, x):
            
            out = self.relu(self.fc4a(x))
            out = self.relu(self.fc4b(out))
            out = self.relu(self.fc4c(out))
            # import pdb; pdb.set_trace()
            out = out.view(out.size(0), 4096, 3, 3)
            #out = self.batchnorm4096(out)
            out = self.relu(self.deconv1(out))#3x3x1024
            #out = self.batchnorm1024(out)
            out = self.relu(self.deconv1b(out))
            #out = self.batchnorm1024(out)
            #out = self.dropout(out)
            out = self.relu(self.deconv1c(out))#9x9x128
            #out = self.batchnorm128(out)
            out = self.relu(self.deconv2(out))#9x9x32
            #out = self.batchnorm32(out)
            out = self.relu(self.deconv2b(out))#32x32x32
            #out = self.batchnorm32(out)
            #out = self.dropout(out)
            out = self.sigmoid( self.relu(self.deconv2c(out)))#32x32x3
            return out
            
        def forward(self, x):
            mu = self.encode(x)

            return self.decode(mu),mu


class Net_deepwide_enc_do(nn.Module):
        def __init__(self):
            super(Net_deepwide_enc_do, self).__init__()
             # Encoder
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)#32x32x32
            self.conv1b = nn.Conv2d(32, 32, kernel_size=4, stride=4, padding=2)#9x9x32
            self.conv1c = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)#9x9x128
            self.conv2 = nn.Conv2d(128, 1024, kernel_size=3, stride=1, padding=1)#9x9x1024
            self.conv2b = nn.Conv2d(1024, 1024, kernel_size=3, stride=4, padding=1)#3x3x1024
            self.conv2c = nn.Conv2d(1024, 4096, kernel_size=3, stride=1, padding=1)#3x3x4096
            self.fc1a = nn.Linear(3 * 3 * 4096, 4608)
            self.fc1b = nn.Linear(4608, 1152)
            self.fc1c = nn.Linear(1152, 144)

            '''# Latent space
            self.fc21 = nn.Linear(128, 20)
            self.fc22 = nn.Linear(128, 20)'''
            # Decoder
            '''self.fc3 = nn.Linear(20, 128)'''

            
            self.fc4a = nn.Linear(144, 1152)
            self.fc4b = nn.Linear(1152,4608)
            self.fc4c = nn.Linear(4608, 3 * 3 * 4096)
            
            self.deconv1 = nn.ConvTranspose2d(4096, 1024, kernel_size=3, stride=1, padding=1)#3x3x1024
            self.deconv1b = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=4, padding=1)#9x9x1024
            self.deconv1c = nn.ConvTranspose2d(1024, 128, kernel_size=3, stride=1, padding=1)#9x9x128
            self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=1, padding=1)#9x9x32
            self.deconv2b = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=4, padding=2)#32x32x32
            self.deconv2c = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)#
            
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.5)
            
            self.batchnorm128 = nn.BatchNorm2d(128)
            self.batchnorm1024 = nn.BatchNorm2d(1024)
            self.batchnorm4096 = nn.BatchNorm2d(4096)
            self.batchnorm32 = nn.BatchNorm2d(32)
            
        def encode(self, x):

            x = F.relu(self.conv1(x))
            x = self.batchnorm32(x)
            x = F.relu(self.conv1b(x))
            x = self.batchnorm32(x)
            x = F.relu(self.conv1c(x))
            x = self.batchnorm128(x)
            x = self.dropout(x)
            x = F.relu(self.conv2(x))
            x = self.batchnorm1024(x)
            x = F.relu(self.conv2b(x))
            x = self.batchnorm1024(x)
            x = F.relu(self.conv2c(x))
            x = self.batchnorm4096(x)
            x = self.dropout(x)
       
            x = x.view(x.size(0),-1)

            x = F.relu(self.fc1a(x))
            x = F.relu(self.fc1b(x))

            x = F.relu(self.fc1c(x))
            return x

        def decode(self, x):
            
            out = self.relu(self.fc4a(x))
            out = self.relu(self.fc4b(out))
            out = self.relu(self.fc4c(out))
            out = out.view(out.size(0), 4096, 3, 3)
            out = self.batchnorm4096(out)
            out = self.relu(self.deconv1(out))#3x3x1024
            out = self.batchnorm1024(out)
            out = self.relu(self.deconv1b(out))
            out = self.batchnorm1024(out)
            #out = self.dropout(out)
            out = self.relu(self.deconv1c(out))#9x9x128
            out = self.batchnorm128(out)
            out = self.relu(self.deconv2(out))#9x9x32
            out = self.batchnorm32(out)
            out = self.relu(self.deconv2b(out))#32x32x32
            out = self.batchnorm32(out)
            #out = self.dropout(out)
            out = self.sigmoid( self.relu(self.deconv2c(out)))#32x32x3
            return out
            
        def forward(self, x):
            mu = self.encode(x)

            return self.decode(mu),mu



class Net_deepwide_dec_do(nn.Module):
        def __init__(self):
            super(Net_deepwide_dec_do, self).__init__()
             # Encoder

            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)#32x32x32
            self.conv1b = nn.Conv2d(32, 32, kernel_size=4, stride=4, padding=2)#9x9x32
            self.conv1c = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)#9x9x128
            self.conv2 = nn.Conv2d(128, 1024, kernel_size=3, stride=1, padding=1)#9x9x1024
            self.conv2b = nn.Conv2d(1024, 1024, kernel_size=3, stride=4, padding=1)#3x3x1024
            self.conv2c = nn.Conv2d(1024, 4096, kernel_size=3, stride=1, padding=1)#3x3x4096
            self.fc1a = nn.Linear(3 * 3 * 4096, 4608)
            self.fc1b = nn.Linear(4608, 1152)
            self.fc1c = nn.Linear(1152, 144)

            '''# Latent space
            self.fc21 = nn.Linear(128, 20)
            self.fc22 = nn.Linear(128, 20)'''
            # Decoder
            '''self.fc3 = nn.Linear(20, 128)'''

            
            self.fc4a = nn.Linear(144, 1152)
            self.fc4b = nn.Linear(1152,4608)
            self.fc4c = nn.Linear(4608, 3 * 3 * 4096)
            
            self.deconv1 = nn.ConvTranspose2d(4096, 1024, kernel_size=3, stride=1, padding=1)#3x3x1024
            self.deconv1b = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=4, padding=1)#9x9x1024
            self.deconv1c = nn.ConvTranspose2d(1024, 128, kernel_size=3, stride=1, padding=1)#9x9x128
            self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=1, padding=1)#9x9x32
            self.deconv2b = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=4, padding=2)#32x32x32
            self.deconv2c = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)#
            
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.5)
            
            self.batchnorm128 = nn.BatchNorm2d(128)
            self.batchnorm1024 = nn.BatchNorm2d(1024)
            self.batchnorm4096 = nn.BatchNorm2d(4096)
            self.batchnorm32 = nn.BatchNorm2d(32)
            
        def encode(self, x):

            x = F.relu(self.conv1(x))
            x = self.batchnorm32(x)
            x = F.relu(self.conv1b(x))
            x = self.batchnorm32(x)
            x = F.relu(self.conv1c(x))
            x = self.batchnorm128(x)
            x = self.dropout(x)
            x = F.relu(self.conv2(x))
            x = self.batchnorm1024(x)
            x = F.relu(self.conv2b(x))
            x = self.batchnorm1024(x)
            x = F.relu(self.conv2c(x))
            x = self.batchnorm4096(x)
            x = self.dropout(x)
       
            x = x.view(x.size(0),-1)

            x = F.relu(self.fc1a(x))
            x = F.relu(self.fc1b(x))

            x = F.relu(self.fc1c(x))
            return x

        def decode(self, x):
            
            out = self.relu(self.fc4a(x))
            out = self.relu(self.fc4b(out))
            out = self.relu(self.fc4c(out))
            out = out.view(out.size(0), 4096, 3, 3)
            out = self.batchnorm4096(out)
            out = self.relu(self.deconv1(out))#3x3x1024
            out = self.batchnorm1024(out)
            out = self.relu(self.deconv1b(out))
            out = self.batchnorm1024(out)
            out = self.dropout(out)
            out = self.relu(self.deconv1c(out))#9x9x128
            out = self.batchnorm128(out)
            out = self.relu(self.deconv2(out))#9x9x32
            out = self.batchnorm32(out)
            out = self.relu(self.deconv2b(out))#32x32x32
            out = self.batchnorm32(out)
            out = self.dropout(out)
            out = self.sigmoid( self.relu(self.deconv2c(out)))
            return out
            
        def forward(self, x):
            mu = self.encode(x)
            return self.decode(mu),mu

class Net_wide(nn.Module):
        def __init__(self):
            super(Net_wide, self).__init__()
             # Encoder

            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)#32x32x32
            self.conv1b = nn.Conv2d(32, 32, kernel_size=4, stride=4, padding=2)#9x9x32
            self.conv1c = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)#9x9x128
            self.conv2 = nn.Conv2d(128, 1024, kernel_size=3, stride=1, padding=1)#9x9x1024
            self.conv2b = nn.Conv2d(1024, 1024, kernel_size=3, stride=4, padding=1)#3x3x1024
            self.conv2c = nn.Conv2d(1024, 4096, kernel_size=3, stride=1, padding=1)#3x3x4096
            self.fc1a = nn.Linear(3 * 3 * 4096, 4608)
            #self.fc1b = nn.Linear(4608, 1152)
            #self.fc1c = nn.Linear(1152, 144)

            '''# Latent space
            self.fc21 = nn.Linear(128, 20)
            self.fc22 = nn.Linear(128, 20)'''
            # Decoder
            '''self.fc3 = nn.Linear(20, 128)'''

            
            #self.fc4a = nn.Linear(144, 1152)
            #self.fc4b = nn.Linear(1152,4608)
            self.fc4c = nn.Linear(4608, 3 * 3 * 4096)
            
            self.deconv1 = nn.ConvTranspose2d(4096, 1024, kernel_size=3, stride=1, padding=1)#3x3x1024
            self.deconv1b = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=4, padding=1)#9x9x1024
            self.deconv1c = nn.ConvTranspose2d(1024, 128, kernel_size=3, stride=1, padding=1)#9x9x128
            self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=1, padding=1)#9x9x32
            self.deconv2b = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=4, padding=2)#32x32x32
            self.deconv2c = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)#
            
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.5)
            
            self.batchnorm128 = nn.BatchNorm2d(128)
            self.batchnorm1024 = nn.BatchNorm2d(1024)
            self.batchnorm4096 = nn.BatchNorm2d(4096)
            self.batchnorm32 = nn.BatchNorm2d(32)
            
        def encode(self, x):

            x = F.relu(self.conv1(x))
            x = self.batchnorm32(x)
            x = F.relu(self.conv1b(x))
            x = self.batchnorm32(x)
            x = F.relu(self.conv1c(x))
            x = self.batchnorm128(x)
            x = self.dropout(x)
            x = F.relu(self.conv2(x))
            x = self.batchnorm1024(x)
            x = F.relu(self.conv2b(x))
            x = self.batchnorm1024(x)
            x = F.relu(self.conv2c(x))
            x = self.batchnorm4096(x)
            x = self.dropout(x)
       
            x = x.view(x.size(0),-1)

            x = F.relu(self.fc1a(x))
            #x = F.relu(self.fc1b(x))

            #x = F.relu(self.fc1c(x))
            return x

        def decode(self, x):
            
            #out = self.relu(self.fc4a(x))
            #out = self.relu(self.fc4b(out))
            #out = self.relu(self.fc4c(out))
            out = self.relu(self.fc4c(x))
            out = out.view(out.size(0), 4096, 3, 3)
            out = self.batchnorm4096(out)
            out = self.relu(self.deconv1(out))#3x3x1024
            out = self.batchnorm1024(out)
            out = self.relu(self.deconv1b(out))
            out = self.batchnorm1024(out)
            out = self.dropout(out)
            out = self.relu(self.deconv1c(out))#9x9x128
            out = self.batchnorm128(out)
            out = self.relu(self.deconv2(out))#9x9x32
            out = self.batchnorm32(out)
            out = self.relu(self.deconv2b(out))#32x32x32
            out = self.batchnorm32(out)
            out = self.dropout(out)
            out = self.sigmoid( self.relu(self.deconv2c(out)))
            return out
            
        def forward(self, x):
            mu = self.encode(x)
            return self.decode(mu),mu


class Net_middle(nn.Module):
        def __init__(self):
            super(Net_middle, self).__init__()
             # Encoder

            self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=4, padding=1)#9x9x32
            self.conv2 = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)#9x9x128
            self.conv3 = nn.Conv2d(128, 1024, kernel_size=3, stride=4, padding=1)#3x3x1024
            self.fc1 = nn.Linear(3 * 3 * 1024, 1024)
            #self.fc1b = nn.Linear(4608, 1152)
            #self.fc1c = nn.Linear(1152, 144)

            '''# Latent space
            self.fc21 = nn.Linear(128, 20)
            self.fc22 = nn.Linear(128, 20)'''
            # Decoder
            '''self.fc3 = nn.Linear(20, 128)'''

            
            #self.fc4a = nn.Linear(144, 1152)
            #self.fc4b = nn.Linear(1152,4608)
            self.fc4 = nn.Linear(1024, 3 * 3 * 1024)#3x3x1024
            
            self.deconv1 = nn.ConvTranspose2d(1024, 128, kernel_size=3, stride=4, padding=1)#9x9x128
            self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=1, padding=1)#9x9x32
            self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=4, padding=1)#32x32x3
            
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.5)
            
            self.batchnorm128 = nn.BatchNorm2d(128)
            self.batchnorm1024 = nn.BatchNorm2d(1024)
            self.batchnorm4096 = nn.BatchNorm2d(4096)
            self.batchnorm32 = nn.BatchNorm2d(32)
            
        def encode(self, x):

            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.batchnorm128(x)
            x = self.dropout(x)
            x = F.relu(self.conv3(x))
            x = self.batchnorm1024(x)
            x = self.dropout(x)
       
            x = x.view(x.size(0),-1)

            x = F.relu(self.fc1(x))

            return x

        def decode(self, x):
            
            #out = self.relu(self.fc4a(x))
            #out = self.relu(self.fc4b(out))
            #out = self.relu(self.fc4c(out))
            out = self.relu(self.fc4(x))
            out = out.view(out.size(0), 1024, 3, 3)
            out = self.relu(self.deconv1(out))#9x9x128
            out = self.batchnorm128(out)
            out = self.dropout(out)
            out = self.relu(self.deconv2(out))#9x9x32
            out = self.batchnorm32(out)
            out = self.dropout(out)
            out = self.sigmoid( self.relu(self.deconv3(out)))
            return out
            
        def forward(self, x):
            mu = self.encode(x)
            return self.decode(mu),mu





