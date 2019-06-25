import torch.nn as nn
import torch.nn.functional as F


def conv(in_channels, out_channels, sized=3, strided=1, padded=1):

    return nn.Sequential(

        nn.Conv2d(in_channels, out_channels, kernel_size=sized, padding=padded, stride=strided),

        nn.LeakyReLU(inplace=True),

    )


def conv_dilate(in_channels, out_channels, sized=3, strided=1,padded=1):

    return nn.Sequential(

        nn.Conv2d(in_channels, out_channels, kernel_size=sized, padding=padded, stride=strided, dilation=3),

        nn.LeakyReLU(inplace=True),

    )

def conv_bn(in_channels, out_channels, sized=3, strided=1,padded=1):

    return nn.Sequential(

        nn.Conv2d(in_channels, out_channels, kernel_size=sized, padding=padded, stride=strided),

        nn.LeakyReLU(inplace=True),

        nn.BatchNorm2d(out_channels),

    )

def conv_bn_dilate(in_channels, out_channels, sized=3, strided=1,padded=1):

    return nn.Sequential(

        nn.Conv2d(in_channels, out_channels, kernel_size=sized, padding=padded, stride=strided, dilation=3),

        nn.LeakyReLU(inplace=True),

        nn.BatchNorm2d(out_channels),

    )

def de_conv_bn(in_channels, out_channels, sized=3, strided=1, padded=1):

    return nn.Sequential(

        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=sized, padding=padded, stride=strided),

        nn.LeakyReLU(inplace=True),

        nn.BatchNorm2d(out_channels),

    )
def de_conv_bn_dilate(in_channels, out_channels, sized=3, strided=1, padded=1):

    return nn.Sequential(

        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=sized, padding=padded, stride=strided, dilation=3),

        nn.LeakyReLU(inplace=True),

        nn.BatchNorm2d(out_channels),

    )


class Net_Selector(nn.Module):

    def __init__(self):
        super(Net_Selector, self).__init__()
        self.encoder = nn.Sequential(
            conv(3, 16, 7,1,3),# 240x240x3 -> 236x236x32
            conv_bn(16,16,3),#240x240x32
            nn.MaxPool2d(2),#120x120x32
            conv_bn(16, 32, 7,1,3),#120x120x32
            conv_bn(32, 32, 3),
            nn.MaxPool2d(2),#60x60x32
            conv_bn(32, 64, 5,1,2),#60x60x64
            conv_bn(64, 64, 3),
            nn.MaxPool2d(2),#30x30x128
            conv_bn(64, 128, 7,1,3),#30x30x128
            conv_bn(128, 128, 3),
            nn.MaxPool2d(2),#15x15x128
            conv_bn(128, 256, 5,1,2),#15x15x256
            conv_bn(256, 256, 3),
            nn.MaxPool2d(2),#8x8x256
            conv_bn(256,512,3),#8x8x512
            conv_bn(512,512,3),
            nn.MaxPool2d(2),#4x4x512
            conv_bn(512,1024,4,2,1),#2x2x1024
            nn.MaxPool2d(2),
            conv_bn(1024,1024,3)#1x1x1024
        )

        # Decoder with Conv and Upsampling
        self.decoder = nn.Sequential(
            conv_bn(1024,1024,3),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_bn(1024,512,4,2,1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_bn(512,512,3),
            conv_bn(512,256,3),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_bn(256, 256, 3),
            conv_bn(256,128, 5,1,2),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_bn(128, 128, 3),
            conv_bn(128, 64, 7,1,3),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_bn(64, 64, 3),
            conv_bn(64, 32, 5,1,2),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_bn(32, 32, 3),
            conv_bn(32, 16, 7,1,3),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_bn(16,16,3),
            conv_bn(16, 3, 7,1,3),
        )

    def forward(self, x):
        input_size = x[0,0,0].size()
        features = self.encoder(x)
        x = self.decoder(features)
        x = nn.functional.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return x, features

class DAE(nn.Module):



    def __init__(self):

        super(DAE, self).__init__()

        self.encoder = nn.Sequential(

            conv(3, 16, 7),

            nn.MaxPool2d(3),

            conv(16, 32, 7),

            nn.MaxPool2d(2),

            conv_bn(32, 64, 7),

            nn.MaxPool2d(2),

            conv_bn(64, 128, 5),

            nn.MaxPool2d(2),

            conv_bn(128, 256, 3),

            nn.MaxPool2d(2),

            conv_bn(256, 512, 3),

            nn.MaxPool2d(2),

            conv_bn(512, 1024, 3),

        )

        # Decoder with Conv and Upsampling

        self.decoder = nn.Sequential(

            conv_bn(1024, 512, 3),

            nn.UpsamplingBilinear2d(scale_factor=2),

            conv_bn(512, 256, 3),

            nn.UpsamplingBilinear2d(scale_factor=2),

            conv_bn(256, 128, 3),

            nn.UpsamplingBilinear2d(scale_factor=2),

            conv_bn(128, 64, 5),

            nn.UpsamplingBilinear2d(scale_factor=2),

            conv_bn(64, 32, 7),

            nn.UpsamplingBilinear2d(scale_factor=2),

            conv_bn(32, 16, 7),

            nn.UpsamplingBilinear2d(scale_factor=3),

            conv_bn(16, 3, 7),

        )



    def forward(self, x):

        input_size = x[0,0,0].size()

        features = self.encoder(x)

        x = self.decoder(features)

        x = nn.functional.interpolate(x, size=input_size, mode='bilinear', align_corners=True)

        return x, features



class DAE_conconpool(nn.Module):

    def __init__(self):
        super(DAE_conconpool, self).__init__()
        self.encoder = nn.Sequential(
            conv(3, 16, 7,1,3),# 240x240x3 -> 236x236x32
            conv_bn(16,16,3),#240x240x32
            nn.MaxPool2d(2),#120x120x32
            conv_bn(16, 32, 7,1,3),#120x120x32
            conv_bn(32, 32, 3),
            nn.MaxPool2d(2),#60x60x32
            conv_bn(32, 64, 5,1,2),#60x60x64
            conv_bn(64, 64, 3),
            nn.MaxPool2d(2),#30x30x128
            conv_bn(64, 128, 7,1,3),#30x30x128
            conv_bn(128, 128, 3),
            nn.MaxPool2d(2),#15x15x128
            conv_bn(128, 256, 5,1,2),#15x15x256
            conv_bn(256, 256, 3),
            nn.MaxPool2d(2),#8x8x256
            conv_bn(256,512,3),#8x8x512
            conv_bn(512,512,3),
            nn.MaxPool2d(2),#4x4x512
            conv_bn(512,1024,4,2,1),#2x2x1024
            nn.MaxPool2d(2),
            conv_bn(1024,1024,3)#1x1x1024
        )

        # Decoder with Conv and Upsampling
        self.decoder = nn.Sequential(
            conv_bn(1024,1024,3),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_bn(1024,512,4,2,1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_bn(512,512,3),
            conv_bn(512,256,3),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_bn(256, 256, 3),
            conv_bn(256,128, 5,1,2),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_bn(128, 128, 3),
            conv_bn(128, 64, 7,1,3),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_bn(64, 64, 3),
            conv_bn(64, 32, 5,1,2),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_bn(32, 32, 3),
            conv_bn(32, 16, 7,1,3),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_bn(16,16,3),
            conv_bn(16, 3, 7,1,3),
        )

    def forward(self, x):
        input_size = x[0,0,0].size()
        features = self.encoder(x)
        x = self.decoder(features)
        x = nn.functional.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return x, features




class DAE_conconpool_average(nn.Module):

    def __init__(self):
        super(DAE_conconpool_average, self).__init__()
        self.encoder = nn.Sequential(
            conv(3, 16, 7,1,3),# 240x240x3 -> 236x236x32
            conv_bn(16,16,3),#240x240x32
            nn.AvgPool2d(2),#120x120x32
            conv_bn(16, 32, 7,1,3),#120x120x32
            conv_bn(32, 32, 3),
            nn.AvgPool2d(2),#60x60x32
            conv_bn(32, 64, 5,1,2),#60x60x64
            conv_bn(64, 64, 3),
            nn.AvgPool2d(2),#30x30x128
            conv_bn(64, 128, 7,1,3),#30x30x128
            conv_bn(128, 128, 3),
            nn.AvgPool2d(2),#15x15x128
            conv_bn(128, 256, 5,1,2),#15x15x256
            conv_bn(256, 256, 3),
            nn.AvgPool2d(2),#8x8x256
            conv_bn(256,512,3),#8x8x512
            conv_bn(512,512,3),
            nn.AvgPool2d(2),#4x4x512
            conv_bn(512,1024,4,2,1),#2x2x1024
            nn.AvgPool2d(2),
            conv_bn(1024,1024,3)#1x1x1024
        )

        # Decoder with Conv and Upsampling
        self.decoder = nn.Sequential(
            conv_bn(1024,1024,3),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_bn(1024,512,4,2,1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_bn(512,512,3),
            conv_bn(512,256,3),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_bn(256, 256, 3),
            conv_bn(256,128, 5,1,2),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_bn(128, 128, 3),
            conv_bn(128, 64, 7,1,3),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_bn(64, 64, 3),
            conv_bn(64, 32, 5,1,2),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_bn(32, 32, 3),
            conv_bn(32, 16, 7,1,3),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_bn(16,16,3),
            conv_bn(16, 3, 7,1,3),
        )

    def forward(self, x):
        input_size = x[0,0,0].size()
        features = self.encoder(x)
        x = self.decoder(features)
        x = nn.functional.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return x, features


class DAE_stridedconvFC(nn.Module):

    def __init__(self):
        super(DAE_stridedconvFC, self).__init__()
        self.encoder = nn.Sequential(
            conv(3, 16, 3),# 240x240x3 -> 240x240x16
            conv_bn(16, 16, 4,2,2),#121x121x16

            conv_bn(16, 32, 3),#121x121x32
            conv_bn(32, 32, 3,2,1),#61x61x32

            conv_bn(32, 64, 3),#61x61x64
            conv_bn(64, 64, 3,2,1),#31x31x64

            conv_bn(64, 128, 3),#31x31x128
            conv_bn(128, 128, 3,2,1),#16x16x128

            conv_bn(128,256,3),#16x16x256
            conv_bn(256,256,3,3,1),#6x6x256

            conv_bn(256,512,4,2,1),#3x3x512
        )

        # Decoder with Conv and Upsampling
        self.decoder = nn.Sequential(
            de_conv_bn(512,256,4,2,1),#3x3x512            
            de_conv_bn(256,256,3,3,1),#6x6x256

            de_conv_bn(256,128,3),
            de_conv_bn(128, 128, 3,2,1),#16x16x128

            de_conv_bn(128, 64, 3),#31x31x128
            de_conv_bn(64, 64, 3,2,1),#31x31x64

            de_conv_bn(64,32, 3),#61x61x64
            de_conv_bn(32, 32, 3,2,1),#61x61x32

            de_conv_bn(32,16, 3),#121x121x32
            de_conv_bn(16, 16, 4,2,2),#121x121x16

            conv(16, 3, 3),# 240x240x3 -> 240x240x16

        )
        self.fc1a = nn.Linear(3 * 3 * 512, 1024)


        self.fc4a = nn.Linear(1024, 3*3*512)
        

    def forward(self, x):
        input_size = x[0,0,0].size()
        x = self.encoder(x)

        x = x.view(x.size(0),-1)
        features = F.relu(self.fc1a(x))

        out = F.relu(self.fc4a(features))
        out = out.view(out.size(0), 512, 3, 3)

        out = self.decoder(out)
        out = nn.functional.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        return out, features


#####

class DAE_stridedconvFC_shallow(nn.Module):

    def __init__(self):
        super(DAE_stridedconvFC_shallow, self).__init__()
        self.encoder = nn.Sequential(
            conv(3, 16, 3),# 240x240x3 -> 240x240x16
            conv_bn(16, 16, 4,2,2),#121x121x16

            conv_bn(16, 32, 3),#121x121x32
            conv_bn(32, 32, 3,2,1),#61x61x32

            conv_bn(32, 64, 3),#61x61x64
            conv_bn(64, 64, 3,2,1),#31x31x64

            conv_bn(64, 128, 3),#31x31x128
            conv_bn(128, 128, 3,2,1),#16x16x128

            conv_bn(128,256,3),#16x16x256
            conv_bn(256,256,2,2,1),#9x9x256

            conv_bn(256,512,3,2,1),#5x5x512
        )

        # Decoder with Conv and Upsampling
        self.decoder = nn.Sequential(
            de_conv_bn(512,256,3,2,1),#5x5x512            
            de_conv_bn(256,256,2,2,1),#9x9x256

            de_conv_bn(256,128,3),
            de_conv_bn(128, 128, 3,2,1),#16x16x128

            de_conv_bn(128, 64, 3),#31x31x128
            de_conv_bn(64, 64, 3,2,1),#31x31x64

            de_conv_bn(64,32, 3),#61x61x64
            de_conv_bn(32, 32, 3,2,1),#61x61x32

            de_conv_bn(32,16, 3),#121x121x32
            de_conv_bn(16, 16, 4,2,2),#121x121x16

            conv(16, 3, 3),# 240x240x3 -> 240x240x16

        )
        self.fc1a = nn.Linear(5 * 5 * 512, 6400)


        self.fc4a = nn.Linear(6400, 5*5*512)
        

    def forward(self, x):
        input_size = x[0,0,0].size()
        x = self.encoder(x)

        x = x.view(x.size(0),-1)
        features = F.relu(self.fc1a(x))

        out = F.relu(self.fc4a(features))
        out = out.view(out.size(0), 512, 5, 5)

        out = self.decoder(out)
        out = nn.functional.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        return out, features






#####
class DAE_dilate(nn.Module):



    def __init__(self):

        super(DAE_dilate, self).__init__()

        self.encoder = nn.Sequential(

            conv_dilate(3, 16, 7),

            nn.MaxPool2d(2),#120#80

            conv(16, 32, 7),

            nn.MaxPool2d(2),#60#40

            conv_bn(32, 64, 7),

            nn.MaxPool2d(2),#30#20

            conv_bn(64, 128, 5),

            nn.MaxPool2d(2),#15#10

            conv_bn(128, 256, 3),

            nn.MaxPool2d(2),#8#5

            conv_bn(256, 512, 3),

            nn.MaxPool2d(2),#4#2.5

            conv_bn(512, 1024, 2, strided=4 )

        )

        # Decoder with Conv and Upsampling

        self.decoder = nn.Sequential(

            conv_bn(1024, 512, 2, strided=4),

            nn.UpsamplingBilinear2d(scale_factor=2),

            conv_bn(512, 256, 3),

            nn.UpsamplingBilinear2d(scale_factor=2),

            conv_bn(256, 128, 3),

            nn.UpsamplingBilinear2d(scale_factor=2),

            conv_bn(128, 64, 5),

            nn.UpsamplingBilinear2d(scale_factor=2),

            conv_bn(64, 32, 7),

            nn.UpsamplingBilinear2d(scale_factor=2),

            conv_bn(32, 16, 7),

            nn.UpsamplingBilinear2d(scale_factor=2),

            conv_bn_dilate(16, 3, 7),

        )



    def forward(self, x):

        input_size = x[0,0,0].size()

        features = self.encoder(x)

        x = self.decoder(features)

        x = nn.functional.interpolate(x, size=input_size, mode='bilinear', align_corners=True)

        return x, features



#################################################

class AutoencoderFrame(nn.Module):

    def __init__(self):
        super(AutoencoderFrame, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=7),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.Conv2d(20, 60, kernel_size=5),
            nn.BatchNorm2d(60),
            nn.ReLU(True),
            nn.Conv2d(60, 120, kernel_size=3),
            nn.BatchNorm2d(120),
            nn.ReLU(True))


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(120, 60, kernel_size=3),
            nn.BatchNorm2d(60),
            nn.ReLU(True),
            nn.ConvTranspose2d(60, 20, kernel_size=5),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.ConvTranspose2d(20, 3, kernel_size=7),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



'''
class DAE(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.dconv_down1 = conv(3, 64, 7, 1)
        self.dconv_down2 = conv(64, 128, 5)
        self.dconv_down3 = conv(128, 256)
        self.dconv_down4 = conv(256, 512)
        self.dconv_down5 = conv(512, 1024)

        self.maxpool1 = nn.MaxPool2d(3)
        self.maxpool2 = nn.MaxPool2d(3)
        self.maxpool3 = nn.MaxPool2d(2)
        self.maxpool4 = nn.MaxPool2d(2)
        self.maxpool5 = nn.MaxPool2d(2)
        #self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.upsampleout = nn.Upsample(size=(238,238), mode='bilinear', align_corners=True)
        #self.zeropad = nn.ZeroPad2d(1)

        self.dconv_up5 = conv(1024, 512)
        self.dconv_up4 = conv(512, 256)
        self.dconv_up3 = conv(256, 128)
        self.dconv_up2 = conv(128, 64, 5)
        self.dconv_up1 = conv(64, 3, 7, 1)

        self.upsample5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=3, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)

        #self.conv_last = nn.Conv2d(64, , 1)

    def forward(self, x):

        input_size = x[0,0].size()

        conv1 = self.dconv_down1(x)
        x = self.maxpool1(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool2(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool3(conv3)

        x = self.dconv_down4(x)
        x = self.maxpool4(x)

        x = self.dconv_down5(x)
        x = self.maxpool5(x)

        #print(x.size())

        x = self.dconv_up5(x)
        #x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.upsample5(x)

        x = self.dconv_up4(x)
        #x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.upsample4(x)

        x = self.dconv_up3(x)
        #x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.upsample3(x)

        x = self.dconv_up2(x)
        #x = nn.functional.interpolate(x, scale_factor=3, mode='bilinear')
        x = self.upsample2(x)

        x = self.dconv_up1(x)
        #x = nn.functional.interpolate(x, size=input_size, mode='bilinear')
        x = self.upsample1(x)

        return x
'''


class AutoencoderMNIST(nn.Module):

    def __init__(self):
        super(AutoencoderMNIST, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3),
            nn.Tanh(),
            #nn.Conv2d(4, 10, kernel_size=3),
            #nn.Tanh(),
            nn.Conv2d(10, 16, kernel_size=4, stride=2),
            nn.Tanh())


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 10, kernel_size=4, stride=2),
            nn.Tanh(),
            nn.ConvTranspose2d(10, 1, kernel_size=3),
            nn.Tanh(),
            #nn.ConvTranspose2d(4, 1, kernel_size=3),
            #nn.Tanh(),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
