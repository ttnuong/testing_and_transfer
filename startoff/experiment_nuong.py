import os
import time

import numpy as np
import torch

from common.dataset.video_dataset import VideoDataset
from common.networks.autoencoder import AutoencoderFrame
from common.networks.autoencoder import DAE
from common.networks.nuong_AE import Net
from trixi.experiment import PytorchExperiment

import glob
import pickle
class ConvAE_Feat(PytorchExperiment):

    def setup(self):

        self.elog.print(self.config)
        data_loader_kwargs = {'num_workers': 12, 'pin_memory': True}

        if self.config.convert:
            raw_videos=glob.glob(os.join(self.config.data_path, self.config.convert_path))
            for video_path in raw_videos:
                video = np.load(video_path, mmap_mode='r')
        self.dataset_train = VideoDataset(os.path.join(self.config.data_path,self.config.train_data))
        self.dataset_test = VideoDataset(os.path.join(self.config.data_path,self.config.test_data))

        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.config.run.hyperparameter.batch_size,
                                                        shuffle=True, **data_loader_kwargs)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.config.run.hyperparameter.batch_size, shuffle = True)

        print('Done with loading.')

        ### Models
        self.model = DAE()
        self.model.cuda() # to gpu

        ### Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.run.hyperparameter.lr)

        ### Load pre-trained model
        if "load_path" in self.config:
            self.load_checkpoint(path=self.config.load_path, name="")

        ### Criterion
        #self.criterion = torch.nn.PairwiseDistance()    #Norm, default 2, in between data pair
        self.criterion = torch.nn.MSELoss() #distance of target to labels

    def train(self, epoch):
        self.model.train()

        loss_list = []

        start_time = time.time()

        for batch_idx, data in enumerate(self.train_loader):
            stop_time = time.time()
            diff = stop_time - start_time
            print(diff)
            start_time = stop_time

            data = data.cuda(async=True)
            target = data.cuda(async=True) #compare to input

            #self.loggers["tensorboard"].plot_model_structure(self.model, data.shape)

            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            #self.clog.show_value(loss.item(), name="loss_training")
            #self.clog.show_image_grid(data[:5], name="train_images")

            if batch_idx % self.config.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.item()))

            loss_list.append(loss.item())

        loss_per_epoch = np.mean(np.asarray(loss_list))
        print('Train Epoch: {} \tMean Loss: {}'.format(epoch, loss_per_epoch))
        self.clog.show_value(loss_per_epoch, name="loss_training")
        self.clog.show_image_grid(data[:2], name="train_images")
        self.clog.show_image_grid(output[:2], name="reproduced_train_images")

    def validate(self, epoch):

        self.model.eval()
        #val_loss = 0
        #correct = 0
        val_loss_list = []

        with torch.no_grad():
            for data in self.test_loader[i]:
                data = data.cuda(async=True)
                target = data.cuda(async=True)

                output = self.model(data)
                loss = self.criterion(output, target)
                val_loss_list.append(loss.item())

        val_loss_per_epoch = np.mean(np.asarray(val_loss_list))
        self.clog.show_value(val_loss_per_epoch, name="loss_testing")
        self.clog.show_image_grid(data[:5], name="test_images")
        self.clog.show_image_grid(output[:5], name="reproduced_test_images")






class ConvAE_CIFAR(PytorchExperiment):

    def setup(self):

        self.elog.print(self.config)
        data_loader_kwargs = {'num_workers': 12, 'pin_memory': True}

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('C:/Users/Nuong/Desktop/heidelberg/MA-Arbeitsordner/data', train=True, download=True,
                          transform=transforms.Compose([
                               transforms.ToTensor(),
                               #transforms.Normalize((0.1307,), (0.3081,))
                           ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('C:/Users/Nuong/Desktop/heidelberg/MA-Arbeitsordner/data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               #transforms.Normalize((0.1307,), (0.3081,))
                           ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)

        '''self.dataset_train = VideoDataset(os.path.join(self.config.data_path,self.config.train_data))
        self.dataset_test = VideoDataset(os.path.join(self.config.data_path,self.config.test_data))

        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.config.run.hyperparameter.batch_size,
                                                        shuffle=True, **data_loader_kwargs)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.config.run.hyperparameter.batch_size, shuffle = True)'''

        print('Done with loading.')

        ### Models
        self.model = Net()
        if torch.cuda.is_available():
            self.model.cuda() # to gpu

        ### Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.run.hyperparameter.lr)

        ### Load pre-trained model
        if "load_path" in self.config:
            self.load_checkpoint(path=self.config.load_path, name="")

        ### Criterion
        #self.criterion = torch.nn.PairwiseDistance()    #Norm, default 2, in between data pair
        self.criterion = torch.nn.MSELoss() #distance of target to labels

    def train(self, epoch):
        self.model.train()

        loss_list = []

        start_time = time.time()

        for batch_idx, data in enumerate(self.train_loader):
            stop_time = time.time()
            diff = stop_time - start_time
            print(diff)
            start_time = stop_time

            #data = data.cuda(async=True)
            #target = data.cuda(async=True) #compare to input

            #self.loggers["tensorboard"].plot_model_structure(self.model, data.shape)

            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()
            #self.clog.show_value(loss.item(), name="loss_training")
            #self.clog.show_image_grid(data[:5], name="train_images")

            if batch_idx % self.config.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.item()))

            loss_list.append(loss.item())

        loss_per_epoch = np.mean(np.asarray(loss_list))
        print('Train Epoch: {} \tMean Loss: {}'.format(epoch, loss_per_epoch))
        self.clog.show_value(loss_per_epoch, name="loss_training")
        self.clog.show_image_grid(data[:2], name="train_images")
        self.clog.show_image_grid(output[:2], name="reproduced_train_images")

    def validate(self, epoch):

        self.model.eval()
        #val_loss = 0
        #correct = 0
        val_loss_list = []

        with torch.no_grad():
            for data in self.test_loader[i]:
                #data = data.cuda(async=True)
                #target = data.cuda(async=True)

                output = self.model(data)
                loss = self.criterion(output, target)
                val_loss_list.append(loss.item())

        val_loss_per_epoch = np.mean(np.asarray(val_loss_list))
        self.clog.show_value(val_loss_per_epoch, name="loss_testing")
        self.clog.show_image_grid(data[:5], name="test_images")
        self.clog.show_image_grid(output[:5], name="reproduced_test_images")


