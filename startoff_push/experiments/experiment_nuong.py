import os
import time

import numpy as np
import torch

from common.dataset.video_dataset import VideoDataset
from torchvision import datasets, transforms
from common.networks.autoencoder import AutoencoderFrame
from common.networks.autoencoder import DAE
import common.networks.nuong_AE
from trixi.experiment import PytorchExperiment

import glob
import pickle
import pytorch_ssim

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
        if "load_path" in self.config and self.config.load_old_checkpoint:
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




import inspect

class ConvAE_CIFAR(PytorchExperiment):

    def setup(self):

        def normalize_data(train_set):
            R_list=[]
            G_list=[]
            B_list=[]
            for this_img,_ in iter(train_set):
                this_img_np=np.asarray(this_img)
                R_list.append(this_img_np[0].flatten())
                G_list.append(this_img_np[1].flatten())
                B_list.append(this_img_np[2].flatten())

            return (np.mean(np.asarray(R_list)).item(),np.mean(np.asarray(G_list)).item(),np.mean(np.asarray(B_list)).item()),(np.std(np.asarray(R_list)).item(),np.std(np.asarray(G_list)).item(),np.std(np.asarray(B_list)).item())

        self.elog.print(self.config)
        data_loader_kwargs = {'num_workers': 12, 'pin_memory': True}

        use_cuda = not self.config.no_cuda and torch.cuda.is_available()
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        if self.config.run.normalize_data:
            train_set=datasets.CIFAR10(self.config.data_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                        #transforms.Normalize((0.1307,), (0.3081,))
                                                        ]))
            mean_, std_ = normalize_data(train_set)

            transforms_=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean_, std_)
                                            ])
        else:
            transforms_=transforms.Compose([
                                            transforms.ToTensor(),
                                            ])

        self.train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(self.config.data_path, train=True, download=True, 
                                                        transform=transforms_),
                                                        batch_size=self.config.run.hyperparameter.batch_size, shuffle=True, **kwargs)

        self.test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(self.config.data_path, train=False, 
                                                        transform=transforms_),
                                                        batch_size=self.config.run.hyperparameter.batch_size, shuffle=False, **kwargs)
        

        '''self.dataset_train = VideoDataset(os.path.join(self.config.data_path,self.config.train_data))
        self.dataset_test = VideoDataset(os.path.join(self.config.data_path,self.config.test_data))

        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.config.run.hyperparameter.batch_size,
                                                        shuffle=True, **data_loader_kwargs)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.config.run.hyperparameter.batch_size, shuffle = True)'''

        print('Done with loading.')
        ### Models
        for name, obj in inspect.getmembers(common.networks.nuong_AE):
            if inspect.isclass(obj):
                if obj.__name__==self.config.using_model:
                    self.model=obj()
                    
        #if self.config.load_old_model:
        #    self.model.load_state_dict(torch.load(self.config.model_load_path))

        if torch.cuda.is_available():
            self.model.cuda() # to gpu

        ### Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.run.hyperparameter.lr)

        ### Load pre-trained model
        if "load_path" in self.config and self.config.load_old_checkpoint:
            #self.load_checkpoint(path=self.config.load_path, name="")
            self.load_checkpoint(path=self.config.load_path, name="checkpoint_current.pth.tar")

        ### Criterion
        #self.criterion = torch.nn.PairwiseDistance()    #Norm, default 2, in between data pair
        self.criterion = torch.nn.MSELoss() #distance of target to labels
        self.criterion2 = pytorch_ssim.SSIM(window_size=5)

    def do_write(self, string):
        f=open(os.path.join(self.elog.work_dir,self.config.text_log),'a+')
        f.write(string)
        f.close()

    def train(self, epoch):
        self.model.train()

        loss_list = []
        loss_SSIM_list = []
        loss_MSE_list = []
        start_time = time.time()

        for batch_idx, (data,target) in enumerate(self.train_loader):
            stop_time = time.time()
            diff = stop_time - start_time
            start_time = stop_time

            #data = data.cuda(async=True)
            #target = data.cuda(async=True) #compare to input

            #self.loggers["tensorboard"].plot_model_structure(self.model, data.shape)

            self.optimizer.zero_grad()
            output = self.model(data)[0]

            loss1 = self.criterion(output, data)*0.3
            loss2 = (1-self.criterion2(output, data))*0.7
            loss=loss1+loss2 # 0.0141 MSE on unnormalized data, 0.5659 MSE on normalized data, 0.33 ssim on both

            loss.backward()
            self.optimizer.step()
            #self.clog.show_value(loss.item(), name="loss_training")
            #self.clog.show_image_grid(data[:5], name="train_images")

            if batch_idx % self.config.log_interval == 0:
                self.do_write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \n'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.item()))
            
            
            
            loss_list.append(loss.item())
            loss_SSIM_list.append(loss2.item())
            loss_MSE_list.append(loss1.item())
            
        loss_per_epoch = np.mean(np.asarray(loss_list))
        self.do_write('Train Epoch: {} \tMean Loss: {}\n'.format(epoch, loss_per_epoch))
        self.do_write('Train Epoch: {} \tMean MSE Loss: {}\n'.format(epoch, np.mean(np.asarray(loss_MSE_list))))
        self.do_write('Train Epoch: {} \tMean SSIM Loss: {}\n'.format(epoch, np.mean(np.asarray(loss_SSIM_list))))
        self.clog.show_value(loss_per_epoch, name="loss_training")
        self.clog.show_image_grid(data[:2], name="train_images")
        self.clog.show_image_grid(output[:2], name="reproduced_train_images")

    def validate(self, epoch):

        self.model.eval()
        #val_loss = 0
        #correct = 0
        val_loss_list = []
        val_loss_MSE_list = []
        val_loss_SSIM_list = []

        with torch.no_grad():
            for data,target in self.test_loader:
                #data = data.cuda(async=True)
                #target = data.cuda(async=True)

                output = self.model(data)[0]
                loss1 = self.criterion(output, data)*0.3
                loss2 = (1-self.criterion2(output, data))*0.7
                loss=loss1+loss2
                val_loss_list.append(loss.item())
                val_loss_SSIM_list.append(loss2.item())
                val_loss_MSE_list.append(loss1.item())

        val_loss_per_epoch = np.mean(np.asarray(val_loss_list))
        self.do_write('\nTest set: Average loss: {:.4f}\n'.format(
            val_loss_per_epoch))
        self.do_write('\nTest set: Average MSE loss: {:.4f}\n'.format(
            np.mean(np.asarray(val_loss_MSE_list))))
        self.do_write('\nTest set: Average SSIM loss: {:.4f}\n'.format(
            np.mean(np.asarray(val_loss_SSIM_list))))
        self.clog.show_value(val_loss_per_epoch, name="loss_testing")
        self.clog.show_image_grid(data[:5], name="test_images")
        self.clog.show_image_grid(output[:5], name="reproduced_test_images")


