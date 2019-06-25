import os
import time

import numpy as np
import torch

#from common.loss.msssim import SSIM, MSSSIM
from common.dataset.video_dataset import DeepFeatureDataset
from common.networks.CL_autoencoder import *
from trixi.experiment.pytorchexperiment import PytorchExperiment
import inspect
import common
from scipy.ndimage.morphology import distance_transform_edt as edt


class Selector(PytorchExperiment):

    def setup(self):

        self.elog.print(self.config)
        data_loader_kwargs = {'num_workers': 12, 'pin_memory': True}

        self.dataset_train =  DeepFeatureDataset(self.config.path_train)

        self.dataset_validate =  DeepFeatureDataset(self.config.path_validate)

        #self.dataset_test =  DeepFeatureDataset(self.config.path_test)

        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.config.run.hyperparameter.batch_size,
                                                        shuffle=True, **data_loader_kwargs)
        self.validate_loader = torch.utils.data.DataLoader(self.dataset_validate, batch_size=self.config.run.hyperparameter.batch_size, shuffle = True, **data_loader_kwargs)
        #self.test_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.config.run.hyperparameter.batch_size, shuffle = True, **data_loader_kwargs)

        print('Done with loading.')

        ### Models
        for name, obj in inspect.getmembers(common.networks.CL_autoencoder):
            if inspect.isclass(obj):
                if obj.__name__==self.config.using_model:
                    self.model=obj()
        if torch.cuda.is_available():
            self.model.cuda()

        ### Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.run.hyperparameter.lr, weight_decay=self.config.run.hyperparameter.weight_decay)

        ### Load pre-trained model
        if self.config.load_old_checkpoint:
            self.load_checkpoint(path=self.config.load_path, name="checkpoint_current.pth.tar")

        ### Criterion
        self.mse = torch.nn.MSELoss()
        #self.msssim = MSSSIM(window_size=self.config.run.msssim_wsz)

    def do_select(self, output):
        probability=list((output-np.min(output))/(np.max(output)- np.min(output)))

        selected=np.zeros(np.shape(output))
        for i in range (len(output)):
            selected[i]=np.random.choice(2,1, p=[1-probability[i],probability[i]])

        return selected

    def do_interpolate(self, orig, selection):
        _, idx=edt(np.logical_not(selection), return_index=True)
        return orig[idx]

    def calc_varloss_selection(self, selection):
        med=np.median(selection)
        cumsum=0
        for p in list(selection):
            cumsum += np.square(p-med)
        return 1/(cumsum/len(selection)+0.001)

    def do_write(self, string):
        f=open(os.path.join(self.elog.work_dir,self.config.text_log),'a+')
        f.write(string)
        f.close()

    def train(self, epoch):
        self.model.train()
        #self.config.n_epochs = 3

        loss_list = []

        start_time = time.time()

        for batch_idx, data in enumerate(self.train_loader):

            data = data.cuda(non_blocking=True)
        
            self.optimizer.zero_grad()
            output = self.model(data)
            output_selection = self.do_select(output)
            interpolated_selection = self.do_interpolate(data,output_selection)
            var_loss=self.calc_varloss_selection(output_selection)
            loss = self.mse(data, interpolated_selection) + np.sum(output_selection) + self.calc_varloss_selection(output_selection)
            loss.backward()
            self.optimizer.step()
            #self.clog.show_value(loss.item(), name="loss_training")
            #self.clog.show_image_grid(data[:5], name="train_images")

            '''if batch_idx % self.config.log_interval == 0:
                self.do_write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.item()))'''

            loss_list.append(loss.item())

        stop_time = time.time()
        diff = stop_time - start_time

        loss_per_epoch = np.mean(np.asarray(loss_list))
        self.do_write('Train Epoch: {} \tMean Train Loss: {} \tDuration: {:1f}s'.format(epoch, loss_per_epoch, diff))
        self.clog.show_image_grid(data[:2], name="train_images")
        self.clog.show_image_grid(output[:2], name="reproduced_train_images")
        self.add_result(loss_per_epoch, name='train', counter=epoch, tag='Loss')
        self.add_result(np.log(loss_per_epoch), name='train', counter=epoch, tag='logLoss')

    def validate(self, epoch):
        self.model.eval()
        val_loss_list = []

        start_time = time.time()
        count_iterations = 0
        with torch.no_grad():
            for data in self.validate_loader:
                data = data.cuda(non_blocking=True)

                output = self.model(data)
                output_selection = self.do_select(output)
                interpolated_selection = self.do_interpolate(data, output_selection)
                loss = self.mse(data, interpolated_selection) + np.sum(output_selection)
                val_loss_list.append(loss.item())

                '''if count_iterations % self.config.log_interval == 0:
                    self.do_write('Validate Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, len(data) *count_iterations, len(self.validate_loader.dataset),
                               100. * count_iterations / len(self.validate_loader), loss.item()))'''
                count_iterations += 1

        stop_time = time.time()
        diff = stop_time - start_time

        val_loss_per_epoch = np.mean(np.asarray(val_loss_list))
        self.do_write('Validate Epoch: {} \tMean Validate Loss: {} \tDuration: {:1f}s'.format(epoch, val_loss_per_epoch, diff))
        self.clog.show_image_grid(data[:4], name="validate_images")
        self.clog.show_image_grid(output[:4], name="reproduced_validation_images")
        self.add_result(val_loss_per_epoch, name='val', counter=epoch, tag='Loss')
        self.add_result(np.log(val_loss_per_epoch), name='val', counter=epoch, tag='logLoss')


##### Test Data 
    def test_set_eval(self):

        self.model.eval()
        test_loss_list = []

        with torch.no_grad():
            for data in self.test_loader:
                data = data.cuda(non_blocking=True)
                target = data.cuda(non_blocking=True)

                output = self.model(data)[0]
                loss = self.mse(output, target)*float(self.config.run.loss1) + (1 - self.msssim(output, target))*float(self.config.run.loss2)
                test_loss_list.append(loss.item())

        test_loss = np.mean(np.asarray(test_loss_list))
        self.do_write('Test Loss: {}'.format(test_loss))


    def end(self):
        #self.test_set_eval()
        #print('Test done.')

        torch.save(self.model.state_dict(), os.path.join(self.config.save_path,f'{self.config.using_model}_{self.config.name}.pt'))
        #print('Model saved.')


