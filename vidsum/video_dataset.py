import os
import multiprocessing
import numpy as np
import torch

from joblib import Parallel, delayed
from torch.utils.data import Dataset
from torchvision.transforms import transforms

###### Data Loader for several videos in one folder (args: folder_path)

class DeepFeatureDataset(Dataset):
    def __init__(self, main_path):
        super(DeepFeatureDataset, self).__init__()
        self.main_path = main_path
        self.video_paths = []
        self.video_lengths = []
        self.max_video_lenth = 1
        self.trafo = transforms.Compose([
            transforms.ToTensor()
        ])
        self.crawl_videos()
        self.get_video_lengths()

    def crawl_videos(self):
        # return all video paths
        main_path = self.main_path
        files = os.listdir(main_path)
        for file_name in files:
            self.video_paths.append(os.path.join(main_path, '{}'.format(file_name)))

    def get_video_lengths(self):
        for index in range(len(self.video_paths)):
            video = np.load(self.video_paths[index], mmap_mode='r')
            output = self.transform_video(video)
            self.video_lengths.append(output.size()[1])
        self.max_video_lenth = max(self.video_lengths)

    def transform_video(self, video):
        video_as_tensor = self.trafo(video)
        return video_as_tensor

    def __getitem__(self, index):
        video = np.load(self.video_paths[index], mmap_mode='r')
        output = self.transform_video(video).permute(1,0,2)
        return output
    # Note: The output has to be of shape [H,C,W] due to padding of different length in dim=1 in order to create batch
    # Change back to [B, C, H, W] in network

    def __len__(self):
        return len(self.video_paths)


####
# video_loader costum_collate

def my_collate(batch): # so far only padding
    padded_seq_batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True).float()
    return padded_seq_batch