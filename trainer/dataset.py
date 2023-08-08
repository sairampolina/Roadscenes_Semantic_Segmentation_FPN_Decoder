import os

from operator import itemgetter


import torch
from torch.utils.data import Dataset, DataLoader

import cv2
from typing import Callable, Iterable,Optional

from .utils import (download_git_folder,
                    init_semantic_seg_dataset)

class SemSegDataset(Dataset):

    def __init__(self,
                 data_path:str,
                 images_folder:str,
                 masks_folder :str,
                 num_classes : int,
                 transforms : Optional[Iterable[Callable]] = None,
                 class_names: Optional[list] = None,
                 dataset_url: Optional[str] = None,
                 dataset_folder : Optional[str] = None
                 ):
        
        self.num_classes = num_classes
        self.transform = transforms
        self.class_names = class_names
    
        if not os.path.isdir(data_path) and dataset_url is not None and dataset_folder is not None:
            download_git_folder(dataset_url,dataset_folder, data_path)
        
        # [image_paths, mask_paths]
        self.dataset = init_semantic_seg_dataset(data_path,images_folder,masks_folder) 

    def __getitem__(self, index):

        img_path = self.dataset[0][index]
        mask_path = self.dataset[1][index]

        sample = {'image':cv2.imread(img_path,1)[...,::-1],
                'mask' : cv2.imread(mask_path,0)}

        if self.transform is not None:
            sample = self.transform(**sample)
            sample["mask"] = sample["mask"].long()
        
        return sample

    def __len__(self):
        return len(self.dataset[0])
    
    def get_num_classes(self):
        return self.num_classes
    
    def get_class_name(self,idx):
        return self.class_names[idx]
