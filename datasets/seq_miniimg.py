import os
from typing import Optional

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image
from torch.utils.data import Dataset

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path

from glob import glob
import yaml
import cv2
import torch

class MiniImagenet(Dataset):

    INPUT_SIZE = (288, 384)

    def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None,
                target_transform: Optional[nn.Module] = None, subset:float = 1.) -> None: 
        self.not_aug_transform = transforms.Compose([
            transforms.Resize(MiniImagenet.INPUT_SIZE, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor()])
        self.root = root 
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        assert 0 < subset <= 1.
        self.subset = subset


        with open('data/seq_miniimg.yml','r') as readfile:
            self.all_classes = yaml.load(readfile, Loader=yaml.FullLoader)  
            
        phase_str = 'train' if train else 'val'           
        
        self.data_path = []
        self.targets = []
        for cls_name in self.all_classes:
            img_list = glob(os.path.join(root, 'images', phase_str, cls_name, '*.JPEG'))
            if self.subset < 1. :
                img_list = img_list[:int(len(img_list)*self.subset)]
            self.data_path.extend(img_list)
            self.targets.extend([self.all_classes[cls_name]] * len(img_list))
        
        self.data = np.array(self.data_path)
        self.targets = np.array(self.targets)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, target = self.data[index], self.targets[index]
        #img_path is only the path where img is located
        img = cv2.imread(img_path)
        img = np.ascontiguousarray(img[:, :, ::-1])
        img = transforms.ToPILImage()(img)
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]
        
        return img, target  


class MyMiniImagenet(MiniImagenet):
   
    def __getitem__(self, index):
        img_path, target = self.data[index], self.targets[index]
        #img is only the path where img is located
        img = cv2.imread(img_path)
        img = np.ascontiguousarray(img[:, :, ::-1])
        img = transforms.ToPILImage()(img)
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]
        
        return img, target, not_aug_img


class SequentialMiniImagenet(ContinualDataset):

    NAME = 'seq-miniimg'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 5
    N_TASKS = 20
    TRANSFORM = transforms.Compose(
        [transforms.Resize(MiniImagenet.INPUT_SIZE, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    
    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = self.TRANSFORM
        

        train_dataset = MyMiniImagenet(self.args.dataset_path + 'seqMINIIMG',
                                      train = True, transform=transform, subset=self.args.dataset_subset)
                            
        if self.args.validation:
            raise NotImplementedError()
        else:
            test_dataset = MiniImagenet(self.args.dataset_path + 'seqMINIIMG',
                                      train = False, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test 

    @staticmethod
    def get_backbone() -> nn.Module:
        return resnet18(SequentialMiniImagenet.N_CLASSES_PER_TASK 
                            * SequentialMiniImagenet.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_epochs():
        return 1

    @staticmethod
    def get_batch_size():
        return 8

    @staticmethod
    def get_minibatch_size():
        return SequentialMiniImagenet.get_batch_size()



class MiniImagenetSal(MiniImagenet):

    #TARGET_SIZE = (480, 640)
    TARGET_SIZE = (288, 384)
    def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None, target_transform: Optional[nn.Module] = None, subset:float = 1.) -> None:
        super().__init__(root, train, transform, target_transform, subset)
        self.map_transform = transforms.Compose([
            transforms.Resize(MiniImagenetSal.TARGET_SIZE, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Lambda(MiniImagenetSal.normalize_tensor)
        ])
         
    def normalize_tensor(tensor, rescale=False):
        tmin = torch.min(tensor)
        if rescale or tmin < 0:
            tensor -= tmin
        tsum = tensor.sum()
        if tsum > 0:
            return tensor / tsum
        print("Zero tensor")
        tensor.fill_(1. / tensor.numel())
        return tensor
    

    def __getitem__(self, index): # we have to add saliency map
        img, target = self.data[index], self.targets[index]
        if isinstance(img, str):
            #img_path is only the path where img is located; 
            img = cv2.imread(img)
            img = np.ascontiguousarray(img[:, :, ::-1])
            img = transforms.ToPILImage()(img)
            original_img = img.copy()
        else:
            raise NotImplementedError('In Saliency-aided Datasets img should be a string.')

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # #get sal_map
        if hasattr(self, 'maps'):
            map = self.maps[index]
        else:
            map_path = self.data[index].replace('images', 'annotations').replace('JPEG', 'png')
            map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
            map = transforms.ToPILImage()(map)    

        if self.map_transform is not None:
            map = self.map_transform(map)

        if hasattr(self, 'logits'):  
            return (img, map), target, original_img, self.logits[index]
        
        return (img, map), target


class MyMiniImagenetSal(MiniImagenetSal):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if isinstance(img, str):
            #img_path is only the path where img is located; it's something like : /mnt/SSD/datasets/MiniImagenet/images/train/n01440764/n01440764_519.JPEG
            img = cv2.imread(img)
            img = np.ascontiguousarray(img[:, :, ::-1])
            img = transforms.ToPILImage()(img)
            original_img = img.copy()
        else:
            raise NotImplementedError('In Saliency-aided Datasets img should be a string.')

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        #get sal_map
        if hasattr(self, 'maps'):
            map = self.maps[index]
        else:
            map_path = self.data[index].replace('images', 'annotations').replace('JPEG', 'png')
            map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
            map = transforms.ToPILImage()(map)
            
        if self.map_transform is not None:
            map = self.map_transform(map)

        if hasattr(self, 'logits'):
            return (img, map), target, not_aug_img, self.logits[index]

        return (img, map), target, not_aug_img


class SequentialMiniImagenetSal(ContinualDataset):

    NAME = 'seq-miniimgsal'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 5
    N_TASKS = 20
    TRANSFORM = transforms.Compose(
        [transforms.Resize(MiniImagenetSal.INPUT_SIZE, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    
    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = self.TRANSFORM
        
        train_dataset = MyMiniImagenetSal(self.args.dataset_path + 'seqMINIIMG',
                                      train = True, transform=transform, subset=self.args.dataset_subset)
                            
        if self.args.validation:
            raise NotImplementedError()
        else:
            test_dataset = MiniImagenetSal(self.args.dataset_path + 'seqMINIIMG',
                                      train = False, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_backbone() -> nn.Module:
        return resnet18(SequentialMiniImagenetSal.N_CLASSES_PER_TASK 
                            * SequentialMiniImagenetSal.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_epochs():
        return 1

    @staticmethod
    def get_batch_size():
        return 8

    @staticmethod
    def get_minibatch_size():
        return SequentialMiniImagenetSal.get_batch_size()