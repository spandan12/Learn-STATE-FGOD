import torch
import torchvision
import numpy as np
import os

import torchvision.transforms as transforms
from torchvision import datasets 
from typing import Dict, List, Tuple

class CustomImageFolder(datasets.ImageFolder):

    def __init__(
        self,
        root: str,
        transform= None,
        novel_offset = 0
    ):
        
        self.novel_offset = novel_offset
        super().__init__(root, transform=transform)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: int(cls_name.split('_')[0])+ self.novel_offset for i, cls_name in enumerate(classes)}
        
        return classes, class_to_idx




def get_CUB_loader( cfg=None):
    trainset = CustomImageFolder( root='./D_ALL/CUB/train/', transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]), novel_offset = 0)

    known_test = CustomImageFolder(root=f'./D_ALL/CUB/known/', transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]), novel_offset = 0)
    
    novel_offset =  len(trainset.classes)
    novel_test = CustomImageFolder( root=f'./D_ALL/CUB/novel/', transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]), novel_offset = novel_offset)
    

    train_loader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True, drop_last = False,)
    known_test_loader = torch.utils.data.DataLoader(known_test, batch_size = 512, shuffle = False, drop_last = False)
    novel_test_loader = torch.utils.data.DataLoader(novel_test, batch_size = 512, shuffle = False, drop_last = False)

    train_loader.dataset.name = 'cub_train'
    known_test_loader.dataset.name = 'cub_known'
    novel_test_loader.dataset.name = 'cub_novel'

    return train_loader, known_test_loader, novel_test_loader



def get_TINYIMAGENET_loader( cfg=None):
    trainset = CustomImageFolder( root='./D_ALL/TINYIMAGENET/train/', transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]), novel_offset = 0)

    known_test = CustomImageFolder(root=f'./D_ALL/TINYIMAGENET/known/', transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]), novel_offset = 0)
    
    novel_offset =  len(trainset.classes)
    novel_test = CustomImageFolder( root=f'./D_ALL/TINYIMAGENET/novel/', transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]), novel_offset = novel_offset)
    

    train_loader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True, drop_last = False,)
    known_test_loader = torch.utils.data.DataLoader(known_test, batch_size = 512, shuffle = False, drop_last = False)
    novel_test_loader = torch.utils.data.DataLoader(novel_test, batch_size = 512, shuffle = False, drop_last = False)

    train_loader.dataset.name = 'tinyimagenet_train'
    known_test_loader.dataset.name = 'tinyimagenet_known'
    novel_test_loader.dataset.name = 'tinyimagenet_novel'

    return train_loader, known_test_loader, novel_test_loader

def get_AWA2_loader( cfg=None):
    trainset = CustomImageFolder( root='./D_ALL/AWA2/train/', transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]), novel_offset = 0)

    known_test = CustomImageFolder(root=f'./D_ALL/AWA2/known/', transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]), novel_offset = 0)
    
    novel_offset =  len(trainset.classes)
    novel_test = CustomImageFolder( root=f'./D_ALL/AWA2/novel/', transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]), novel_offset = novel_offset)
    

    train_loader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True, drop_last = False,)
    known_test_loader = torch.utils.data.DataLoader(known_test, batch_size = 512, shuffle = False, drop_last = False)
    novel_test_loader = torch.utils.data.DataLoader(novel_test, batch_size = 512, shuffle = False, drop_last = False)

    train_loader.dataset.name = 'AWA2_train'
    known_test_loader.dataset.name = 'AWA2_known'
    novel_test_loader.dataset.name = 'AWA2_novel'

    return train_loader, known_test_loader, novel_test_loader

def main():
    get_CUB_loader()

if __name__=='__main__':
    main()