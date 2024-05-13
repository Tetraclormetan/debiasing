#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import os
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from typing import Tuple, Generator, Union

data_transform = transforms.Compose([
    # Modifica le dimensioni delle immagini come necessario
    transforms.ToTensor(),
])

class CMNIST(Dataset):
    def __init__(self, data_dir="./data/cmnist", env="train", bias_amount=0.95, transform=data_transform, return_index = False):
        self.data_dir = data_dir
        self.transform = transform
        self.env=env
        self.bias_amount=bias_amount
        self.return_index = return_index
        self.num_classes = 10

        self.bias_folder_dict = {
            0.95: "5pct",
            0.98: "2pct",
            0.99: "1pct",
            0.995: "0.5pct",

        }
        if self.env == "train":
            self.samples, self.class_labels, self.bias_labels = self.load_train_samples()

        if self.env == "val":
            self.samples, self.class_labels, self.bias_labels = self.load_val_samples()

        if self.env == "test":
            self.samples, self.class_labels, self.bias_labels = self.load_test_samples()

    def load_train_samples(self):
        samples_path = []
        class_labels=[]
        bias_labels=[]
        bias_folder=self.bias_folder_dict[self.bias_amount]
        for class_folder in sorted(os.listdir(os.path.join(self.data_dir,bias_folder, "align"))):
            for filename in sorted(os.listdir(os.path.join(self.data_dir,bias_folder, "align",class_folder))):
                samples_path.append(os.path.join(self.data_dir,bias_folder, "align",class_folder,filename))
                class_labels.append(self.assign_class_label(filename))
                bias_labels.append(self.assign_bias_label(filename))
        
        for class_folder in sorted(os.listdir(os.path.join(self.data_dir,bias_folder, "conflict"))):
            for filename in sorted(os.listdir(os.path.join(self.data_dir,bias_folder, "conflict",class_folder))):
                samples_path.append(os.path.join(self.data_dir,bias_folder, "conflict",class_folder,filename))
                class_labels.append(self.assign_class_label(filename))
                bias_labels.append(self.assign_bias_label(filename))

        return (
            np.array(samples_path), 
            np.array(class_labels), 
            np.array(bias_labels)
        ) 
    
    def load_val_samples(self):
        samples_path = []
        class_labels=[]
        bias_labels=[]

        bias_folder=self.bias_folder_dict[self.bias_amount]
        for filename in sorted(os.listdir(os.path.join(self.data_dir,bias_folder,"valid"))):
            samples_path.append(os.path.join(self.data_dir,bias_folder, "valid",filename))
            class_labels.append(self.assign_class_label(filename))
            bias_labels.append(self.assign_bias_label(filename))

        return (
            np.array(samples_path), 
            np.array(class_labels), 
            np.array(bias_labels)
        ) 
    
    def load_test_samples(self):
        samples_path = []
        class_labels=[]
        bias_labels=[]

        for class_folder in sorted(os.listdir(os.path.join(self.data_dir,"test"))):
            for filename in  sorted(os.listdir(os.path.join(self.data_dir,"test",class_folder))):
                samples_path.append(os.path.join(self.data_dir,"test",class_folder,filename))
                class_labels.append(self.assign_class_label(filename))
                bias_labels.append(self.assign_bias_label(filename))

        return (
            np.array(samples_path), 
            np.array(class_labels), 
            np.array(bias_labels)
        ) 


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        class_label=self.class_labels[idx]
        bias_label=self.bias_labels[idx]

        image = self.transform(Image.open(file_path))   #senza self.transofrm per vedere le immagini 
        
        if self.return_index:
            return image, class_label, bias_label, idx

        return image, class_label, bias_label

    def assign_bias_label(self, filename):
        no_extension=filename.split('.')[0]
        _, y, z = no_extension.split('_')
        y, z = int(y), int(z)
        if y == z:
            return 1
        return -1
    
    def assign_class_label(self, filename):
        no_extension=filename.split('.')[0]
        _, y, _ = no_extension.split('_')
        return int(y)
    
    def perclass_populations(self, return_labels: bool = False) -> Union[Tuple[float, float], Tuple[Tuple[float, float], torch.Tensor]]:
        labels: torch.Tensor = torch.zeros(len(self))
        for i in range(len(self)):
            labels[i] = self[i][1]

        _, pop_counts = labels.unique(return_counts=True)

        if return_labels:
            return pop_counts.long(), labels.long()

        return pop_counts
    
    def get_bias_labels(self) -> Generator[None, None, torch.Tensor]:
        for i in range(len(self)):
            yield self[i][2]

    def __repr__(self) -> str:
        return f"CMNIST(env={self.env}, bias_amount={self.bias_amount}, num_classes={self.num_classes})"
     

if __name__ == "__main__":


    train_set=CMNIST(env="train",bias_amount=0.95)
    val_set=CMNIST(env="val",bias_amount=0.95)
    test_set=CMNIST(env="test",bias_amount=0.95)


    #group and display colorized images of the same digit together.
    plt.figure()
    for i in range(0, 55000, 5000):
        train_image, l, bl = train_set[i]
        print(train_set.samples[i])
        print("class ", l)
        print("bias ", bl)
        plt.imshow(train_image.permute(1,2,0))

        plt.show()
        plt.savefig(f"data/figure{i}.png")

    # for i in range(0, 300, 50):
    #     val_image, l, bl = val_set[i]
    #     print(val_set.samples[i])
    #     print("class ", l)
    #     print("bias ", bl)
    #     plt.imshow(val_image)

    #     plt.show()

    # for i in range(0, 4000, 100):
    #     test_image, l, bl = test_set[i]
    #     print(test_set.samples[i])
    #     print("class ", l)
    #     print("bias ", bl)
    #     plt.imshow(test_image)

    #     plt.show()