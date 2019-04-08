import torch
from torch.utils.data import Dataset
from torchvision.datasets.cifar import CIFAR10
from torch.autograd import Variable
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np

class MyCifar(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 isgpu = True):

        dset = CIFAR10(root, train, transform, target_transform, download)
        if train:
            self.data = dset.train_data
            self.labels = dset.train_labels
        else:
            self.data = dset.test_data
            self.labels = dset.test_labels

        # self.data = self.data.float()/255.0 - 0.5
        # self.data = self.data.unsqueeze(1)
        self.transform = transform
        self.target_transform = target_transform

    def show(self,img):
        img = img.numpy()
        img = np.transpose(img,[1,2,0])
        plt.imshow(img)
        plt.show()

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]

        # if self.transform is not None:
            # img = transforms.ToPILImage()(transforms.ToTensor()(img)).convert('RGB')
            # img = self.transform(img)

        # if self.target_transform is not None:
        #     label = self.target_transform(label)
        # self.show(img)
        return img, label

    def __len__(self):
        return self.data.shape[0]