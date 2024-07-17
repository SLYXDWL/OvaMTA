import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import pandas as pd
from pathlib import Path
import random
import torch
import numpy as np

class SegDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self,
                 trainsize,
                 augmentations,
                 file_excel = r"C:/Users/ZD030/Desktop/lyt/301/多分类/20230314-image-based.xlsx",
                 mode = "train"):
        self.augmentations = augmentations
        self.trainsize = trainsize
        self.mode = mode
        self.root = Path("C:/Users/ZD030/Desktop/lyt/301/多分类/")
        df = pd.read_excel(file_excel, sheet_name=mode)
        self.infos = df


        self.size = len(self.infos)
        if self.augmentations == True and mode == "train":
            print('Using RandomRotation, RandomFlip while Training')
            self.img_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomRotation(90, resample=False, expand=False, center=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomRotation(90, resample=False, expand=False, center=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])

        else:
            # print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])

        # self.img_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406],
        #                          [0.229, 0.224, 0.225])])
        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor()])

    def __getitem__(self, item):
        info = self.infos.iloc[item]

        image = Image.open(self.root / info["tumor"]).convert("RGB")
        if self.mode != 'ra_test':
            gt = Image.open(self.root / info["roi"]).convert("L")
        else:
            gt = Image.open(self.root / info["tumor"]).convert("RGB")
        label = info["HCC"]
        # print(label)
        # clinical = [info['Age'], info['Gender']-1, info['HBV'], info['HCV'], info['Diabetes'],
        #             info['Alcoholism'], info['TUMORhistory'], info['FamilyTUMORhistory']]

        clinical = [info['Age']/90, info['Gender']-1, info['HBV'], info['HCV'],
                    info['Age']/90, info['Gender'] - 1, info['HBV'], info['HCV'],
                    info['Age']/90, info['Gender'] - 1, info['HBV'], info['HCV'],
                    info['Age']/90, info['Gender'] - 1, info['HBV'], info['HCV']]
        # print(clinical)
        clinical = torch.tensor(clinical)


        seed = np.random.randint(2147483647) # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)

        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        # image = np.array(image)
        # gt = np.array(gt)
        # clinical = np.array(clinical)
        # image[2,:,:] = gt[:,:]
        # # image[1, :, :] = gt[:, :] * image[1, :, :]
        name = info['tumor']
        return image, gt, clinical, label, name

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

def get_loader(mode, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, augmentations=False):

    dataset = SegDataset(mode=mode, trainsize=trainsize, augmentations=augmentations)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

class test_dataset:
    def __init__(self,
                 testsize,
                 file_excel=r"C:/Users/ZD030/Desktop/lyt/301/多分类/20230310-image-based.xlsx",
                 mode="train"):
        self.testsize = testsize
        self.root = Path("C:/Users/ZD030/Desktop/lyt/301/多分类/")
        df = pd.read_excel(file_excel, sheet_name=mode)
        self.infos = df

        self.gt_transform = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.size = len(self.infos)
        self.index = 0

    def load_data(self):
        info = self.infos.iloc[self.index]

        image = Image.open(self.root / info["tumor"]).convert("RGB")
        gt = Image.open(self.root / info["roi"]).convert("L")
        image = self.transform(image).unsqueeze(0)
        # gt = self.gt_transform(gt)
        label = info['HCC']
        name = info["tumor"].split('\\')[-1]
        # if info['HBV'] ==0 and info['HCV'] ==0:
        #     b = [1,0,0,0]
        # elif info['HBV'] ==0:
        #     b = [0,1,0,0]
        # elif info['HBV'] ==1 and info['HCV'] ==0:
        #     b = [0,0,1,0]
        # elif info['HBV'] ==1:
        #     b = [0,0,0,1]
        # clinical = [[info['Age']/100, 2-info['Gender'], b[0], b[1],b[2],b[3],
        #             info['Age'] / 100, 2-info['Gender'], b[0], b[1], b[2], b[3],
        #             info['Age'] / 100, 2-info['Gender'], b[0], b[1], b[2], b[3],
        #             info['Age'] / 100, 2-info['Gender'], b[0], b[1], b[2], b[3]
        #             ]]
        # # print(clinical)
        # clinical = torch.tensor(clinical)
        # gt = Image.open(self.predroot / name).convert("L")

        # image = np.array(image)
        # gt = np.array(gt)
        # print(image.shape,gt.shape)
        # image[:,2,:,:] = gt[:,:,:]
        # image.ToTensor()
        self.index += 1
        return image, gt, label, name


