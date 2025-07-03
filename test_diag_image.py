
import torch
import argparse
from datetime import datetime
from lib.swim_transformer import swin_tiny_patch4_window7_224 as create_model
from lib.OvaMTA_diag import TransRaUNet_CLF_xiaorong as OvaMTA_CLF
# from utils.dataloader import get_loader,test_dataset,SegDataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter, WarmupMultiStepLR
import torch.optim.lr_scheduler as lr_scheduler
from utils.focal_loss import Focal_loss, FocalLoss
from utils.ghm_loss import GHMC,GHMR
from utils.smooth_l1_loss import SmoothL1Loss
import sklearn
from sklearn.metrics import roc_curve
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import torch.utils.data as data
import pandas as pd
from pathlib import Path
import numpy as np
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from PIL import Image
import random

class SegDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self,
                 trainsize,
                 augmentations,
                 file_excel = r"C:\Users\Administrator\Desktop\dwl\240122OvarySeg\project\卵巢分割\240314-image-based.xlsx",
                 mode = "train"):
        self.augmentations = augmentations
        self.trainsize = trainsize
        self.mode = mode
        self.root = Path("C:/Users/Administrator/Desktop/卵巢多分类/")
        df = pd.read_excel(r"C:\Users\Administrator\Desktop\dwl\240122OvarySeg\project\卵巢分割\240314-image-based.xlsx",sheet_name=mode)
        # df = pd.read_excel(r"C:\Users\Administrator\Desktop\dwl\231102Ovary\project\卵巢多分类\240108-image-based-BM.xlsx",sheet_name=mode)
        # print(df,mode)
        self.infos = df

        self.size = len(self.infos)
        if self.augmentations == True and mode == "train":
            print('Using RandomRotation, RandomFlip while Training')
            self.img_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomRotation(90, expand=False, center=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomRotation(90, expand=False, center=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])

        elif mode == "train":
            # print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

            self.gt_transform = transforms.Compose([
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
                # transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])

    def __getitem__(self, item):
        info = self.infos.iloc[item]
        image = Image.open(info["tumor"]).convert("RGB")
        gt = Image.open(info["roi"]).convert("L")
        label = info["bbm"]
        if info["bbm"]==2:
            label=1
        else:
            label=0
        name = info['tumor'].split('\\')[-1]

        if info['ca125'] == -1:
            b=[0]
        else:
            b =[1]
        clinical = [info['age'] / 100,info['ca125'],b[0],
                    info['age'] / 100,info['ca125'],b[0],
                    info['age'] / 100, info['ca125'],b[0],
                    info['age'] / 100, info['ca125'],b[0]
                    ]
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
        return image, gt, clinical, label, name
    def __len__(self):
        return self.size

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def log(fd, message, time=True):
    if time:
        message = ' ==> '.join([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message])
    fd.write(message + '\n')
    fd.flush()
    print(message)

def prepare_logger(params):
    # prepare logger directory
    make_dir(params.log_dir)
    make_dir(os.path.join(params.log_dir, params.exp_name))

    logger_path = os.path.join(params.log_dir, params.exp_name, params.category)
    ckpt_dir = os.path.join(params.log_dir, params.exp_name, params.category, 'checkpoints')
    epochs_dir = os.path.join(params.log_dir, params.exp_name, params.category, 'epochs')

    make_dir(logger_path)
    make_dir(ckpt_dir)
    make_dir(epochs_dir)

    logger_file = os.path.join(params.log_dir, params.exp_name, params.category, 'logger.log')
    log_fd = open(logger_file, 'a')

    log(log_fd, "Experiment: {}".format(params.exp_name), False)
    log(log_fd, "Logger directory: {}".format(logger_path), False)
    log(log_fd, str(params), False)

    train_writer = SummaryWriter(os.path.join(logger_path, 'train'))
    val_writer = SummaryWriter(os.path.join(logger_path, 'val'))

    return ckpt_dir, epochs_dir, log_fd, train_writer, val_writer

def get_loader(mode, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, augmentations=False):

    dataset = SegDataset(mode=mode, trainsize=trainsize, augmentations=augmentations)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def _thresh(img):
    thresh=0.5
    img[img > thresh] = 1
    img[img <= thresh] = 0
    return img

def get_fv_prob_auc(train_dataloader,model,log_fd,params):
    model.eval()
    y_true_auc0, y_prob_auc0 = [], []
    y_true_auc1, y_prob_auc1 = [], []
    features_list=[]
    name_list=[]
    for batch_idx, (images, masks, infos, labels, name) in enumerate(train_dataloader):
        images, infos, labels = images.to(device), infos.to(device), labels.to(device)
        # outputs5,outputs4, outputs3,outputs2,outputs1 = model(images,infos)
        outputs5, outputs4, outputs3, outputs2, outputs1, features = model(images)

        #--------------一个批次一个批次来--------------
        _, predicted = torch.max(outputs1.data, dim=1)
        for i in range(len(outputs1.data)):
            # print(outputs1.data.shape)
            feature_vector = []
            for j in range(features.shape[1]):
                feature_vector.append(features[i, j].item())
            features_list.append(feature_vector)
            name_list.append(name[i])

            y_prob_auc0.append(outputs1[i][0].item())
            y_prob_auc1.append(outputs1[i][1].item())

            if labels[i].item() == 0:
                y_true_auc0.append(1)
            else:
                y_true_auc0.append(0)
            if labels[i].item() == 1:
                y_true_auc1.append(1)
            else:
                y_true_auc1.append(0)
            print(batch_idx, name[i], len(feature_vector),labels[i].item(), y_true_auc1[-1],outputs1[i][1].item())

    fpr, tpr, thresholds = roc_curve(y_true_auc0, y_prob_auc0)
    auc0 = sklearn.metrics.auc(fpr, tpr)
    log(log_fd, 'Train\'s Benign auc is: %.3f%%' % (100 * auc0))

    fpr, tpr, thresholds = roc_curve(y_true_auc1, y_prob_auc1)
    auc1 = sklearn.metrics.auc(fpr, tpr)
    log(log_fd, 'Train\'s Malignant auc is: %.3f%%' % (100 * auc1))

    return name_list, y_true_auc1, y_prob_auc1, np.array(features_list)

def diag_validation_batch(test_dataloader,  ra_test_dataloader,ra_test_video_dataloader, model,log_fd,params):
    print('-------------Waiting for Test...-------------')
    name_list, y_true_auc1, y_prob_auc1, features_list = get_fv_prob_auc(test_dataloader, model, log_fd, params)
    df_train_out = pd.DataFrame()
    df_train_out['图片'] = name_list
    df_train_out['gt'] = y_true_auc1
    df_train_out['ai prob'] = y_prob_auc1
    for i in range(features_list.shape[1]):
        df_train_out['ai fv ' + str(i)] = features_list[:, i]
    save_path = 'output/{}/'.format('internal_test_batch')
    os.makedirs(save_path, exist_ok=True)
    df_train_out.to_csv(os.path.join(save_path,'Malignant-notMalignant.csv'))

    print('-------------Waiting for RA_Test...-------------')
    name_list, y_true_auc1, y_prob_auc1, features_list = get_fv_prob_auc(ra_test_dataloader, model, log_fd, params)
    df_train_out = pd.DataFrame()
    df_train_out['图片'] = name_list
    df_train_out['gt'] = y_true_auc1
    df_train_out['ai prob'] = y_prob_auc1
    for i in range(features_list.shape[1]):
        df_train_out['ai fv ' + str(i)] = features_list[:, i]
    save_path = 'output/{}/'.format('external_test_batch')
    os.makedirs(save_path, exist_ok=True)
    df_train_out.to_csv(os.path.join(save_path,'Malignant-notMalignant.csv'))

    print('-------------Waiting for RA_Test_VIDEO...-------------')
    # name_list, y_true_auc1, y_prob_auc1, features_list = get_fv_prob_auc(ra_test_nc_dataloader, model, log_fd, params)
    # df_train_out = pd.DataFrame()
    # df_train_out['图片'] = name_list
    # df_train_out['gt'] = y_true_auc1
    # df_train_out['ai prob'] = y_prob_auc1
    # for i in range(features_list.shape[1]):
    #     df_train_out['ai fv ' + str(i)] = features_list[:, i]
    # df_train_out.to_csv(r'C:\Users\Administrator\Desktop\dwl\231102Ovary\project\result\bm_mb_ex_test_noca.csv')

    return 0

#------------设定参数列表------------
def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-6, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=10, help='training batch size')
    parser.add_argument('--testbatchsize', type=int,
                        default=1, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=15, help='every n epochs decay learning rate')
    parser.add_argument('--train_save', type=str,
                        default='Transunet_chaosheng_xiaorong_Fusion_CLF_FINAL')
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--weights', type=str, default='checkpoints/swin_tiny_patch4_window7_224.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)

    parser.add_argument('--exp_name', type=str, default='24-2-26' + TYPE,
                        help='Tag of experiment')
    parser.add_argument('--log_dir', type=str, default='log', help='Logger directory')
    parser.add_argument('--ckpt_path', type=str, default=None, help='The path of pretrained model')
    parser.add_argument('--epochs_dir', type=str, default=None, help='The path of pretrained model')
    parser.add_argument('--category', type=str, default='all', help='Category of point clouds')
    parser.add_argument('--visual_interval', type=int, default='100', help='visual_interval')
    opt = parser.parse_args()
    return opt

TYPE = 'MB2'
num_class = 2
class_names = ['Borderline', 'Malignant']
if __name__ == '__main__':
    opt = parse_option()

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_dir, epochs_dir, log_fd, train_writer, val_writer = prepare_logger(opt)
    # model = create_model(num_classes=opt.num_classes, has_logits=False).to(device)
    model = OvaMTA_CLF(training=True).to(device)
    model_state = torch.load(r".\diagmodel\epoch16-95.5531661237785-0.9256709832918011-0.7748959561863705.pth")
    # model_state = torch.load(r".\diagmodel\epoch16-94.87093406593395-0.8697274003396451-0.7828489620615605.pth")
    model.load_state_dict(model_state)

    log(log_fd, 'Loading Data...')
    test_dataloader = get_loader(mode='test', batchsize=opt.testbatchsize, trainsize=opt.trainsize, shuffle=False)
    ra_test_dataloader = get_loader(mode='ra_test', batchsize=opt.testbatchsize, trainsize=opt.trainsize, shuffle=False)
    ra_test_video_dataloader = get_loader(mode='ra_test', batchsize=opt.testbatchsize, trainsize=opt.trainsize,shuffle=False)

    diag_validation_batch(test_dataloader, ra_test_dataloader, ra_test_video_dataloader, model, log_fd, opt)

    log_fd.close()


