import torch
import argparse
from datetime import datetime
from lib.OvaMTA_seg import TransRaUNet_CLF_xiaorong
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
from tqdm import tqdm

class SegDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self,
                 trainsize,
                 augmentations,
                 file_excel = r"C:\Users\Administrator\Desktop\Table-OvaSeg\240108-image-based.xlsx",
                 mode = "train"):
        self.augmentations = augmentations
        self.trainsize = trainsize
        self.mode = mode
        self.root = Path("C:/Users/Administrator/Desktop/Table-OvaSeg/")
        df = pd.read_excel(file_excel,sheet_name=mode)
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
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])

    def __getitem__(self, item):
        info = self.infos.iloc[item]
        image = Image.open(info["tumor"]).convert("RGB")
        if self.mode != 'ra_test':
            gt = Image.open(info["roi"]).convert("L")
        else:
            gt = Image.open(info["tumor"]).convert("RGB")
            # gt = Image.open(self.root / info["tumor"]).convert("RGB")
        label = info["BBM"]
        name = info['tumor'].split('\\')[-1]

        if info['ca125'] == -1:
            b=[0]
        else:
            b =[1]
        clinical = [info['Age'] / 100,info['ca125'],b[0],
                    info['Age'] / 100,info['ca125'],b[0],
                    info['Age'] / 100, info['ca125'],b[0],
                    info['Age'] / 100, info['ca125'],b[0]
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

def val_save(train_dataloader,val_dataloader, test_dataloader,  ra_test_dataloader,ra_test_nc_dataloader, model,log_fd,params):
    def get_fv_prob_auc(train_dataloader, model, log_fd, params):
        y_true_auc0, y_prob_auc0 = [], []
        y_true_auc1, y_prob_auc1 = [], []
        y_true_auc2, y_prob_auc2 = [], []
        features_list = []
        name_list = []
        for batch_idx, (images, masks, infos, labels, name) in enumerate(train_dataloader):
            model.eval()
            images, infos, labels = images.to(device), infos.to(device), labels.to(device)
            outputs5, outputs4, outputs3, outputs2, outputs1, features = model(images)

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
                y_prob_auc2.append(outputs1[i][2].item())

                if labels[i].item() == 0:
                    y_true_auc0.append(1)
                else:
                    y_true_auc0.append(0)
                if labels[i].item() == 1:
                    y_true_auc1.append(1)
                else:
                    y_true_auc1.append(0)
                if labels[i].item() == 2:
                    y_true_auc2.append(1)
                else:
                    y_true_auc2.append(0)
                # print(batch_idx, name[i], len(feature_vector), labels[i].item(), y_true_auc1[-1], outputs1[i][1].item())

        fpr, tpr, thresholds = roc_curve(y_true_auc0, y_prob_auc0)
        auc0 = sklearn.metrics.auc(fpr, tpr)
        log(log_fd, 'Ovary auc is: %.3f%%' % (100 * auc0))

        return name_list, y_true_auc0, y_prob_auc0, np.array(features_list)

    print('-------------Waiting for Training...-------------')
    train_img_list, train_y_true, train_y_prob_auc, train_features_list = get_fv_prob_auc(train_dataloader, model,
                                                                                          log_fd, params)
    df_train_out = pd.DataFrame()
    df_train_out['图片'] = train_img_list
    df_train_out['gt'] = train_y_true
    df_train_out['ai prob'] = train_y_prob_auc
    for i in range(train_features_list.shape[1]):
        df_train_out['ai fv ' + str(i)] = train_features_list[:, i]
    df_train_out.to_csv(r'C:\Users\Administrator\Desktop\dwl\231102Ovary\project\result\train.csv')

    print('-------------Waiting for Val...-------------')
    name_list, y_true_auc1, y_prob_auc1, features_list = get_fv_prob_auc(val_dataloader, model, log_fd, params)
    df_val_out = pd.DataFrame()
    df_val_out['图片'] = name_list
    df_val_out['gt'] = y_true_auc1
    df_val_out['ai prob'] = y_prob_auc1
    for i in range(features_list.shape[1]):
        df_val_out['ai fv ' + str(i)] = features_list[:, i]
    df_val_out.to_csv(r'C:\Users\Administrator\Desktop\dwl\231102Ovary\project\result\val.csv')

    return 0

def train(train_loader, model, optimizer, epoch,train_step,log_fd,device):
    model.train()
    size_rates = [0.75,1,1.25]
    # ---- multi-scale training ----
    loss_record1, loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts, infos, labels, name = pack
            # print(labels)
            images, gts, infos, labels = images.to(device), gts.to(device), infos.to(device), labels.to(device)
            # ---- rescale ----
            # ---- forward ----
            # lateral_map = model(images)
            # ---- loss function ----
            weight = torch.tensor([0.5,0.5])
            weight = weight.to(device)
            loss_function_1 = nn.CrossEntropyLoss()
            # weight = torch.tensor([0.6,1,0.6,0.5])
            # weight = weight.to(device)
            # loss_function_2 = GHMR()
            # loss_function_2 = GHMC()
            loss_function_2 = SmoothL1Loss()
            # ---- forward ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1,features = model(images)
            # print(lateral_map_1.shape, labels)
            # ---- loss function ----
            loss5 = structure_loss(lateral_map_5, gts)
            loss4 = structure_loss(lateral_map_4, gts)
            loss3 = structure_loss(lateral_map_3, gts)
            loss2 = structure_loss(lateral_map_2, gts)
            # loss1 = loss_function_1(lateral_map_1, labels)
            # print(loss_function_1(lateral_map_1, labels))
            # print(loss_function_2(lateral_map_1, labels,weight = weight))
            loss1 = loss_function_1(lateral_map_1, labels) + loss_function_2(lateral_map_1, labels, weight = weight)
            loss = loss2 + loss3 + loss4 + loss5 + loss1 # TODO: try different weights for loss

            # loss = loss_function(lateral_map, labels)
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record5.update(loss5.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record1.update(loss1.data, opt.batchsize)

        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            log(log_fd,
                "Training Epoch [{:03d}/{:03d}] - Iteration [{:03d}/{:03d}]: [lateral-1: {:.4f}, lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]"
                .format(epoch, opt.epoch, i + 1, len(train_loader), loss_record1.show(),
                         loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))
            train_writer.add_scalar('loss1', loss1.item(), train_step)
            train_writer.add_scalar('loss2', loss2.item(), train_step)
            train_writer.add_scalar('loss3', loss1.item(), train_step)
            train_writer.add_scalar('loss4', loss1.item(), train_step)
            train_writer.add_scalar('loss5', loss2.item(), train_step)
            train_writer.add_scalar('total', loss.item(), train_step)
            train_step += 1

            # print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
            #       '[lateral-1: {:.4f}, lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
            #       format(datetime.now(), epoch, opt.epoch, i, total_step,loss_record1.show(),
            #              loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))
    return train_step


TYPE = 'OvaMTA-Seg'
num_class = 3
class_names = ['Ovary','Benign', 'Malignant']
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
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

    parser.add_argument('--exp_name', type=str, default='24-1-14' + TYPE,
                        help='Tag of experiment')
    parser.add_argument('--log_dir', type=str, default='log', help='Logger directory')
    parser.add_argument('--ckpt_path', type=str, default=None, help='The path of pretrained model')
    parser.add_argument('--epochs_dir', type=str, default=None, help='The path of pretrained model')
    parser.add_argument('--category', type=str, default='all', help='Category of point clouds')
    opt = parser.parse_args()

    # ---- environment setting ----
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir, epochs_dir, log_fd, train_writer, val_writer = prepare_logger(opt)

    # ---- build models ----
    model = TransRaUNet_CLF_xiaorong(training=True).to(device)
    model_state = torch.load(r'.\diagmodel\BM\BM-95.5531661237785-0.9256709832918011-0.7748959561863705.pth')
    model.load_state_dict(model_state)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(pg, opt.lr, weight_decay=1e-4)

    log(log_fd, 'Loading Data...')
    val_dataloader = get_loader(mode='val', batchsize=opt.testbatchsize, trainsize = opt.trainsize, shuffle=True)
    test_dataloader = get_loader(mode='test', batchsize=opt.testbatchsize, trainsize=opt.trainsize, shuffle=False)
    ra_test_dataloader = get_loader(mode='ra_test', batchsize=opt.testbatchsize, trainsize=opt.trainsize, shuffle=False)
    ra_test_nc_dataloader = get_loader(mode='ra_test_noca125', batchsize=opt.testbatchsize, trainsize=opt.trainsize, shuffle=False)
    train_loader = get_loader(mode='train', batchsize=opt.testbatchsize, trainsize=opt.trainsize, shuffle=False)
    total_step = len(train_loader)
    log(log_fd, "Dataset loaded!")
    print("#"*20, "Start Training", "#"*20)
    # val_save(train_loader, val_dataloader, test_dataloader, ra_test_dataloader, ra_test_nc_dataloader, model, log_fd,
    #          opt)
    val_step=0
    train_step=0
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        # WarmupMultiStepLR(optimizer, [10, 30, 50, 70],warmup_iters=10)
        # train_step=train(train_loader, model, optimizer, epoch,train_step,log_fd,device)
        # scheduler.step()
        if (epoch + 1) % 1 == 0:
            val_save(train_loader,val_dataloader, test_dataloader, ra_test_dataloader,ra_test_nc_dataloader, model,log_fd,opt)
            val_step += 1

    log_fd.close()


