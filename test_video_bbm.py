import cv2
import matplotlib.pyplot as plt
import torch
import glob
import argparse
from lib.OvaMTA_seg import TransRaUNet_CLF_xiaorong as OvaMTA_OVASEG
from lib.OvaMTA_diag import TransRaUNet_CLF_xiaorong as OvaMTA_CLF
from datetime import datetime
from lib.swim_transformer import swin_tiny_patch4_window7_224 as create_model
# from lib.PraNet_Res2Net_PLAX import TransRaUNet_CLF_xiaorong
# from utils.dataloader import get_loader,test_dataset,SegDataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter, WarmupMultiStepLR, _thresh
import torch.optim.lr_scheduler as lr_scheduler
from lib.PraNet_Res2Net_clinical_1 import TransRaUNet_CLF_INFO_FUSION_xiaorong

from utils.focal_loss import Focal_loss, FocalLoss
from utils.ghm_loss import GHMC,GHMR
from scipy.interpolate import splprep,splev
from utils.smooth_l1_loss import SmoothL1Loss
from utils.utils import *
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
def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Ovary_threshold', type=float,
                        default=0.1, help='Ovary p0 threshold')
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

    parser.add_argument('--exp_name', type=str, default='24-2-26',
                        help='Tag of experiment')
    parser.add_argument('--log_dir', type=str, default='log', help='Logger directory')
    parser.add_argument('--ckpt_path', type=str, default=None, help='The path of pretrained model')
    parser.add_argument('--epochs_dir', type=str, default=None, help='The path of pretrained model')
    parser.add_argument('--category', type=str, default='all', help='Category of point clouds')
    parser.add_argument('--visual_interval', type=int, default='100', help='visual_interval')
    opt = parser.parse_args()
    return opt
def get_mask(lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, framelist,params):
    lateral_map_5 = F.upsample(lateral_map_5, size=(params.trainsize, params.trainsize), mode='bilinear',
                               align_corners=False)
    lateral_map_4 = F.upsample(lateral_map_4, size=(params.trainsize, params.trainsize), mode='bilinear',
                               align_corners=False)
    lateral_map_3 = F.upsample(lateral_map_3, size=(params.trainsize, params.trainsize), mode='bilinear',
                               align_corners=False)
    lateral_map_2 = F.upsample(lateral_map_2, size=(params.trainsize, params.trainsize), mode='bilinear',
                               align_corners=False)

    res = F.upsample(lateral_map_5 + lateral_map_4 + lateral_map_3 + lateral_map_2,
                     size=(params.trainsize, params.trainsize), mode='bilinear', align_corners=False)

    framelist[1:, :, :, :] = framelist[:-1, :, :, :]

    # print('framelist:',framelist.shape,framelist[:-1, :, :, :].shape,framelist[1:, :, :, :].shape)
    framelist[0, :, :, :] = res[0].cpu().detach().numpy()
    res = (framelist[0, :, :, :] * 4 + framelist[1, :, :, :] + framelist[2, :, :, :] + framelist[3, :, :,
                                                                                       :] + framelist[4, :, :,
                                                                                            :] + framelist[5, :, :,
                                                                                                 :]) / 9
    # print(i_f,framelist[0,:,:,:].sum(),framelist[1,:,:,:].sum(),framelist[2,:,:,:].sum(),framelist[3,:,:,:].sum(),framelist[4,:,:,:].sum(),framelist[5,:,:,:].sum())
    # thre_use = 0.5 * res.min().cpu().detach().numpy() + 0.5 * res.max().cpu().detach().numpy()

    res = np.array(res[0, :, :] > 0, dtype=np.uint8)
    return res
def patch_mask2mask(patch_mask, images, x1, x2, y1, y2):
    mask = np.zeros((images.shape[0], images.shape[1]))
    mask[x1:x2, y1:y2] = cv2.resize(patch_mask, ( y2 - y1,x2 - x1), interpolation=cv2.INTER_NEAREST)
    return mask

def image2masklist(frame, model, model_BM,model_MB,params,img_transform,framelist,video_path,frame_patch_list,i_f):
    print('载入图片,标准化视频帧')
    images = img_transform(Image.fromarray(frame))
    images = images.reshape((1, 3, params.trainsize, params.trainsize))
    images = images.to(device)
    # print('image:',images.shape)
    images = F.upsample(images, size=(params.trainsize, params.trainsize), mode='bilinear', align_corners=True)
    print('模型粗分割')
    lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, outputs1, fvSeg = model(images)

    # --------粗分类结果---------
    p0 = outputs1[0][0].item()
    p1 = outputs1[0][1].item()
    p2 = outputs1[0][2].item()
    print('粗分类结果',"卵巢概率",p0,"良性概率",p1,'恶性概率',p2)

    # ------分割结果-------
    # ------粗分割结果-------
    res = get_mask(lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, framelist,params)
    contours, hierarchy = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours)==0:
        return ([], [], [], [], [],
         [], [], [])

    p0_patch_list, p1_patch_list, p2_patch_list,p3_patch_list =[], [], [], []
    fvBM_list, fvMB_list, fvSeg_list=[],[],[]
    refine_res = []
    rect_out = (0, 0, 0, 0)
    max_contour = max(contours, key=cv2.contourArea)
    print('最大围线：',max_contour.shape)

    for j in range(1):
        # -------对每一个contours进行裁剪后决定-------
        (x, y, w, h) = cv2.boundingRect(max_contour)
        rect_out = (x, y, w, h)
        # 找高清原图进行图片patch截取
        img = np.array(frame, dtype=np.uint8)

        if rect_out[0] != 0:
            y1 = int((max(0, rect_out[0] - 10) / 352) * img.shape[1])
            y2 = int((min(352, rect_out[0] + rect_out[2] + 10) / 352) * img.shape[1])
            x1 = int((max(0, rect_out[1] - 10) / 352) * img.shape[0])
            x2 = int((min(352, rect_out[1] + rect_out[3] + 10) / 352) * img.shape[0])
            # print('img shape:', img.shape)
            img_patch = img[x1:x2, y1:y2, :]
            # -----------patch截取后归一化并存储----------
            img_patch = cv2.resize(img_patch, (512, 512), interpolation=cv2.INTER_NEAREST)
            out_dir = os.path.join(r'D:\Research\231118卵巢\240124卵巢分割\app_test\cut',
                                    os.path.basename(video_path).split('_')[3])
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir ,
                                    os.path.basename(video_path).split('.')[0] + '_f' + str(i_f).zfill(4) + '_0' + str(
                                        j) + 'cut.png')

            cv2.imencode('.png', img_patch)[1].tofile(out_path)
        # ---------现在存好了当前图像的所有patch---------
        # ---------对每一个patch进行进一步的判断-----------
        # ---------如果不为卵巢---------
        if p0 < params.Ovary_threshold:
            if rect_out[0] != 0:
                print('粗分类为结节')
                # ------------使用patch进行良恶性预测------------
                # ------------读入patch并进行图像归一化------------
                img_patch = Image.open(out_path).convert("RGB")
                img_patch = img_transform(img_patch)
                img_patch = img_patch.reshape((1, 3, params.trainsize, params.trainsize))
                img_patch = img_patch.to(device)
                img_patch = F.upsample(img_patch, size=(params.trainsize, params.trainsize), mode='bilinear',
                                       align_corners=True)
                # ------------用诊断模型进行patch的测试------------
                lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, outputs1_BM, fvBM = model_BM(
                    img_patch)
                lateral_map_5_, lateral_map_4_, lateral_map_3_, lateral_map_2_, outputs1_MB, fvMB = model_MB(
                    img_patch)
                print('fvBM的形状：', fvBM.shape,'fvMB的形状：', fvBM.shape,
                      'fvSeg的形状：', fvSeg.shape)
                patch_mask = get_mask(lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, frame_patch_list,params)

                refine_res.append(patch_mask2mask(patch_mask,img, x1, x2, y1, y2))

                p_Benign, p_Malignant = outputs1_BM[0][0].item(), outputs1_BM[0][1].item()
                p_Borderline = outputs1_MB[0][0].item()
                p1=p_Benign
                p2=p_Malignant*p_Borderline
                p3=p_Malignant*(1-p_Borderline)
                # print('看看概率和：', p1, p2, p1 + p2)
                print('精细分类结果', "卵巢概率", p0, "良性概率", p1, '交界性概率', p2,'恶性概率',p3)
                if p1 > 0.65:
                    pred = 1
                elif p2>0.1:
                    pred = 2
                else:
                    pred=3
            else:
                _, pred = torch.max(outputs1.data, dim=1)
        # ---------如果判断为卵巢---------
        elif p0 >= params.Ovary_threshold:
            print('粗分类为卵巢,不继续进行推断')
            # ------------使用patch进行预测------------
            if rect_out[0] != 0:
                img_patch = Image.open(out_path).convert("RGB")
                img_patch = img_transform(img_patch)
                img_patch = img_patch.reshape((1, 3, params.trainsize, params.trainsize))
                img_patch = img_patch.to(device)
                img_patch = F.upsample(img_patch, size=(params.trainsize, params.trainsize), mode='bilinear',
                                       align_corners=True)

                lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, outputs1_ON, features = model(img_patch)
                patch_mask = get_mask(lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, frame_patch_list,params)
                refine_res.append(patch_mask2mask(patch_mask,img, x1, x2, y1, y2))

                p0, p1, p3 = outputs1_ON[0][0].item(), outputs1_ON[0][1].item(), outputs1_ON[0][2].item()

                p2=0
                lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, outputs1_BM, fvBM = model_BM(
                    img_patch)
                lateral_map_5_, lateral_map_4_, lateral_map_3_, lateral_map_2_, outputs1_MB, fvMB = model_MB(
                    img_patch)
                # print('看看概率和：', p1, p2, p1 + p2)
            else:
                _, pred = torch.max(outputs1.data, dim=1)
        p0_patch_list.append(p0)
        p1_patch_list.append(p1)
        p2_patch_list.append(p2)
        p3_patch_list.append(p3)
        fvSeg_list.append(fvSeg.cpu().detach().numpy())
        fvBM_list.append(fvBM.cpu().detach().numpy())
        fvMB_list.append(fvMB.cpu().detach().numpy())
    p0_patch_list, p1_patch_list, p2_patch_list, p3_patch_list=(np.array(p0_patch_list),
                                                                np.array(p1_patch_list),
                                                                np.array(p2_patch_list),
                                                                np.array(p3_patch_list))
    fvBM_list, fvMB_list, fvSeg_list=(np.array(fvBM_list),
                                      np.array(fvMB_list),
                                      np.array(fvSeg_list))
    return (refine_res, p0_patch_list, p1_patch_list, p2_patch_list,p3_patch_list,
            fvBM_list,fvMB_list,fvSeg_list)

def compute_acc_dice_iou(res,gt):
    gt = gt.squeeze()
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)

    input = res
    target = np.array(gt)
    # input = _thresh(input)
    target = _thresh(target)
    smooth = 1
    input_flat = np.reshape(input, (-1))
    target_flat = np.reshape(target, (-1))

    intersection = (input_flat * target_flat)
    unin_sum = np.array((target + input) > 0, dtype=int)
    iou = intersection.sum() / unin_sum.sum()
    acc = (input_flat==target_flat).sum() / len(input_flat)
    dice = 2 * intersection.sum() / (input.sum() + target.sum())
    return iou,dice,acc

def compute_hd(smoothed_contour_pred,contours_gt):
    from scipy.spatial.distance import directed_hausdorff
    if len(smoothed_contour_pred)==0:
        return float('inf')
    else:
        return max(directed_hausdorff(smoothed_contour_pred,contours_gt)[0],directed_hausdorff(contours_gt,smoothed_contour_pred)[0])
def pred_video(model,model_BM,model_MB,device,params,video_path):
    Ovary_threshold=0.1
    Benign_threshold=0.65
    Borderline_threshold=0.15
    print('模型变为测试模式')
    model.eval()
    model_BM.eval()
    model_MB.eval()
    img_transform = transforms.Compose([
        transforms.Resize((params.trainsize, params.trainsize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    # ------找videopath对应的mask------
    # anonationdir = os.path.join(r"C:\Users\install-44\卵巢动态数据集文件1.1.0.1_to代\有病灶", 'MarkDB',
    #                             video_path.split('\\')[6], video_path.split('\\')[7])
    # print(anonationdir)
    # annotations = glob.glob(anonationdir + '\\*\\*.png')
    # print('标注帧数：',len(annotations))
    # annotatedframeid=[int(annotation.split('\\')[-1].split('_')[-2]) for annotation in annotations]

    cap = cv2.VideoCapture(video_path)
    i_f = 0
    # t_f = np.arange(0, 20, 1)*np.ones((7,1))
    component_mass_list=np.zeros((3,500))
    chafenlist =np.zeros(6)
    framelist = np.zeros((6,3,352,352))
    frame_patch_list=np.zeros((6,3,352,352))
    predList,p0List,p1List,p2List,p3List,annotation_typeList=[],[],[],[],[],[]
    fvBM_list,fvMB_list,fvSeg_list=[],[],[]
    Iou=[]
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print('开始读帧',i_f)
            siou, sdice, sacc = -1, -1, -1
            print('将图片进行mask预测image2masklist')
            # if i_f<28:
            #     i_f+=1
            #     print('>>>>>>>帧数小')
            #     continue
            (refine_res, p0_patch_list, p1_patch_list, p2_patch_list,p3_patch_list,
             fvBM,fvMB,fvSeg)=image2masklist(frame, model, model_BM,model_MB,params,img_transform,framelist,video_path,frame_patch_list,i_f)
            if refine_res==[]:
                print('>>>>>>>>没割出来东西')
                continue

            print('res.shape', np.array(refine_res).shape)
            res=np.zeros((frame.shape[0],frame.shape[1]))
            for i_rf in range(len(refine_res)):
                res+=refine_res[i_rf]

            images = img_transform(Image.fromarray(frame))
            images = images.reshape((1, 3, params.trainsize, params.trainsize))
            images = images.to(device)
            # print('image:',images.shape)
            images = F.upsample(images, size=(params.trainsize, params.trainsize), mode='bilinear', align_corners=True)
            # ------用于画contour------
            canvas = cv2.normalize(images[0, 0, :, :].cpu().detach().numpy(), None, 0, 1, cv2.NORM_MINMAX)
            # ------用于画框------
            canvas2 = cv2.normalize(images[0, 0, :, :].cpu().detach().numpy(), None, 0, 1, cv2.NORM_MINMAX)

            p0=np.array(p0_patch_list).mean()
            p1 = np.array(p1_patch_list).mean()
            p2 = np.array(p2_patch_list).mean()
            p3 = np.array(p3_patch_list).mean()
            if p0>Ovary_threshold:
                pred=0
            elif p1>Benign_threshold:
                pred=1
            elif p2>Borderline_threshold:
                pred=2
            else:
                pred=3

            print('>>>>>>>>>>>>>>>>预测为',pred,'类别')

            predList.append(pred)
            p0List.append(p0)
            p1List.append(p1)
            p2List.append(p2)
            p3List.append(p3)
            fvBM_list.append(fvBM)
            fvMB_list.append(fvMB)
            fvSeg_list.append(fvSeg)

            subimg1 = cv2.normalize(images[0, 0, :, :].cpu().detach().numpy(), None, 0, 255, cv2.NORM_MINMAX)
            subimg1 = cv2.cvtColor(subimg1.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            res = cv2.resize(res, (352, 352), interpolation=cv2.INTER_NEAREST)
            contours_res, hierarchy = cv2.findContours(np.array(res, dtype=np.uint8), cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_NONE)
            canvas2 = cv2.drawContours(canvas2, contours_res, -1, (1, 1, 1), 3)
            subimg2 = cv2.normalize(canvas2, None, 0, 255, cv2.NORM_MINMAX)
            subimg2=cv2.cvtColor(subimg2.astype(np.uint8), cv2.COLOR_GRAY2RGB)

            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
            for j in range(len(contours_res)):
                (x, y, w, h) = cv2.boundingRect(contours_res[j])

                if pred==0:
                    cv2.putText(canvas, 'Ovary %.3f'%(p0), (x-10, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (1, 1, 1), 2)
                    canvas = cv2.rectangle(canvas, pt1=(x - 10, y - 10), pt2=(x + w + 10, y + h + 10), color=(1, 1, 1),
                                           thickness=2)

                elif pred==1:
                    cv2.putText(canvas, 'Benign %.3f'%(p1), (x-10, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 1, 0), 2)
                    canvas = cv2.rectangle(canvas, pt1=(x - 10, y - 10), pt2=(x + w + 10, y + h + 10), color=(0, 1, 0),
                                           thickness=2)
                elif pred==2:
                    # print('注意,是交界性!!!,为什么不画框?')
                    cv2.putText(canvas, 'Borderline %.3f'%(p2), (x-10, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 1, 1), 2)
                    canvas = cv2.rectangle(canvas, pt1=(x - 10, y - 10), pt2=(x + w + 10, y + h + 10), color=(0, 1, 1),
                                           thickness=2)
                    # plt.figure()
                    # plt.imshow(canvas)
                    # plt.show()
                elif pred==3:
                    cv2.putText(canvas, 'Malignant %.3f'%(p3), (x-10, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 1), 2)
                    canvas = cv2.rectangle(canvas, pt1=(x - 10, y - 10), pt2=(x + w + 10, y + h + 10), color=(0, 0, 1),
                                           thickness=2)
            subimg3 = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX)

            subimg4 = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
            subimg4 = cv2.cvtColor(subimg4.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            cv2.putText(subimg4, 'pred', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)

            subimg5 = cv2.normalize(framelist[0, 0, :, :], None, 0, 255, cv2.NORM_MINMAX)
            # print(framelist[0, 0, :, :].shape)
            subimg5 = cv2.cvtColor(subimg5.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            subimg6 = cv2.normalize(framelist[1, 0, :, :], None, 0, 255, cv2.NORM_MINMAX)
            subimg6 = cv2.cvtColor(subimg6.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            heatmap = cv2.applyColorMap(subimg6, cv2.COLORMAP_JET)

            subimg6=0.5*heatmap+0.5*subimg1
            print(subimg4.shape, subimg5.shape, subimg6.shape)
            subimgs1 = np.concatenate([subimg1, subimg2, subimg3], 1)
            subimgs2 = np.concatenate([subimg4, subimg5, subimg6], 1)
            print(subimgs1.shape,subimgs2.shape)
            IMG = np.concatenate([subimgs1, subimgs2], 0)
            # IMG=subimgs1
            cv2.imencode('.png', IMG)[1].tofile(
                os.path.join(r"E:\231118-卵巢-总\240124-卵巢分割分类pipeline\数据相关\AI处理后的视频帧-BBM",
                             os.path.basename(video_path).split('.')[
                                 0] + '_f' + str(i_f).zfill(4) + '_00.png'))
            i_f += 1
        else:
            break
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>',os.path.basename(video_path),
              '帧',i_f,'预测为:',pred, '分割iou:',siou, '分割dice:',sdice)#, sacc)

    cap.release()
    # ------存储videopath对应的几个fv于npz文件------
    fvSeg_list=np.array(fvSeg_list)
    fvBM_list=np.array(fvBM_list)
    fvMB_list=np.array(fvMB_list)
    p0List,p1List,p2List,p3List = np.array(p0List),np.array(p1List),np.array(p2List),np.array(p3List)
    print('>>>',p0List.shape,'特征向量形状:',fvSeg_list.shape,fvBM_list.shape,fvMB_list.shape)
    np.savez(os.path.join(os.path.dirname(video_path),video_path.split('\\')[-1].split('.')[-2]+'.npz'),
             predList=predList,p0List=p0List,p1List=p1List,p2List=p2List,p3List=p3List,
             fvSegList=fvSeg_list,fvBMList=fvBM_list,fvMBList=fvMB_list)

    out_video_path=os.path.join(r'E:\231118-卵巢-总\240124-卵巢分割分类pipeline\数据相关\AI处理后的视频-BBM',os.path.basename(video_path).split('.')[0]+'.avi')
    # framepath = os.path.join(r'E:\231118-卵巢-总\240124-卵巢分割分类pipeline\数据相关\AI处理后的视频-BBM',
    #                          os.path.basename(video_path).split('.')[0] + '_f' + str(0).zfill(4) + '_0.png')
    # img = cv2.imdecode(np.fromfile(framepath, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    # print(img.shape)
    video=cv2.VideoWriter(out_video_path,cv2.VideoWriter_fourcc('M','J','P','G'),10, (1500,1000))
    for i_f_in in tqdm(range(i_f)):
        framepath=os.path.join(r"E:\231118-卵巢-总\240124-卵巢分割分类pipeline\数据相关\AI处理后的视频帧-BBM",
                               os.path.basename(video_path).split('.')[0] + '_f' + str(i_f_in).zfill(4) + '_00.png')
        img = cv2.imdecode(np.fromfile(framepath, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
        img=cv2.resize(img, (1500,1000),interpolation=cv2.INTER_NEAREST)
        video.write(img)
    video.release()

    return video_path,predList,p0List,p1List,p2List,p3List


def main(device):
    opt = parse_option()
    print(torch.cuda.is_available())

    model_path=r".\segmodel\epoch19-0.9703952223063982-88.81376262626276-82.25021750388382.pth"
    model = OvaMTA_OVASEG(training=True).to(device)
    model_state = torch.load(model_path)
    model.load_state_dict(model_state)

    model_BM = OvaMTA_CLF(training=True).to(device)
    model_state = torch.load(r".\diagmodel\BM\BM-95.5531661237785-0.9256709832918011-0.7748959561863705.pth")
    model_BM.load_state_dict(model_state)

    model_MB = OvaMTA_CLF(training=True).to(device)
    model_state = torch.load(r".\diagmodel\MB\MB-94.87093406593395-0.8697274003396451-0.7828489620615605.pth")
    model_MB.load_state_dict(model_state)

    Borderline_idL=['US-Ovary202006201','US-Ovary2021020611','US-Ovary202105201','US-Ovary202106034',
                    'US-Ovary202106178','US-Ovary202106181','US-Ovary2021061921','US-Ovary2021062018',
                    'US-Ovary202107045','US-Ovary2021071122','US-Ovary202107172','US-Ovary202107215',
                    'US-Ovary202107217','US-Ovary202108191','US-Ovary202110284','US-Ovary202202133',
                    'US-Ovary202205314','US-Ovary202206015']

    video_pathList=glob.glob(r"D:\Research\231118-卵巢\240124-卵巢分割分类pipeline\数据相关\原视频-用于判读\*\*.mp4")

    videoL=[]
    for videoP in video_pathList:
        if videoP.split('\\')[-2] not in Borderline_idL:
            videoL.append(videoP)
    video_pathList=videoL

    # df_=pd.read_csv(r"C:\Users\Administrator\Desktop\app_test\ORADS以及人机判读test_video.csv",encoding='ANSI')
    print(len(video_pathList),video_pathList)
    # video_path=r"C:\Users\install-44\卵巢动态数据集文件1.1_to代\无病灶\BaseDB\V_DATA001-GB01-UNK-20230510\OVAB_D_AH04_US-Ovary202106021_0001\OVAB_D_AH04_US-Ovary202106021_0001.mp4"
    videoframeList, videoidList, frameAnnotationList, framepredList=[],[],[],[]
    frameP0, frameP1, frameP2,frameP3 = [], [], [], []
    frameBMfvList,frameMBfvList,frameSegfvList=[],[],[]
    # smoothedtypeList = []
    # frameIoU = []
    for video_path in video_pathList[:]:
        video_path,predList,p0List,p1List,p2List,p3List=pred_video(model,model_BM,model_MB,device,opt,video_path)
        # videoframeList.append(os.path.basename(annotation))
        # videoframeList=range(len(p0List))
        # plt.figure()
        for frame_id in range(len(p0List)):
            videoframeList.append(frame_id)
            videoidList.append(os.path.basename(video_path))
            # frameAnnotationList.append(df_[df_['bah']==video_path.split('\\')[-2]]['bbm'].tolist()[0])
            framepredList.append(predList[frame_id])
            # smoothedtypeList.append(annotation_typeList[frame_id])
            frameP0.append(p0List[frame_id])
            frameP1.append(p1List[frame_id])
            frameP2.append(p2List[frame_id])
            frameP3.append(p3List[frame_id])
            # if frame_id%10==0:
                # plt.scatter(frame_id, df_[df_['bah']==video_path.split('\\')[-2]]['bbm'].tolist()[0], c='r')

            # frameIoU.append(Iou[frame_id])
        # plt.plot(range(len(predList)), predList, 'gray', label='pred')
        # plt.plot(range(len(predList)), p0List, 'go-', label='p0')
        # plt.plot(range(len(predList)), p1List, 'orange', label='p1')
        # plt.plot(range(len(predList)), p2List, 'b-.', label='p2')
        # plt.legend()
        # plt.ylim([-0.1, 2.1])
        # plt.title(video_path.split('\\')[-1].split('.')[0])
        # plt.savefig(os.path.join(r'C:\Users\Administrator\Desktop\app_test\generated-video-prob',
        #                          os.path.basename(video_path).split('.')[0] + '.png'))
        # plt.close()
    df = pd.DataFrame()
    df['videoid'] = videoidList
    df['videoframe'] = videoframeList
    # df['annotation'] = frameAnnotationList
    df['pred'] = framepredList
    df['p0'] = frameP0
    df['p1'] = frameP1
    df['p2'] = frameP2
    df['p3'] = frameP3
    # df['iou'] = frameIoU
    df.to_csv(r'E:\231118-卵巢-总\240124-卵巢分割分类pipeline\数据相关\AI处理后的视频-BBM\240610bbm_video_frame.csv', encoding='ANSI')

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(device)