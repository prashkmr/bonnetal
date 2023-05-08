import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.utils.data.distributed
import pytorch_lightning as pl
import imp
import yaml
import time
from PIL import Image
import __init__ as booger
import collections
import copy
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import numba
import time
from common.logger import Logger
from common.avgmeter import *
from common.sync_batchnorm.batchnorm import convert_model
from common.warmupLR import *
from tasks.semantic.modules.segmentator import *
from tasks.semantic.modules.ioueval import *
from tasks.semantic.modules.SLAM_ERROR import *

# @numba.jit(nopython=True, parallel=True, cache=True)
# def new_cloud_bonnetal(points, labels, above_gnd: np.array):
#   # import time
#   # time.sleep(2)
#   lidar_data = points[:, :2]  # neglecting the z co-ordinate
#   #height_data = points[:, 2] #+ 1.732
#   points2 = np.zeros((points.shape[0],4), dtype=np.float32) - 1 
#   # lidar_data -= grid_sub
#   # lidar_data = lidar_data /voxel_size # multiplying by the resolution
#   # lidar_data = np.floor(lidar_data)
#   # lidar_data = lidar_data.astype(np.int32)
#   # above_gnd = np.array([44, 48, 50, 51, 70, 71, 80, 81])
#   N = lidar_data.shape[0] # Total number of points
 
#   for i in numba.prange(N):
#       # x = lidar_data[i,0]
#       # y = lidar_data[i,1]
#       # z = height_data[i]
#       # if (0 < x < elevation_map.shape[0]) and (0 < y < elevation_map.shape[1]):
#       #     if z > elevation_map[x,y] + threshold:
#       #         points2[i,:] = points[i,:]
#       # print(labels[i])
#     # if i >= labels.shape[0] or i>=points.shape[0]:
#     #   break
#     # try:
#     if labels[i] in above_gnd:
#       points2[i,:] = points[i,:]
#     # except Exception as e:
#     #   print(e)
#   points2 = points2[points2[:,0] != -1]
#   return points2



def softargmax(x):
    print(x.shape)
    beta = 1000000.0
    xx = beta  * x
    c,h,w=x.shape
    sm = torch.nn.functional.softmax(xx, dim=0)
    #print(sm.shape)
    indices=torch.tensor([i for i in range(c)]).cuda()
    indices = torch.stack([torch.stack([indices]*h,dim=1)]*w,dim=2)
    #print(indices.shape)
    y = torch.mul(indices, sm)
    #print(y.shape)
    result=torch.sum(y,dim=0).cuda()
    # import pdb;pdb.set_trace()
    return result





def new_cloud_bonnetal(points, labels, above_gnd):
    # import pdb;pdb.set_trace()
    #print(labels[:100])
    # above_gnd= above_gnd_red = torch.tensor([15.0,18.0],device="cuda",dtype=torch.float32) 
    mask = torch.zeros_like(labels).cuda().float()
    labels1 = torch.tensor(torch.abs(labels+0.1).int(), dtype= torch.float).cuda()
    # labels = torch.tensor(labels, dtype= torch.float)
    
    mask[torch.where(torch.isin(labels1, above_gnd))] = 1
    mul=labels*mask
    mul[mul>1.0]=1.0
    # import pdb;pdb.set_trace()

    result=torch.mul(points.T,mul)
    #points = points.T[mask.bool(),:]
    return result.T






class BonnetalSeg(pl.LightningModule):
    
    def __init__(self, ARCH, DATA, datadir, logdir, my_callback ,path=None):
        super().__init__()        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.log = logdir
        self.path = path
        self.my_callback=my_callback
        self.training_step_outputs = []
        self.validation_step_outputs=[]

    # put logger where it belongs
        self.tb_logger = Logger(self.log + "/tb")
        self.info = {"train_update": 0,
                 "train_loss": 0,
                 "train_acc": 0,
                 "train_iou": 0,
                 "valid_loss": 0,
                 "valid_acc": 0,
                 "valid_iou": 0,
                 "backbone_lr": 0,
                 "decoder_lr": 0,
                 "head_lr": 0,
                 "post_lr": 0}

    # get the data
        parserPath = os.path.join(booger.TRAIN_PATH, "tasks", "semantic",  "dataset", self.DATA["name"], "parser.py")
        parserModule = imp.load_source("parserModule", parserPath)
        self.parser = parserModule.Parser(root=self.datadir,
                                      train_sequences=self.DATA["split"]["train"],
                                      valid_sequences=self.DATA["split"]["valid"],
                                      test_sequences=None,
                                      labels=self.DATA["labels"],
                                      color_map=self.DATA["color_map"],
                                      learning_map=self.DATA["learning_map"],
                                      learning_map_inv=self.DATA["learning_map_inv"],
                                      sensor=self.ARCH["dataset"]["sensor"],
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      batch_size=self.ARCH["train"]["batch_size"],
                                      workers=self.ARCH["train"]["workers"],
                                      gt=True,
                                      shuffle_train=False)

    # weights for loss (and bias)
    # weights for loss (and bias)
        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        for cl, freq in DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)  # map actual class to xentropy class
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)   # get weights
        for x_cl, w in enumerate(self.loss_w):  # ignore the ones necessary to ignore
            if DATA["learning_ignore"][x_cl]:
        # don't weigh
                self.loss_w[x_cl] = 0
        print("Loss weights from content: ", self.loss_w.data)

    # concatenate the encoder and the head
        with torch.no_grad():
            self.model = Segmentator(self.ARCH,
                               self.parser.get_n_classes(),
                               self.path)
        self.model_single = self.model
            
        if "loss" in self.ARCH["train"].keys() and self.ARCH["train"]["loss"] == "xentropy":
            self.criterion = nn.NLLLoss(weight=self.loss_w)
        else:
            raise Exception('Loss not defined in config file')
    # loss as dataparallel too (more images in batch)
       # if self.n_gpus > 1:
       #     self.criterion = nn.DataParallel(self.criterion) # spread in gpus
    
    
    def forward(self,a,b):
        return self.model(a,b)

    @staticmethod
    def get_mpl_colormap(cmap_name):
        cmap = plt.get_cmap(cmap_name)
    # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)
    # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
        return color_range.reshape(256, 1, 3)

    @staticmethod
    def make_log_img(depth, mask, pred, gt, color_fn):
    # input should be [depth, pred, gt]
    # make range image (normalized to 0,1 for saving)
        depth = (cv2.normalize(depth, None, alpha=0, beta=1,
                           norm_type=cv2.NORM_MINMAX,
                           dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
        out_img = cv2.applyColorMap(
         depth, Trainer.get_mpl_colormap('viridis')) * mask[..., None]
    # make label prediction
        pred_color = color_fn((pred * mask).astype(np.int32))
        out_img = np.concatenate([out_img, pred_color], axis=0)
    # make label gt
        gt_color = color_fn(gt)
        out_img = np.concatenate([out_img, gt_color], axis=0)
        return (out_img).astype(np.uint8)

    @staticmethod
    def save_to_log(logdir, logger, info, epoch, w_summary=False, model=None, img_summary=False, imgs=[]):
    # save scalars
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch)

    # save summaries of weights and biases
        if w_summary and model:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                if value.grad is not None:
                    logger.histo_summary(
                     tag + '/grad', value.grad.data.cpu().numpy(), epoch)

        if img_summary and len(imgs) > 0:
            directory = os.path.join(logdir, "predictions")
            if not os.path.isdir(directory):
                os.makedirs(directory)
            for i, img in enumerate(imgs):
                name = os.path.join(directory, str(i) + ".png")
                cv2.imwrite(name, img)


    def training_step(self, batch, batch_idx):
        in_vol, proj_mask, proj_labels, unproj_labels, path_seq, path_name, p_x, p_y, _, _, _, unproj_xyz, _, unproj_remissions, npoints=batch
        proj_labels = proj_labels.cuda(non_blocking=True).long()
        output = self(in_vol, proj_mask)
        #-------------------------------------------------------------------------------
        # unproj_argmax = torch.stack([output[k].argmax(dim=0)[p_y[k],p_x[k]] for k in range(len(output))])
        unproj_argmax = torch.stack([ softargmax(output[k]) [p_y[k],p_x[k]] for k in range(len(output))])
        # pred_np = unproj_argmax.cpu().numpy()
        pred_np = unproj_argmax.cuda()
        # pred_np = pred_np.reshape((len(output),-1)).astype(np.int32)
        pred_np = pred_np.reshape((len(output),-1)).float().cuda()
        # pred_np.requires_grad = True
        above_gnd_red = torch.tensor([10,11,13,14,15,16,17,18,19], dtype=torch.float).cuda() 
        points = torch.cat([unproj_xyz[:, :, :], unproj_remissions[:, :].unsqueeze(2)], dim=2).cuda()
        #-------------------------------------------------------------------------------
        loss = self.criterion(torch.log(output.clamp(min=1e-8)), proj_labels)
        # slam_err=0
        
        if self.current_epoch%1==0  and self.current_epoch >= 0:
            # trues = [torch.Tensor(new_cloud_bonnetal(points[k, :npoints[k]].copy(), unproj_labels[k,:npoints[k]].cpu().numpy().copy(), above_gnd_red)[:,:3]) for k in range(len(points))]
            # preds = [torch.Tensor(new_cloud_bonnetal(points[k, :npoints[k]].copy(), pred_np[k, :npoints[k]].copy(), above_gnd_red)[:,:3]) for k in range(len(points))]
            # import pdb;pdb.set_trace()
            trues = [(new_cloud_bonnetal(points[k, :npoints[k]], unproj_labels[k,:npoints[k]], above_gnd_red)[:,:3]) for k in range(len(points))]
            preds = [(new_cloud_bonnetal(points[k, :npoints[k]], pred_np[k, :npoints[k]], above_gnd_red)[:,:3]) for k in range(len(points))]
            # import pdb;pdb.set_trace()
            slam_err=Slam_error(trues, preds)            # slam_err
            print("SLAM_ERROR_SUCCESSFUL")

        # 
            loss +=  (slam_err*10)
        
        loss1 = loss.mean()
        print(loss1, "************************")
        with torch.no_grad():
            self.my_callback.evaluator.reset()
            argmax = output.argmax(dim=1)
            self.my_callback.evaluator.addBatch(argmax, proj_labels)
            accuracy = self.my_callback.evaluator.getacc()
            jaccard, class_jaccard = self.my_callback.evaluator.getIoU()
        self.my_callback.losses.update(loss1.item(), in_vol.size(0))
        self.my_callback.acc.update(accuracy.item(), in_vol.size(0))
        self.my_callback.iou.update(jaccard.item(), in_vol.size(0))
        self.my_callback.batch_time.update(time.time() - self.my_callback.end)
        end = time.time()
        
        
        update_ratios = []
        for g in self.optimizer.param_groups:
            lr = g["lr"]
            for value in g["params"]:
                if value.grad is not None:
                    w = np.linalg.norm(value.data.cpu().numpy().reshape((-1)))
                    update = np.linalg.norm(-max(lr, 1e-10) *
                                    value.grad.cpu().numpy().reshape((-1)))
                    update_ratios.append(update / max(w, 1e-10))
        print(update_ratios,"******************************************************************")
        update_ratios = np.array(update_ratios)
        update_mean = update_ratios.mean()
        update_std = update_ratios.std()
        self.my_callback.update_ratio_meter.update(update_mean)
        

        #this has to be after epoch
        if batch_idx % self.ARCH["train"]["report_batch"] == 0:
            
            print('Lr: {lr:.3e} | '
             'Update: {umean:.3e} mean,{ustd:.3e} std | '
             'Epoch: [{0}][{1}/{2}] | '
             'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
             'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
             'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
             'acc {acc.val:.3f} ({acc.avg:.3f}) | '
             'IoU {iou.val:.3f} ({iou.avg:.3f})'.format(
                 self.current_epoch, batch_idx, -999, batch_time=self.my_callback.batch_time,
                 data_time=self.my_callback.data_time, loss=self.my_callback.losses, acc=self.my_callback.acc, iou=self.my_callback.iou, lr=lr,
                 umean=update_mean, ustd=update_std))

        


        self.training_step_outputs.append(self.my_callback.acc.avg)
        self.training_step_outputs.append(self.my_callback.iou.avg)
        self.training_step_outputs.append(self.my_callback.losses.avg)
        self.training_step_outputs.append(self.my_callback.update_ratio_meter.avg)
        
        
        
            
        return loss

    def validation_step(self,val_batch,batch_idx):
        # self.parser.get_xentropy_class_string
        in_vol, proj_mask, proj_labels, unproj_labels, path_seq, path_name, p_x, p_y, _, _, _, unproj_xyz, _, unproj_remissions, npoints = val_batch
        proj_labels = proj_labels.cuda(non_blocking=True).long()
        output=self(in_vol,proj_mask)
        loss=self.criterion(torch.log(output.clamp(min=1e-8)), proj_labels)
        argmax = output.argmax(dim=1)
        self.my_callback.evaluator.addBatch(argmax, proj_labels)
        self.my_callback.val_losses.update(loss.mean().item(), in_vol.size(0))
        self.my_callback.val_batch_time.update(time.time() - self.my_callback.val_end)
        self.my_callback.val_end = time.time()
        accuracy = self.my_callback.evaluator.getacc()
        jaccard, class_jaccard = self.my_callback.evaluator.getIoU()
        self.my_callback.val_acc.update(accuracy.item(), in_vol.size(0))
        self.my_callback.val_iou.update(jaccard.item(), in_vol.size(0))
        print('Validation set:\n'
            'Time avg per batch {batch_time.avg:.3f}\n'
            'Loss avg {loss.avg:.4f}\n'
            'Acc avg {acc.avg:.3f}\n'
            'IoU avg {iou.avg:.3f}'.format(batch_time=self.my_callback.val_batch_time,
                                           loss=self.my_callback.val_losses,
                                           acc=self.my_callback.val_acc,
                                           iou=self.my_callback.val_iou))
        for i, jacc in enumerate(class_jaccard):
            print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(i=i, class_str = self.parser.get_xentropy_class_string(i), jacc=jacc))
        self.validation_step_outputs.clear()
        self.validation_step_outputs.append(self.my_callback.val_acc.avg)
        self.validation_step_outputs.append(self.my_callback.val_iou.avg)
        self.validation_step_outputs.append(self.my_callback.val_losses.avg)
    
    def configure_optimizers(self):
        if self.ARCH["post"]["CRF"]["use"] and self.ARCH["post"]["CRF"]["train"]:
            self.lr_group_names = ["post_lr"]
            self.train_dicts = [{'params': self.model_single.CRF.parameters()}]
        else:
            self.lr_group_names = []
            self.train_dicts = []
        if self.ARCH["backbone"]["train"]:
            self.lr_group_names.append("backbone_lr")
            self.train_dicts.append(
              {'params': self.model_single.backbone.parameters()})
        if self.ARCH["decoder"]["train"]:
            self.lr_group_names.append("decoder_lr")
            self.train_dicts.append(
          {'params': self.model_single.decoder.parameters()})
        if self.ARCH["head"]["train"]:
            self.lr_group_names.append("head_lr")
            self.train_dicts.append({'params': self.model_single.head.parameters()})

        # Use SGD optimizer to train
        self.optimizer = optim.SGD(self.train_dicts,
                               lr=self.ARCH["train"]["lr"],
                               momentum=self.ARCH["train"]["momentum"],
                               weight_decay=self.ARCH["train"]["w_decay"])
        steps_per_epoch = self.parser.get_train_size()
        up_steps = int(self.ARCH["train"]["wup_epochs"] * steps_per_epoch)
        final_decay = self.ARCH["train"]["lr_decay"] ** (1/steps_per_epoch)
        self.scheduler = warmupLR(optimizer=self.optimizer,
                              lr=self.ARCH["train"]["lr"],
                              warmup_steps=up_steps,
                              momentum=self.ARCH["train"]["momentum"],
                              decay=final_decay)
        
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}



        
        
        





