import os
import cv2
import torch
import math
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import torch.cuda.amp as amp
from datetime import datetime
from skimage.transform import resize
from PIL import Image
import wandb
import torchvision.utils as vutils
def renormalize(x):
  """Renormalize from [-1, 1] to [0, 1]."""
  return x / 2. + 0.5


class BatchColorize(object):
    def __init__(self, n=40):
        self.cmap = color_map(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((size[0], 3, size[1], size[2]), dtype=np.float32)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[:,0][mask] = self.cmap[label][0]
            color_image[:,1][mask] = self.cmap[label][1]
            color_image[:,2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[:,0][mask] = color_image[:,1][mask] = color_image[:,2][mask] = 255

        return color_image

def color_map(N=256, normalized=True):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap




class Visualizer(object):
    def __init__(self, args, viz=None):

        self.exp_name = wandb.run.name
        self.args = args
        self.vis_interval = self.args.vis_interval
        self.colorize = BatchColorize(args.num_classes)


    def vis_images(self, setting, i_iter, name, images, mean):
        print("Entering visual mode")
        # image = renormalize(images)
        i_shape = images.shape
        mean_tensor = torch.tensor(mean).float().expand(i_shape[0], i_shape[3], i_shape[2], 3).transpose(1,3)
        imgs_viz = torch.clamp(images+mean_tensor, 0.0, 255.0)
        imgs_viz = vutils.make_grid(imgs_viz, normalize=False, scale_each=False)
        print(imgs_viz.shape)
        wandb.log({f'{setting}/{name}': wandb.Image(imgs_viz)}, step = i_iter)

    def vis_image_pred(self, setting, i_iter, name,images, pred, mean):
        print("Visualizing results")
        print("images.shape, pred.shape ", images.shape, pred.shape)
        i_shape = images.shape
        mean_tensor = torch.tensor(mean).float().expand(i_shape[0], i_shape[3], i_shape[2], 3).transpose(1,3)
        imgs_viz = torch.clamp(images+mean_tensor, 0.0, 255.0)
        pred = pred.detach().cpu().float().numpy()
        # print("np.unique(pred) :", np.unique(pred))
        pred = np.asarray(np.argmax(pred, axis=1), dtype=np.int)
        print("np.unique(pred) mask: ", np.unique(pred))
        pred = self.colorize(pred)
        # print("colorize np.unique(pred) : ", np.unique(pred))
        pred = vutils.make_grid(torch.tensor(pred), normalize=False, scale_each=False)
        # tps_imgs_viz = vutils.make_grid(images, normalize=False, scale_each=False)
        # print("tps_imgs_viz.shape, pred.shape ", tps_imgs_viz.shape, pred.shape)
        # pred = (tps_imgs_viz + pred)/2
        wandb.log({f'{setting}/{name}': wandb.Image(pred)}, step=i_iter)
        

    # def vis_slots(self, setting, i_iter, name, recons, masks):
    #     idx = 0
    #     recons = renormalize(recons)[idx]
    #     masks = masks[idx]
    #     num_slots = len(masks)
    #     print("Num of slots: ", num_slots)
    #     slots_list = []
    #     for i in range(num_slots):
    #         slot = recons[i] * masks[i] + (1 - masks[i])
    #         slot = slot.permute(2, 0, 1)
    #         slots_list.append(slot)
    #         print(slot.shape)
    #     stack_slots = torch.stack(slots_list, dim = 0)
    #     print("stack_slots : ", stack_slots.shape)
    #     tps_slots = vutils.make_grid((stack_slots), normalize=False, scale_each=False)
    #     wandb.log({f'{setting}/{name}': wandb.Image(tps_slots)}, step=i_iter)

    def vis_part_heatmaps(self, setting, i_iter, images, response_maps, name, threshold=0.5):
            B,K,H,W = response_maps.shape
            part_response = np.zeros((B,K,H,W,3)).astype(np.uint8)

            for b in range(B):
                for k in range(K):
                    response_map = response_maps[b,k,...].cpu().numpy()
                    response_map = cv2.applyColorMap((response_map*255.0).astype(np.uint8), cv2.COLORMAP_HOT)[:,:,::-1] # BGR->RGB
                    part_response[b,k,:,:,:] = response_map.astype(np.uint8)

            part_response = part_response.transpose(0,1,4,2,3)
            part_response = torch.tensor(part_response.astype(np.float32))
            for k in range(K):
                map_viz_single = vutils.make_grid(part_response[:,k,:,:,:].squeeze()/255.0, normalize=False, scale_each=False)
                wandb.log({f'{setting}/{name}': wandb.Image(map_viz_single)}, step=i_iter)

            # color segmentation
            response_maps_np = response_maps.cpu().numpy()
            response_maps_np = np.concatenate((np.ones((B,1,H,W))*threshold, response_maps_np), axis=1)
            response_maps_np = np.asarray(np.argmax(response_maps_np, axis=1), dtype=np.int)
            response_maps_np = self.colorize(response_maps_np)
            response_maps_np = vutils.make_grid(torch.tensor(response_maps_np), normalize=False, scale_each=False)
            tps_imgs_viz = vutils.make_grid(images, normalize=False, scale_each=False)
            print("tps_imgs_viz.shape, response_maps_np.shape ", tps_imgs_viz.shape, response_maps_np.shape)
            response_maps_np = (tps_imgs_viz + response_maps_np)/2
            wandb.log({f'{setting}/{"heatmap_np"}': wandb.Image(response_maps_np)}, step=i_iter)

