from datasets.pascal_parts import PPDataset
import copy
import glob
import os
import os.path as osp
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
import matplotlib.pyplot as plt
import cv2

# solve potential deadlock https://github.com/pytorch/pytorch/issues/1355
cv2.setNumThreads(0)

import hydra
import torch.backends.cudnn as cudnn
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from datasets.lit_dataset import LitDataset
from utils.utils import seed_worker
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

######################################
# create train file                  #
######################################
image_list = os.listdir('/Users/harsh/Thesis/pascal_part_manipulation/dataset/bird/original/')
image_list = [os.path.splitext(x)[0] for x in image_list]
image_list
with open('temp.txt', 'w') as f:
    for line in image_list:
        f.write(f"{line+' -1'}\n")


@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):
    cfg = dict(cfg)
    if cfg['dataset_name'] is not None:
        cfg['dataset'] = cfg['dataset_name']

    
    wandb.login()
    wandb.init(project='unsup-parts')
    wandb.config.update(cfg)
    args = wandb.config
    cudnn.enabled = True
    print(args)

    # if args.exp_name is not None:
    #     api = wandb.Api()
    #     run = api.run(path=f'wandb_userid/unsup-parts/{wandb.run.id}')
    #     run.name = f'{args.exp_name}-{run.name}'
    #     run.save()

    print("---------------------------------------")
    print(f"Arguments received: ")
    print("---------------------------------------")
    for k, v in sorted(args.__dict__.items()):
        print(f"{k:25}: {v}")
    print("---------------------------------------")
    print(osp.join('/home/harsh21122/tmp/models/unsup_parts', 'model_bird_' + str(args.num_steps) + '.pth'))
    print(osp.join('/home/harsh21122/tmp/models/unsup_parts', 'model_'+ str(args.pascal_class) + '_'+ str(args.num_steps) + '.pth'))
    if args.dataset == 'PP':
        from datasets import pascal_parts
        args.split = 'train'
        train_dataset = pascal_parts.PPDataset(args)

    else:
        print(f'Invalid dataset {args.dataset}')
        sys.exit()

    trainloader = DataLoader(
        LitDataset(train_dataset, args.use_lab),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True)
    print(trainloader)
    print(train_dataset)
    

    trainloader_iter = enumerate(trainloader)
    # it = iter(trainloader)
    # first = next(it)
    # print(first['img_vgg'].shape, first['img'].shape, first['img_rec'].shape, first['img_cj2'].shape, first['img_cj1'].shape, first['mask'].shape, first['seg'].shape)
    # plt.imshow(first['img_vgg'].permute(0, 2, 3, 1).numpy()[0])
    # plt.show()
    # plt.imshow(first['img'].permute(0, 2, 3, 1).numpy()[0])
    # plt.show()
    # plt.imshow(first['img_cj2'].permute(0, 2, 3, 1).numpy()[0])
    # plt.show()
    # plt.imshow(first['img_cj1'].permute(0, 2, 3, 1).numpy()[0])
    # plt.show()
    # plt.imshow(first['mask'].permute(0, 2, 3, 1).numpy()[0])
    # plt.show()
    # plt.imshow(first['seg'].permute(0, 2, 3, 1).numpy()[0])
    # plt.show()

    

if __name__ == '__main__':
    main()

