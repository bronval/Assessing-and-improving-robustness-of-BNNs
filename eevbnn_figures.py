

#################################################################
#                                                               #
# Master's thesis : Binarized Neural Networks                   #
# functions to show and save figures                            #
#                                                               #
# Author : Benoit Ronval                                        #
#                                                               #
#################################################################


## IMPORT

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from eevbnn.satenv import SolveResult
from eevbnn.eval_bin import ModelVerifier, init_argparser
from eevbnn.net_bin import BinLinear, BinConv2d, BinConv2dPos, BinLinearPos, InputQuantizer, MnistMLP, TernaryWeightWithMaskFn, SeqBinModelHelper, Binarize01Act, BatchNormStatsCallbak, setattr_inplace
from eevbnn.utils import Flatten, ModelHelper, torch_as_npy
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pickle
import time

from eevbnn_models import *

#======================================================================================================================================
## 


def save_picture(img, index, clsfy, dataset_name, label, type, adv=False, eps=None):
    if dataset_name not in DATASET_NAMES and not dataset_name == "epsilon":
        print(f"ERROR save_picture function\n Invalid dataset name : {dataset_name}")

    if dataset_name == "CIFAR10":
        # for CIFAR10 dataset
        plt.imshow(img.transpose((1, 2, 0)), interpolation='nearest')
    else:
        plt.imshow(img.squeeze(), cmap="gray")
    filename = ""
    if adv:
        plt.title(f"classified as {clsfy} (adv, actual : {label}, eps={eps})\nmodel{type}")
        filename = f"saved_imgs_datasets/{dataset_name}/img_{index}_model{type}_adv_{eps}.png"
    else:
        plt.title(f"classified as {clsfy} (original, actual : {label})\nmodel{type}")
        filename = f"saved_imgs_datasets/{dataset_name}/img_{index}_model{type}.png"
    plt.savefig(filename)
    return


