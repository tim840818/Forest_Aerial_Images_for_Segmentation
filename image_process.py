import os, sys

import numpy as np
import cv2
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.model_selection import train_test_split, KFold

## Prprocess for label images
def rgb2bin(img, threshold=IMAGE_THRESHOLD):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return img

## Preprocess for images
def _std_contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    return np.std(gray)

def enhance_contrast_rgb(img, thresh=50.):
    std = _std_contrast(img)
    if std > thresh:
        return img
    # Split the image into R, G, B channels
    b, g, r = cv2.split(img)
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Apply CLAHE to each channel
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    # Merge the enhanced channels back
    enhanced = cv2.merge((b, g, r))
    return enhanced

## Augmentations for images
def aug_shape(img, idx=np.random.randint(0, 7)):
    ## rotation
    img_aug1 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_aug2 = cv2.rotate(img, cv2.ROTATE_180)
    img_aug3 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    ## flip
    img_aug4 = cv2.flip(img, 0)
    img_aug5 = cv2.flip(img, 1)
    ## flip diagonal
    img_aug6 = cv2.flip(img_aug1, 0)
    img_aug7 = cv2.flip(img_aug1, 1)
    ## return one of the augmentations randomly
    img_augs = [img_aug1, img_aug2, img_aug3, img_aug4, img_aug5, img_aug6, img_aug7]
    ## idx is decided randomly and provided, return the corresponding augmentation
    return img_augs[idx], f"shape{idx+1}"

def _erase_get_params(
    scale=(10, 25),
    start=0
):
    h, w = IMAGE_SHAPE
    h_size = np.random.randint(*scale)
    w_size = np.random.randint(*scale)
    start_max = min(IMAGE_SHAPE[0] - h_size, IMAGE_SHAPE[1] - w_size)
    if start > start_max:
        start = start_max
    h_start = np.random.randint(start, start_max)
    w_start = np.random.randint(start, start_max)
    return (h_size, w_size), (h_start, w_start)
def aug_erase(img, size=(0, 0), start=(0, 0)):
    img_erase = img.copy()
    h_start, w_start = start
    img_erase[h_start:h_start+size[0], w_start:w_start+size[1]] = 0
    return img_erase, "erase"

def _crop_get_params(
    scale=(216, 240),
    start=0
):
    h, w = IMAGE_SHAPE
    size = np.random.randint(*scale)
    start_max = IMAGE_SHAPE[0] - size
    if start > start_max:
        start = start_max
    h_start = np.random.randint(start, start_max)
    w_start = np.random.randint(start, start_max)
    return (size,)*2, (h_start, w_start)
def aug_crop_resize(img, size=IMAGE_SHAPE, start=(0, 0)):
    h_start, w_start = start
    ## crop the image
    img_crop = img[h_start:h_start+size[0], w_start:w_start+size[1]]
    ## resize the image
    img_crop = cv2.resize(img_crop, IMAGE_SHAPE)
    return img_crop, "crop"

def aug_brightcontrast(img, alpha=1.0, beta=0.0):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta), "brightcontrast"

def aug_gaussian_noise(img, mean=0, std=10):
    row, col, ch = img.shape
    gauss = np.random.normal(mean, std, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = (img + gauss).round().clip(0, 255).astype(np.uint8)
    return noisy, "noise"

def aug_gaussian_blur(img, kernel_size=(3, 3), sigma=0):
    return cv2.GaussianBlur(img, kernel_size, sigma), "blur"

def apply_augmentation(sample, label, tag, augidx): # apply a single augmentation regarding the given augidx
    match augidx:
        ## apply shape augmentation
        case 1: # apply shape augmentation to (sample, label) simultaneously # if np.random.rand() < 0.5:
            idx_shape = np.random.randint(0, 7)
            sample, aug_tag = aug_shape(sample, idx_shape)
            label, _ = aug_shape(label, idx_shape)
            tag += f"_{aug_tag}"
        ## apply erase augmentation
        case 2: # apply erase augmentation to (sample, label) simultaneously # if np.random.rand() < 0.4:
            size, start = _erase_get_params()
            sample, aug_tag = aug_erase(sample, size, start)
            label, _ = aug_erase(label, size, start)
            tag += f"_{aug_tag}"
        ## apply crop augmentation
        case 3: # apply crop augmentation to (sample, label) simultaneously # if np.random.rand() < 0.5:
            size, start = _crop_get_params()
            sample, aug_tag = aug_crop_resize(sample, size, start)
            label, _ = aug_crop_resize(label, size, start)
            tag += f"_{aug_tag}"
        # ## apply brightness/contrast augmentation
        # if np.random.rand() < 0.4: # apply brightness/contrast augmentation to sample
        #     alpha = np.random.uniform(0.85, 1.1)
        #     beta = np.random.uniform(-20, 15)
        #     sample, aug_tag = aug_brightcontrast(sample, alpha, beta)
        #     tag += f"_{aug_tag}"
        ## apply noise augmentation
        case 4: # apply noise augmentation to sample # if np.random.rand() < 0.4:
            sample, aug_tag = aug_gaussian_noise(sample)
            tag += f"_{aug_tag}"
        ## apply blur augmentation
        case 5: # apply blur augmentation to sample # if np.random.rand() < 0.3:
            sample, aug_tag = aug_gaussian_blur(sample)
            tag += f"_{aug_tag}"
    return sample, label, tag
        # ## apply elastic augmentation
        # if np.random.rand() < 0.4: # apply elastic augmentation to sample
        #     alpha = np.random.uniform(15, 40)
        #     elastic_transform = T.Compose([
        #         T.ToTensor(),
        #         T.ElasticTransform(alpha=alpha)
        #     ])
        #     sample = elastic_transform(sample)
        #     tag += "_elastic"
        #     return sample, self.transform(label), tag


class ImageDataset(Dataset):
    def __init__(self, image_shape, img_dir, label_dir, out_channel=1, data_list=None, mode="train", aug=None): # file_format: 'xxx.png'
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.mode = mode # train, val, test
        self.aug = aug
        self.blank = np.zeros((*image_shape, out_channel), dtype=np.uint8)
        ## organize files into X, Y, tag
        img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
        label_files = [f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]

        if not data_list: # if data_list is not provided
            data_list = list(set(img_files) - set(label_files)) if self.mode == "test" else list(set(img_files) & set(label_files))
        
        self.tag = [f.replace('.png', '') for f in data_list]
        self.X = np.array([cv2.imread(os.path.join(img_dir, f)) for f in data_list])
        self.Y = np.array([rgb2bin(cv2.imread(os.path.join(label_dir, f))) for f in data_list]) if self.mode != "test" else self.blank

        self.transform = T.Compose([T.ToTensor()]) # convert to tensor and normalize to [0, 1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        tag = self.tag[idx]
        ### image preprocessing
        ## enhance contrast for low-contrast images
        sample = enhance_contrast_rgb(sample)
        ## no labels for test data
        if self.mode == "test":
            return self.transform(sample), self.transform(self.blank), tag
        label = self.Y[idx]
        if self.mode == "val":
            return self.transform(sample), self.transform(label), self.tag[idx]
        ### apply augmentation with probability 0.5 (for traning)
        # torch has built-in augmentation, but when applying the same rotation/flip/crop to both sample and label, it might be better to do it manually
        if self.aug and np.random.rand() < 0.5: # apply augmentation with probability 0.5
            ## first decide how many augmentaitons to apply (1-3)
            n_aug = np.random.randint(1, 4)
            ## pick n_aug different augmentations from 1-5 and sort them
            # case 1: shape augmentation | case 2: erase augmentation | case 3: crop augmentation | case 4: noise augmentation | case 5: blur augmentation
            augidxs = np.random.choice(range(1, 6), n_aug, replace=False, p=[0.3, 0.15, 0.25, 0.15, 0.15])
            augidxs.sort()
            for augidx in augidxs:
                sample, label, tag = apply_augmentation(sample, label, tag, augidx)
            # ================================================================================================== #

        return self.transform(sample), self.transform(label), tag

