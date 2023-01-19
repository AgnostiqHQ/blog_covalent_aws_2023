"""
Covalent-wrapped execution of a machine learning model for detecting
anomalous tissue in MRI scans of human brains.

This script is adapted from the following code:
    https://github.com/mateuszbuda/brain-segmentation-pytorch
    https://www.kaggle.com/code/mateuszbuda/brain-segmentation-pytorch/script

Input data can be downloaded here:
    https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

Original author: Mateusz Buda (https://www.kaggle.com/mateuszbuda)

Adapted by: Ara Ghukasyan (https://github.com/araghukas) @ AgnostiqHQ

Adapted date: Jan 3, 2023
"""
import os
import random
import sys
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from zipfile import ZipFile

import boto3
import cloudpickle as pickle
import covalent as ct
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave
from skimage.transform import rescale, resize, rotate
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

_TIMESTAMP = datetime.now().strftime("%m-%d-%Y_%H-%M-%S-%f")

# ==============================================================================
#                               ADAPTED CODE
# ==============================================================================


def crop_sample(v, m):
    """threshold and resize volume-mask pair"""
    v[v < np.max(v) * 0.1] = 0

    z_projection = np.max(np.max(np.max(v, axis=-1), axis=-1), axis=-1)
    z_nonzero = np.nonzero(z_projection)
    z_min = np.min(z_nonzero)
    z_max = np.max(z_nonzero) + 1

    y_projection = np.max(np.max(np.max(v, axis=0), axis=-1), axis=-1)
    y_nonzero = np.nonzero(y_projection)
    y_min = np.min(y_nonzero)
    y_max = np.max(y_nonzero) + 1

    x_projection = np.max(np.max(np.max(v, axis=0), axis=0), axis=-1)
    x_nonzero = np.nonzero(x_projection)
    x_min = np.min(x_nonzero)
    x_max = np.max(x_nonzero) + 1

    v = v[z_min:z_max, y_min:y_max, x_min:x_max]
    m = m[z_min:z_max, y_min:y_max, x_min:x_max]

    return v, m


def pad_sample(v, m):
    """add padding to volume-mask pair"""
    a = v.shape[1]
    b = v.shape[2]

    if a == b:
        return v, m

    diff = (max(a, b) - min(a, b)) / 2.0

    if a > b:
        padding = ((0, 0), (0, 0), (int(np.floor(diff)), int(np.ceil(diff))))
    else:
        padding = ((0, 0), (int(np.floor(diff)), int(np.ceil(diff))), (0, 0))

    m = np.pad(m, padding, mode="constant", constant_values=0)
    padding = padding + ((0, 0),)
    v = np.pad(v, padding, mode="constant", constant_values=0)

    return v, m


def resize_sample(v, m, size: int):
    """resize and volume and mask images"""
    v_shape = v.shape

    out_shape = (v_shape[0], size, size)
    m = resize(m,
               output_shape=out_shape,
               order=0,
               mode="constant",
               cval=0,
               anti_aliasing=False)

    out_shape = out_shape + (v_shape[3],)
    v = resize(v,
               output_shape=out_shape,
               order=2,
               mode="constant",
               cval=0,
               anti_aliasing=False)

    return v, m


def normalize_volume(v):
    """normalize the volume image by rescaling intensity"""
    p10 = np.percentile(v, 10)
    p99 = np.percentile(v, 99)

    v = rescale_intensity(v, in_range=(p10, p99))
    m = np.mean(v, axis=(0, 1, 2))
    s = np.std(v, axis=(0, 1, 2))

    v = (v - m) / s

    return v


def normalize_mask(m):
    """normalize the mask image by dividing through with maximum value"""
    mask_max = np.max(m)
    if mask_max > 0:
        return m / mask_max
    return m


class Dataset:
    """
    A pickle-able wrapper class for iterating over datasets.
    Assumes image data is stored as PKL files.
    """

    def __init__(self,
                 data_access_dict: Dict[str, Dict[str, Any]],
                 random_sampling: bool,
                 batch_size: int,
                 shuffle: bool,
                 drop_last: bool,
                 random_seed: Optional[int] = None):
        """Note: always random sampling; equal weights"""
        self._random_sampling = random_sampling
        self._batch_size = batch_size
        self._shuffle = shuffle if shuffle else None
        self._drop_last = drop_last
        self._random_seed = random_seed  # also use in `_init_data`

        # initialize data
        self._build_access_index(data_access_dict["volumes"])
        self._init_data(self._get_iterator(data_access_dict["volumes"],
                                           data_access_dict["masks"],
                                           data_access_dict["weights"]))

    def _build_access_index(self, volumes: Dict[str, List[Path]]) -> None:

        slices = []
        patients = []
        for patient in sorted(volumes):

            if len(volumes[patient]) == 0:
                print(f"no volumes for {patient}")
                continue

            indices = []
            for i, volume_path in enumerate(volumes[patient]):
                idx = int(volume_path.name.rsplit("_", maxsplit=1)[1].split(".")[0])
                patients.append(patient)
                indices.append((i, idx))

            slices.extend(indices)

        self._slices = np.array(slices)
        self._patients = np.array(patients)

    def _get_iterator(self,
                      volumes: Dict[str, List[Path]],
                      masks: Dict[str, List[Path]],
                      weights: Dict[str, List[float]]) -> Iterator:

        np.random.seed(self._random_seed)

        # build flattened list of slice indices
        rand_slices: List[Tuple[int, int]] = []
        rand_patients: List[str] = []
        for patient in np.unique(self._patients):

            locs = self._patients == patient
            indices = np.array([s for i, s in enumerate(self._slices) if locs[i]])

            if self._random_sampling:
                rand_idx = np.random.choice(range(indices.shape[0]),
                                            size=indices.shape[0],
                                            p=weights[patient])
                indices = indices[rand_idx]

            rand_patients.extend([patient] * indices.shape[0])
            rand_slices.extend(indices)

        # construct data store
        data = []
        for patient, index in zip(rand_patients, rand_slices):
            data.append([str(volumes[patient][index[0]]),
                         str(masks[patient][index[0]])])

        self._slices = np.array(rand_slices)
        self._patients = np.array(rand_patients)

        return iter(DataLoader(data,  # type: ignore
                               num_workers=1,
                               batch_size=self._batch_size,
                               shuffle=self._shuffle,
                               drop_last=self._drop_last))

    def _init_data(self, iterator):
        """populate the data store"""
        self._data_store = []

        x = next(iterator, None)
        while x is not None:
            self._data_store.append(x)
            x = next(iterator, None)

        self._length = len(self._data_store)

    @property
    def batch_size(self):
        """images batch size"""
        return self._batch_size

    @property
    def slices(self):
        """1D array of slice indices"""
        return np.array(self._slices)

    @property
    def patients(self):
        """1D array of patient ids"""
        return np.array(self._patients)

    @property
    def patient_index(self):
        """2D array of (patient_id, slice_number)"""
        return np.stack((self._patients, self._slices[:, 1])).T

    def __getitem__(self, idx):
        # tuples f `batch_size` many volume and matching mask files
        volume_paths, mask_paths = self._data_store[idx]
        vm_pairs = []
        for vp, mp in zip(volume_paths, mask_paths):
            with open(vp, "rb") as volume_file_obj:
                v = pickle.load(volume_file_obj)
            with open(mp, "rb") as mask_file_obj:
                m = pickle.load(mask_file_obj)
            vm_pairs.append((v, m))

        vm_pairs = (np.stack([v[0] for v in vm_pairs]), np.stack([v[1] for v in vm_pairs]))

        vm_pairs = (torch.from_numpy(vm_pairs[0].transpose(0, 3, 1, 2).astype(np.float32)),
                    torch.from_numpy(vm_pairs[1].transpose(0, 3, 1, 2).astype(np.float32)))

        return vm_pairs


class Scale:
    """scaling transformation"""

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        image, mask = sample
        img_size = image.shape[0]

        scale = np.random.uniform(low=1.0 - self.scale, high=1.0 + self.scale)

        image = rescale(image,
                        (scale, scale),
                        channel_axis=-1,
                        preserve_range=True,
                        mode="constant",
                        anti_aliasing=False)

        mask = rescale(mask,
                       (scale, scale),
                       order=0,
                       channel_axis=-1,
                       preserve_range=True,
                       mode="constant",
                       anti_aliasing=False)

        if scale < 1.0:
            diff = (img_size - image.shape[0]) / 2.0
            padding = ((int(np.floor(diff)), int(np.ceil(diff))),) * 2 + ((0, 0),)
            image = np.pad(image, padding, mode="constant", constant_values=0)
            mask = np.pad(mask, padding, mode="constant", constant_values=0)
        else:
            x_min = (image.shape[0] - img_size) // 2
            x_max = x_min + img_size
            image = image[x_min:x_max, x_min:x_max, ...]
            mask = mask[x_min:x_max, x_min:x_max, ...]

        return image, mask


class Rotate:
    """rotation transformation"""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image, mask = sample

        angle = np.random.uniform(low=-self.angle, high=self.angle)
        image = rotate(image,
                       angle,
                       resize=False,
                       preserve_range=True,
                       mode="constant")

        mask = rotate(mask,
                      angle,
                      resize=False,
                      order=0,
                      preserve_range=True,
                      mode="constant")

        return image, mask


class HorizontalFlip:
    """transformation that randomly flips the image"""

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.flip_prob:
            return image, mask

        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

        return image, mask


def get_transform(scale: float, angle: float, flip_prob: float) -> Compose:
    """A composition of pre-defined image transforms"""
    transform_list = []

    if scale is not None:
        transform_list.append(Scale(scale))
    if angle is not None:
        transform_list.append(Rotate(angle))
    if flip_prob is not None:
        transform_list.append(HorizontalFlip(flip_prob))

    return Compose(transform_list)


class UNet(nn.Module):
    """Neural network definition"""

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super().__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16,
                                          features * 8,
                                          kernel_size=2,
                                          stride=2)

        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(features * 8,
                                          features * 4,
                                          kernel_size=2,
                                          stride=2)

        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(features * 4,
                                          features * 2,
                                          kernel_size=2,
                                          stride=2)

        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(features * 2,
                                          features,
                                          kernel_size=2,
                                          stride=2)

        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features,
                              out_channels=out_channels,
                              kernel_size=1)

    def forward(self, x):
        """execute when called"""
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):

        od = OrderedDict([(name + "conv1",
                         nn.Conv2d(in_channels=in_channels,
                                   out_channels=features,
                                   kernel_size=3,
                                   padding=1,
                                   bias=False)),

                          (name + "norm1",
                         nn.BatchNorm2d(num_features=features)),

                          (name + "relu1",
                          nn.ReLU(inplace=True)),

                          (name + "conv2",
                           nn.Conv2d(in_channels=features,
                                     out_channels=features,
                                     kernel_size=3,
                                     padding=1,
                                     bias=False)),

                          (name + "norm2",
                           nn.BatchNorm2d(num_features=features)),

                          (name + "relu2",
                           nn.ReLU(inplace=True))])

        return nn.Sequential(od)


class DiceLoss(nn.Module):
    """wrapper for dice loss"""

    def __init__(self):
        super().__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        """execute when called"""
        assert y_pred.size() == y_true.size()

        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()

        d = (2. * intersection + self.smooth)
        d /= (y_pred.sum() + y_true.sum() + self.smooth)

        return 1. - d


def gray2rgb(image):
    """converts a gray image to an rgb image"""
    w, h = image.shape
    image += np.abs(np.min(image))
    image_max = np.abs(np.max(image))

    if image_max > 0:
        image /= image_max

    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = image * 255

    return ret


def outline(image, mask, color):
    """draws an outline on the image from hatch in the mask"""
    mask = np.round(mask)
    yy, xx = np.nonzero(mask)

    for y, x in zip(yy, xx):
        if 0.0 < np.mean(mask[max(0, y - 1): y + 2, max(0, x - 1): x + 2]) < 1.0:
            image[max(0, y): y + 1, max(0, x): x + 1] = color

    return image


def dsc(y_pred, y_true):
    """basic dice loss function between two arrays"""
    y_pred = np.round(y_pred).astype(int)
    y_true = np.round(y_true).astype(int)

    y_sum = (np.sum(y_pred) + np.sum(y_true))

    loss = np.sum(y_pred[y_true == 1]) * 2.0
    if y_sum > 0:
        loss /= y_sum
    return loss


def dsc_distribution(volumes):
    """dice loss function by patient"""
    dsc_dict = {}

    for p in volumes:
        y_pred = volumes[p][1]
        y_true = volumes[p][2]
        dsc_dict[p] = dsc(y_pred, y_true)

    return dsc_dict


def dsc_per_volume(validation_pred, validation_true, patient_index):
    """dice losses for each slice for a every patient"""
    dsc_list = []
    unique_patients = np.unique(patient_index[:, 0])
    num_slices = [len(patient_index[patient_index == p]) for p in unique_patients]
    index = 0

    for nl in num_slices:
        y_pred = np.array(validation_pred[index: index + nl])
        y_true = np.array(validation_true[index: index + nl])
        dsc_list.append(dsc(y_pred, y_true))
        index += nl

    return dsc_list


def postprocess_per_volume(input_list, pred_list, true_list, patient_index, patients):
    """input, predictions, and ground truth over all results for each patient"""
    volumes = {}
    unique_patients = np.unique(patients)
    num_slices = [len(patient_index[patient_index == p]) for p in unique_patients]
    index = 0

    for p, sl in enumerate(num_slices):
        volume_in = np.array(input_list[index: index + sl])
        volume_pred = np.round(np.array(pred_list[index: index + sl])).astype(int)
        volume_true = np.array(true_list[index: index + sl])
        volumes[unique_patients[p]] = (volume_in, volume_pred, volume_true)
        index += sl

    return volumes


def plot_dsc(dsc_dist: dict, output_dir: Path) -> np.ndarray:
    """makes horizontal bar graph of dice coefficients by patient ID"""
    y_positions = np.arange(len(dsc_dist))
    dsc_dist = sorted(dsc_dist.items(), key=lambda x: x[1])
    values = [x[1] for x in dsc_dist]
    labels = [str(x[0]) for x in dsc_dist]
    labels = ["_".join(l.split("_")[1:-1]) for l in labels]
    fig = plt.figure(figsize=(12, 8))
    canvas = FigureCanvasAgg(fig)
    plt.barh(y_positions, values, align="center", color="skyblue")
    plt.yticks(y_positions, labels)
    plt.xticks(np.arange(0.0, 1.0, 0.1))
    plt.xlim([0.0, 1.0])
    plt.gca().axvline(np.mean(values), color="tomato", linewidth=2)
    plt.gca().axvline(np.median(values), color="forestgreen", linewidth=2)
    plt.xlabel("Dice coefficient", fontsize="x-large")
    plt.gca().xaxis.grid(color="silver", alpha=0.5, linestyle="--", linewidth=1)
    plt.tight_layout()
    canvas.draw()
    plt.close()
    s, (width, height) = canvas.print_to_buffer()

    # also create record file
    with open(output_dir / "dsc_dist.csv", "w", encoding="utf8") as f:
        f.write("patient_id,dsc\n")
        for patient_id, _dsc in reversed(dsc_dist):
            f.write(f"{patient_id}, {_dsc}\n")
        f.write(f"mean, {sum(x[1] for x in dsc_dist) / len(dsc_dist)}\n")

    return np.frombuffer(s, np.uint8).reshape((height, width, 4))


def get_access_dict(images_dir: Path,
                    extension: str,
                    split_fraction: float = 0.75) -> Dict[str, Dict[str, Dict[str, List[Any]]]]:
    """
    Find all image data files with the given extension and assign for training/validation.

    Returns a dictionary:
    {
        "primary": {
            "volumes": volumes_dict_train,
            "masks": masks_dict_train,
            "weights": weights_dict_train
        },
        "secondary": {
            "volumes": volumes_dict_valid,
            "masks": masks_dict_valid,
            "weights": weights_dict_valid
        }
    }
    """
    # find all mask files
    masks = []
    for mask_path in images_dir.rglob(f"*_mask.{extension}"):
        patient_id = mask_path.parent.name  # ex: TCGA_HT_7693_19950520
        mp = mask_path.relative_to(mask_path.parent.parent.parent)
        masks.append((patient_id, mp))

    # assign train/validation subsets
    n_masks = len(masks)
    n_train = int(split_fraction * n_masks)
    indices = set(range(n_masks))
    indices_train = set(random.sample(indices, n_train))
    indices_valid = indices - indices_train

    # convert to sorted lists of indices
    indices_train = sorted(indices_train)
    indices_valid = sorted(indices_valid)

    # build mask and volume dictionaries by patient id for training
    masks_dict_train: Dict[str, List[Path]] = {}
    for patient_id, mp in [masks[i] for i in indices_train]:
        if patient_id in masks_dict_train:
            masks_dict_train[patient_id].append(mp)
        else:
            masks_dict_train[patient_id] = [mp]

    volumes_dict_train: Dict[str, List[Path]] = {}
    for patient_id, mask_paths in masks_dict_train.items():
        volumes = []
        for mask_path in mask_paths:
            volumes.append(mask_path.parent / mask_path.name.replace("_mask", ""))
        volumes_dict_train[patient_id] = volumes

    # compute training image weights from hatch size on each mask
    weights_dict_train: Dict[str, List[float]] = {}
    for patient_id, mask_paths in masks_dict_train.items():
        weights_dict_train[patient_id] = []
        for mask_file in mask_paths:
            # have to read volume to determine weight
            if extension == "pkl":
                with open(mask_file, "rb") as mask_file_obj:
                    arr = pickle.load(mask_file_obj)
                weights = arr.sum(axis=0).sum(axis=0)
            else:
                arr = imread(mask_file, as_gray=True)
            weights_dict_train[patient_id].append(arr.sum())

    normalized_weights_dict_train: Dict[str, List[float]] = {}
    for patient_id, weights in weights_dict_train.items():
        sum_weights = sum(weights)
        if sum_weights != 0:
            normalized_weights_dict_train[patient_id] = np.array(weights) / sum_weights
        else:
            print(f"sum of weights is 0 for patient {patient_id}")
            n_w = len(weights)
            normalized_weights_dict_train[patient_id] = np.ones(n_w) / n_w

    # build mask and volume dictionaries by patient id for validation
    masks_dict_valid: Dict[str, List[Path]] = {}
    for patient_id, mp in [masks[i] for i in indices_valid]:
        if patient_id in masks_dict_valid:
            masks_dict_valid[patient_id].append(mp)
        else:
            masks_dict_valid[patient_id] = [mp]

    volumes_dict_valid: Dict[str, List[Path]] = {}
    for patient_id, mask_paths in masks_dict_valid.items():
        volumes = []
        for mask_path in mask_paths:
            volumes.append(mask_path.parent / mask_path.name.replace("_mask", ""))
        volumes_dict_valid[patient_id] = volumes

    # compute validation image weights from hatch size on each mask
    weights_dict_valid: Dict[str, List[float]] = {}
    for patient_id, mask_paths in masks_dict_valid.items():
        weights_dict_valid[patient_id] = []
        for mask_file in mask_paths:
            # have to read volume to determine weight
            if extension == "pkl":
                with open(mask_file, "rb") as mask_file_obj:
                    arr = pickle.load(mask_file_obj)
                weights = arr.sum(axis=0).sum(axis=0)
            else:
                arr = imread(mask_file, as_gray=True)
            weights_dict_valid[patient_id].append(arr.sum())

    normalized_weights_dict_valid: Dict[str, List[float]] = {}
    for patient_id, weights in weights_dict_valid.items():
        sum_weights = sum(weights)
        if sum_weights != 0:
            normalized_weights_dict_valid[patient_id] = np.array(weights) / sum_weights
        else:
            print(f"sum of weights is 0 for patient {patient_id}")
            n_w = len(weights)
            normalized_weights_dict_valid[patient_id] = np.ones(n_w) / n_w

    return {
        "primary": {
            "volumes": volumes_dict_train,
            "masks": masks_dict_train,
            "weights": normalized_weights_dict_train
        },
        "secondary": {
            "volumes": volumes_dict_valid,
            "masks": masks_dict_valid,
            "weights": normalized_weights_dict_valid
        }
    }


# ==============================================================================
#                                  PARSER
# ==============================================================================
_PARSER = ArgumentParser(
    prog="python workflow2.py",
    description="dispatch the ML workflow using covalent and AWSBatch"
)

_PARSER.add_argument("-B",
                     help="list of batch sizes, number of images per step",
                     type=int,
                     nargs="+",
                     metavar="b",
                     default=[16])
_PARSER.add_argument("-E",
                     help="list of number of epochs",
                     type=int,
                     nargs="+",
                     metavar="e",
                     default=[10])
_PARSER.add_argument("-Z",
                     help="image sizes to resize images to",
                     type=int,
                     nargs="+",
                     metavar="e",
                     default=[112])
_PARSER.add_argument("-L",
                     help="learnings rate for Adam optimizer",
                     type=float,
                     nargs="+",
                     metavar="l",
                     default=[0.0001])

_PARSER.add_argument("-s",
                     help="scaling factor for `Scale` transformation",
                     type=float,
                     metavar="AUG_SCALE",
                     default=0.05)
_PARSER.add_argument("-g",
                     help="rotation angle for `Rotate` transformation",
                     type=float,
                     metavar="AUG_ANGLE",
                     default=15)
_PARSER.add_argument("-d",
                     help="data directory name",
                     type=str,
                     metavar="NAME",
                     default="data_full")

_PARSER.add_argument("--dispatch_id",
                     help="provide dispatch ID in attempt to retrieve or reconnect",
                     type=str,
                     metavar="ID",
                     default=None)
_PARSER.add_argument("--seed",
                     help="random seed to fix data selection and order",
                     type=int,
                     metavar="N",
                     default=None)
_PARSER.add_argument("--local",
                     help="run script locally, without covalent",
                     action="store_true",
                     default=False)

# ==============================================================================
#                                   PATHS
# ==============================================================================
CWD = Path(os.environ["PWD"]).resolve()
OUTPUT_DIR = Path(f"outputs_{_TIMESTAMP}")
UNET_FILE = Path(f"unet_{_TIMESTAMP}.pt")

# ==============================================================================
#                                  EXECUTOR
# ==============================================================================
BUCKET_NAME = "my-s3-bucket"  # use same bucket for both executor and data

_EXECUTOR_KWARGS = {
    "credentials": "~/.aws/credentials",
    "region": "us-east-1",
    "s3_bucket_name": BUCKET_NAME,
    "batch_queue": "my-batch-queue",
    "batch_job_log_group_name": "my-log-group",
}

BATCH_GPU_EXECUTOR = ct.executor.AWSBatchExecutor(
    num_gpus=1,
    vcpu=2,
    memory=4,
    time_limit=4 * 3600,
    base_uri="docker.io/araghukas/covalent-executor-gpu:cuda-11.4.1",
    **_EXECUTOR_KWARGS,
)

BATCH_CPU_EXECUTOR = ct.executor.AWSBatchExecutor(
    num_gpus=0,
    vcpu=2,
    memory=4,
    time_limit=2 * 3600,
    ** _EXECUTOR_KWARGS
)

# ==============================================================================
#                                   HELPERS
# ==============================================================================


def _download_file(client, key: str, filename: str) -> None:
    """download from S3 bucket"""
    print(f"downloading: s3://{BUCKET_NAME}/{key} -> {filename}")
    client.download_file(Bucket=BUCKET_NAME,
                         Key=key,
                         Filename=filename)


def _upload_file(client, filename: str, key: str) -> None:
    """upload to S3 bucket"""
    print(f"uploading: {filename} -> s3://{BUCKET_NAME}/{key}")
    client.upload_file(Filename=filename,
                       Bucket=BUCKET_NAME,
                       Key=key)


def _download_and_unzip(client,
                        zip_filename: str) -> None:
    """download from S3 bucket and unzip"""
    # download from s3 bucket
    _download_file(client, zip_filename, zip_filename)

    # extract contents
    print(f"extracting from: {zip_filename}")
    with ZipFile(zip_filename, "r") as zipped_file:
        zipped_file.extractall()


def _zip_and_upload(client,
                    dir_path: Path,
                    key: str = "",
                    pattern: str = "*") -> None:
    """zip and upload to S3 bucket"""
    # compress directory
    print(f"compressing: {dir_path}")
    with ZipFile(f"{dir_path.name}.zip", "x") as zipped_file:
        for file_path in dir_path.rglob(pattern):
            zipped_file.write(file_path, file_path.parent / file_path.name)

    # upload to s3 bucket
    key = f"{dir_path.name}.zip" if not key else key
    _upload_file(client, filename=f"{dir_path.name}.zip", key=key)


def _preprocess(volumes_dict: Dict[str, List[Path]],
                masks_dict: Dict[str, List[Path]],
                image_size: int,
                transform: Optional[Callable],
                save_to_dir: Path) -> None:
    """pre-processes images into data arrays and pickle the data arrays"""

    print(f"number of volume/mask pairs: {sum(len(v) for _,v in volumes_dict.items())}")
    for patient_id in list(volumes_dict):
        volume_paths = volumes_dict[patient_id]
        mask_paths = masks_dict[patient_id]

        if not (volume_paths and mask_paths):
            print(f"ignored patient {patient_id}")
            continue

        print(f"preprocessing patient {patient_id} ({len(volume_paths)} slices)")
        slice_numbers = []
        vm_pairs = []
        for vf, mf in zip(volume_paths, mask_paths):
            slice_numbers.append(int(vf.name.rsplit("_", maxsplit=1)[1].split(".")[0]))
            v = imread(vf)
            m = imread(mf, as_gray=True)
            vm_pairs.append((v, m))

        vm_pairs = (
            np.stack([v[0] for v in vm_pairs]),
            np.stack([v[1] for v in vm_pairs])
        )

        vm_pairs = crop_sample(*vm_pairs)
        vm_pairs = pad_sample(*vm_pairs)
        vm_pairs = resize_sample(*vm_pairs, image_size)
        vm_pairs = normalize_volume(vm_pairs[0]), normalize_mask(vm_pairs[1])

        vm_pairs = vm_pairs[0], vm_pairs[1][..., np.newaxis]

        if transform:
            vs, ms = vm_pairs
            for i in range(vs.shape[0]):
                vs[i], ms[i] = transform((vs[i], ms[i]))
            vm_pairs = vs, ms

        # save images
        patient_dir = save_to_dir / patient_id
        patient_dir.mkdir()
        for v_array, m_array, sn in zip(*vm_pairs, slice_numbers):
            with open(patient_dir / f"{patient_id}_{sn}.pkl", "wb") as v_file:
                pickle.dump(v_array, v_file)
            with open(patient_dir / f"{patient_id}_{sn}_mask.pkl", "wb") as m_file:
                pickle.dump(m_array, m_file)


def _get_device(assert_gpu: bool) -> torch.device:
    """create and return a `device` object that uses GPU backend if available"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif assert_gpu:
        raise RuntimeError("GPU backend not available.")
    else:
        print(">>WARNING!<< GPU backend not available.")
        device = torch.device("cpu")
    return device


def _get_unet_and_optimizer(device: torch.device,
                            learning_rate: float) -> Tuple[UNet, optim.Adam]:
    """initialize the model and associate it with a device and optimizer"""

    unet = UNet()
    unet.to(device)
    optimizer = optim.Adam(unet.parameters(), lr=learning_rate)

    return unet, optimizer


@dataclass
class InputPaths:
    """preprocessed image output paths container"""
    train: Path
    valid: Path


@dataclass(frozen=True)
class Params:
    """a container for hyperparameter values"""
    batch_size: int
    epochs: int
    learning_rate: float
    image_size: int


@dataclass
class TrainingResult:
    """training outputs container"""

    # results
    compute_time: float  # in seconds
    unet_filename: str
    data_dir_train: str
    data_dir_valid: str
    label: int

    # settings
    batch_size: int
    epochs: int
    image_size: int
    learning_rate: float
    aug_scale: float
    aug_angle: float
    random_seed: Optional[int]

    # input information
    patient_index: np.ndarray = field(repr=False)


# ==============================================================================
#                                    DEPS
# ==============================================================================
DEPS_PREP = ["torchvision", "numpy", "scikit-image"]
DEPS_TRAIN = ["torch", "torchvision", "numpy", "scikit-image"]
DEPS_PRED = ["torch", "torchvision", "numpy", "scikit-image", "matplotlib"]

# ==============================================================================
#                                 PREPROCESS
# ==============================================================================


@ct.electron(executor=BATCH_CPU_EXECUTOR, deps_pip=ct.DepsPip(DEPS_PREP))
def preprocess_images(data_dir: Path,
                      image_size: int,
                      transform: Callable,
                      upload: bool) -> InputPaths:
    """Download, reduce, preprocess, and upload image data as pickled arrays"""
    # download and unzip data
    s3 = None
    if not data_dir.exists():
        s3 = boto3.client("s3")
        _download_and_unzip(s3, f"{data_dir.name}.zip")

    data_dir_train = Path(f"train-{image_size}-{data_dir.name}")
    print(f"saving training data to: {data_dir_train}")
    data_dir_train.mkdir(exist_ok=True)

    data_dir_valid = Path(f"valid-{data_dir.name}")
    print(f"saving validation data to: {data_dir_valid}")
    data_dir_valid.mkdir(exist_ok=True)

    # get disjoint training/validation paths by patient id
    access_dict = get_access_dict(data_dir,
                                  extension="tif",
                                  split_fraction=0.75)

    # pre-process training and validation data
    _preprocess(access_dict["primary"]["volumes"],
                access_dict["primary"]["masks"],
                image_size=image_size,
                transform=transform,
                save_to_dir=data_dir_train)

    if not data_dir_valid.exists() or len(list(data_dir_valid.iterdir())) == 0:
        print("validation directory already exists")
        _preprocess(access_dict["secondary"]["volumes"],
                    access_dict["secondary"]["masks"],
                    image_size=image_size,
                    transform=None,
                    save_to_dir=data_dir_valid)

    # zip and upload data
    if upload and s3:
        _zip_and_upload(s3, data_dir_train, pattern="*.pkl")
        _zip_and_upload(s3, data_dir_valid, pattern="*.pkl")

    return InputPaths(train=data_dir_train,
                      valid=data_dir_valid)

# ==============================================================================
#                                  TRAINING
# ==============================================================================


@ct.electron(executor=BATCH_GPU_EXECUTOR, deps_pip=ct.DepsPip(DEPS_TRAIN))
def train_neural_network(input_paths: InputPaths,
                         params: Params,
                         label: Any,
                         aug_scale: float,
                         aug_angle: float,
                         upload: bool,
                         random_seed: Optional[int] = None) -> TrainingResult:
    """
    Main training/validation loop.
    For each epoch, runs training and validation, compares loss to best so far,
    updates best unet on disk.
    """
    s3 = None
    if not input_paths.train.exists():
        s3 = boto3.client("s3")
        _download_and_unzip(s3, f"{input_paths.train.name}.zip")
    if not input_paths.valid.exists():
        s3 = boto3.client("s3") if s3 is None else s3
        _download_and_unzip(s3, f"{input_paths.valid.name}.zip")

    # initialize model
    device = _get_device(assert_gpu=upload)
    unet, optimizer = _get_unet_and_optimizer(device, params.learning_rate)

    # initialize dataset loaders
    access_dict_train = get_access_dict(input_paths.train,
                                        extension="pkl",
                                        split_fraction=1.0)["primary"]
    access_dict_valid = get_access_dict(input_paths.valid,
                                        extension="pkl",
                                        split_fraction=1.0)["primary"]

    loader_train = Dataset(data_access_dict=access_dict_train,
                           random_sampling=True,
                           batch_size=params.batch_size,
                           shuffle=True,
                           drop_last=True,
                           random_seed=random_seed)
    loader_valid = Dataset(data_access_dict=access_dict_valid,
                           random_sampling=False,
                           batch_size=params.batch_size,
                           shuffle=False,
                           drop_last=False)

    # train/validate over desired number of epochs
    loss_func = DiceLoss()
    loss_train = []
    loss_validation = []
    best_dsc = 0.0
    unet_file_path = Path(f"{label}-{UNET_FILE.name}")

    print("images / batch_size = "
          f"{len(loader_train.slices)} / {loader_train.batch_size} = "
          f"{len(loader_train.slices) // loader_train.batch_size}")
    print(f"training/validating for {params.epochs} epochs...")

    start_time = perf_counter()

    for epoch in range(params.epochs):

        # training loop
        unet.train()
        for i, (x, y_true) in enumerate(loader_train):
            optimizer.zero_grad()
            x, y_true = x.to(device), y_true.to(device)
            with torch.set_grad_enabled(True):
                # compute and record loss, back-propagate
                loss = loss_func(unet(x), y_true)
                loss_train.append(loss.item())
                loss.backward()
                optimizer.step()
            print(f"training: {epoch}-{i} -> {loss.item():.4f}")

        # validation loop
        unet.eval()
        validation_pred = []
        validation_true = []
        for i, (x, y_true) in enumerate(loader_valid):
            x, y_true = x.to(device), y_true.to(device)
            with torch.set_grad_enabled(False):
                # get prediction and record loss
                y_pred = unet(x)
                loss = loss_func(y_pred, y_true)
                loss_validation.append(loss.item())
            # store predicted/true labels
            validation_pred.extend(y_pred.detach().cpu().numpy())
            validation_true.extend(y_true.detach().cpu().numpy())
            print(f"validation: {epoch}-{i} -> {loss.item():.4f}")

        # calculate mean loss for given epoch
        print(f"computing mean dsc, epoch {epoch}")
        mean_dsc = np.mean(dsc_per_volume(validation_pred,
                                          validation_true,
                                          loader_valid.patient_index))

        # update unset state for best result
        if mean_dsc >= best_dsc:
            print(f"mean > best: ({mean_dsc:.4f} >= {best_dsc:.4f}) writing state dict...")
            best_dsc = mean_dsc
            torch.save(unet.state_dict(), unet_file_path)
        else:
            print(f"mean < best: ({mean_dsc:.4f} < {best_dsc:.4f})")

    end_time = perf_counter()

    # upload unet state
    if upload:
        s3 = boto3.client("s3") if s3 is None else s3
        _upload_file(s3, unet_file_path.name, unet_file_path.name)

    return TrainingResult(compute_time=end_time - start_time,
                          unet_filename=unet_file_path.name,
                          data_dir_train=input_paths.train.name,
                          data_dir_valid=input_paths.valid.name,
                          label=label,
                          batch_size=params.batch_size,
                          epochs=params.epochs,
                          image_size=params.image_size,
                          learning_rate=params.learning_rate,
                          aug_scale=aug_scale,
                          aug_angle=aug_angle,
                          random_seed=random_seed,
                          patient_index=loader_train.patient_index)

# ==============================================================================
#                                 PREDICTIONS
# ==============================================================================


@ct.electron(executor=BATCH_GPU_EXECUTOR, deps_pip=ct.DepsPip(DEPS_PRED))
def write_predictions(training_result: TrainingResult, upload: bool) -> Path:
    """
    Visualize predictions by outlining brain segmentation with green/red lines
    for ground-truth/prediction.
    """
    data_dir_valid = Path(training_result.data_dir_valid)
    unet_file_path = Path(training_result.unet_filename)

    s3 = None
    if not data_dir_valid.exists():
        s3 = boto3.client("s3")
        _download_and_unzip(s3, f"{data_dir_valid}.zip")
    if not unet_file_path.exists():
        s3 = boto3.client("s3") if s3 is None else s3
        _download_file(s3, unet_file_path.name, unet_file_path.name)

    # create output directory
    output_dir = Path(f"{training_result.label}-{OUTPUT_DIR.name}")
    output_dir.mkdir(exist_ok=True)

    # initialize model
    device = _get_device(assert_gpu=upload)
    unet, _ = _get_unet_and_optimizer(device, training_result.learning_rate)

    # load trained parameters
    print("reconstructing unet...")
    unet.load_state_dict(torch.load(training_result.unet_filename))
    unet.to(device)
    unet.eval()

    # initialize dataset loader
    access_dict_valid = get_access_dict(Path(training_result.data_dir_valid),
                                        extension="pkl",
                                        split_fraction=1.0)["primary"]

    loader_valid = Dataset(data_access_dict=access_dict_valid,
                           random_sampling=False,
                           batch_size=training_result.batch_size,
                           shuffle=False,
                           drop_last=False)

    input_list = []
    pred_list = []
    true_list = []

    # evaluate predictions
    print("evaluating predictions...")
    print("images / batch_size = "
          f"{len(loader_valid.slices)} / {loader_valid.batch_size} = "
          f"{len(loader_valid.slices) // loader_valid.batch_size}")
    for i, data in enumerate(loader_valid):
        x, y_true = data
        x, y_true = x.to(device), y_true.to(device)  # pylint: disable=no-member
        with torch.set_grad_enabled(False):
            y_pred = unet(x)
            y_pred_np = y_pred.detach().cpu().numpy()
            pred_list.extend([y_pred_np[s] for s in range(y_pred_np.shape[0])])
            y_true_np = y_true.detach().cpu().numpy()
            true_list.extend([y_true_np[s] for s in range(y_true_np.shape[0])])
            x_np = x.detach().cpu().numpy()
            input_list.extend([x_np[s] for s in range(x_np.shape[0])])
            print(f"evaluating: {i}")

    # preprocess predictions and compute losses
    print("running postprocessing...")
    volumes = postprocess_per_volume(input_list,
                                     pred_list,
                                     true_list,
                                     loader_valid.patient_index,
                                     loader_valid.patients)

    print("plotting dice loss...")
    dsc_dist = dsc_distribution(volumes)
    dsc_dist_plot = plot_dsc(dsc_dist, output_dir)
    imsave(output_dir / "dsc.png", dsc_dist_plot)

    # write output images
    print("writing output images...")
    for p, v in volumes.items():
        x = v[0]
        y_pred = v[1]
        y_true = v[2]
        for s in range(x.shape[0]):
            image = gray2rgb(x[s, 1])  # channel 1 is for FLAIR
            image = outline(image, y_pred[s, 0], color=[255, 0, 0])
            image = outline(image, y_true[s, 0], color=[0, 255, 0])
            file_name = f"{p}-{str(s).zfill(2)}.png"
            imsave(output_dir / file_name, image)

    # upload outputs
    if upload:
        s3 = boto3.client("s3") if s3 is None else s3
        _zip_and_upload(s3, output_dir)

    return output_dir

# ==============================================================================
#                                  RETRIEVE
# ==============================================================================


@ct.electron(executor="local")
def download_output(output_dir: Path) -> None:
    """use boto3 to download zipped outputs directory into cwd"""
    if not output_dir.exists():
        s3 = boto3.client("s3")
        _download_file(s3, f"{output_dir.name}.zip", f"{CWD / output_dir.name}.zip")
    else:
        print(f"local output directory exists: {output_dir}")


# ==============================================================================
#                                   LATTICE
# ==============================================================================


@ct.lattice
def workflow(data_dir: Path,
             params_list: List[Params],
             transform: Callable,
             aug_scale: float,
             aug_angle: float,
             random_seed: Optional[int],
             upload: bool) -> Dict[Params, TrainingResult]:
    """
    The workflow function that runs that hyperparameter sweep.

    Parameters
    ----------
    data_dir : Path
        name of the data directory to use, must exist as zip in S3 bucket
    params_list : List[Params]
        list of hyperparameter values to sweep over
    transform : Callable
        a custom image transformation
    aug_scale : float
        image scaling amount
    aug_angle : float
        image rotation amount
    random_seed : Optional[int]
        optional random seed to fix random outputs
    upload : bool
        flag to disables file upload/download when testing locally

    Returns
    -------
    Dict[Params, TrainingResult]
        a dictionary of hyperparameter sets and corresponding training results
    """

    # preprocess images for each unique image size
    inputs_dict: Dict[int, InputPaths] = {}
    for ps in params_list:
        if ps.image_size not in inputs_dict:
            inputs_dict[ps.image_size] = preprocess_images(data_dir=data_dir,
                                                           image_size=ps.image_size,
                                                           transform=transform,
                                                           upload=upload)

    # run training over all parameter combinations
    training_results: Dict[Params, TrainingResult] = {}
    for i, ps in enumerate(params_list):
        # training
        training_result = train_neural_network(input_paths=inputs_dict[ps.image_size],
                                               params=ps,
                                               label=i + 1,
                                               aug_scale=aug_scale,
                                               aug_angle=aug_angle,
                                               random_seed=random_seed,
                                               upload=upload)

        # predictions
        output_dir = write_predictions(training_result, upload)
        download_output(output_dir)

        # record training result
        training_results[ps] = training_result

    return training_results

# ==============================================================================
#                                    MAIN
# ==============================================================================


def main(args: Namespace) -> Dict[tuple, TrainingResult]:
    """
    MAIN
    """
    transform = get_transform(scale=args.s, angle=args.g, flip_prob=0.5)
    params_list = [Params(*ps) for ps in product(args.B, args.E, args.L, args.Z)]

    for ps in params_list:
        print(ps)

    if args.local:
        print("Running script locally...")
        return workflow(data_dir=Path(args.d),
                        params_list=params_list,
                        transform=transform,
                        aug_scale=args.s,
                        aug_angle=args.g,
                        random_seed=args.seed,
                        upload=False)

    if args.dispatch_id is None:
        print("\nEXECUTORS:")
        for executor in (BATCH_CPU_EXECUTOR, BATCH_GPU_EXECUTOR):
            for k, v in vars(executor).items():
                print(f"{k:>20} = {'None' if v is None else v:<10}")
            print()
        print(f"\nTIMESTAMP: {_TIMESTAMP}")
        print("\nARGUMENTS:")
        for k, v in vars(args).items():
            print(f"{k:>15} = {'None' if v is None else v}")
        proceed = input("\ntype 'go' to continue: ")
        if proceed != 'go':
            sys.exit(0)

        dispatch_id = ct.dispatch(workflow)(data_dir=Path(args.d),
                                            params_list=params_list,
                                            transform=transform,
                                            aug_scale=args.s,
                                            aug_angle=args.g,
                                            random_seed=args.seed,
                                            upload=True)

        print(f"\ndispatch_id: {dispatch_id}")
    else:
        dispatch_id = args.dispatch_id

    return ct.get_result(dispatch_id, wait=True).result


if __name__ == "__main__":
    print(main(_PARSER.parse_args()))
