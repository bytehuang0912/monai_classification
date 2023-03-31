import os

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from typing import Sequence, Union, Tuple
from skimage import transform
import pandas as pd
import torch 

def load_sitk(path: Union[Path, str], **kwargs) -> sitk.Image:
    """
    Functional interface to load image with sitk

    Args:
        path: path to file to load

    Returns:
        sitk.Image: loaded sitk image
    """
    return sitk.ReadImage(str(path), **kwargs)


def load_sitk_as_array(path: Union[Path, str], **kwargs) -> Tuple[np.ndarray, dict]:
    """
    Functional interface to load sitk image and convert it to an array

    Args:
        path: path to file to load

    Returns:
        np.ndarray: loaded image data
        dict: loaded meta data
    """
    img_itk = load_sitk(path, **kwargs)
    meta = {key: img_itk.GetMetaData(key) for key in img_itk.GetMetaDataKeys()}
    return sitk.GetArrayFromImage(img_itk), meta




class seg_dataset:
    def __init__(self, fold) -> None:
        self.fold=fold / 'crop_data'
        self.csc_dir = fold / 'test.csv'
        self.df = pd.read_csv(self.csc_dir)
        self.data_list = sorted(self.fold.rglob('*.nii.gz'))
        self.data_num = len(self.data_list)
        self.shape = (64, 64, 32)

    def get_batch(self, batch = 1, index = 0):
        image, _ = load_sitk_as_array(self.data_list[index])
        image = image.transpose(1, 2, 0)
        image = transform.resize(image, (64, 64, 32))
        image = np.expand_dims(image, 0)
        image = np.expand_dims(image, -1)
        return(image)

    def get_label(self, index = 0):
        #计算seg对应label
        listindex = int(str(self.data_list[index]).split('_')[-2])
        #print(listindex)
        out = np.expand_dims([self.df.iloc[listindex]['label']], 0)
        #print(out)

        return(out)

        #         #计算seg对应label
        # listindex = int(str(self.data_list[0]).split('_')[3])
        # out = [0,0]
        # out[self.df.iloc[listindex]['label']] = 1
        # out = np.expand_dims(out, 0)
        # return(out)

    def get_label_list(self):
        labellist = []
        
        for i in range(self.data_num):
            listindex = int(str(self.data_list[i]).split('_')[3])

            labellist.append(self.df.iloc[listindex]['label'])

        return(labellist)


class RandomSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, data_source, replacement=False, num_samples=None, s1 = None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.s1 = s1

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(self.s1)

    def __len__(self):
        return self.num_samples

class seg_dataset_mult:
    def __init__(self, fold, modoules) -> None:
        self.fold = []
        self.data_list = []
        for i in range(len(modoules)):
            temp = fold / 'crop_data' 
            self.fold.append(temp/modoules[i] )
            self.data_list.append(sorted(self.fold[i].rglob('*.nii.gz')))

        self.csc_dir = fold / 'test.csv'
        self.df = pd.read_csv(self.csc_dir)
        self.modoules_num = len(modoules)
        self.data_num = len(self.data_list[0])
        self.shape = (64, 64, 32)

    def get_batch(self, batch = 1, index = 0):
        image, _ = load_sitk_as_array(self.data_list[0][index])
        image = image.transpose(1, 2, 0)
        image = transform.resize(image, (64, 64, 32))
        image = np.expand_dims(image, 0)
        image = np.expand_dims(image, -1)
        return(image)

    def get_label(self, index = 0):
        #计算seg对应label
        listindex = int(str(self.data_list[0][index]).split('_')[-2])
        #print(listindex)
        out = np.expand_dims([self.df.iloc[listindex]['label']], 0)
        #print(out)

        return(out)

        #         #计算seg对应label
        # listindex = int(str(self.data_list[0]).split('_')[3])
        # out = [0,0]
        # out[self.df.iloc[listindex]['label']] = 1
        # out = np.expand_dims(out, 0)
        # return(out)

    def get_label_list(self):
        labellist = []
        
        for i in range(self.data_num):
            listindex = int(str(self.data_list[0][i]).split('_')[3].split('.')[0])

            labellist.append(self.df.iloc[listindex]['label'])

        return(labellist)
