"""The BraTS2021 dataset divides the tumor regions into GD-enhancing tumor (ET — label 4) and necrotic tumor core (
NCR — label 1). However, these two regions cannot be distinguished before obtaining T1ce. Therefore, we need to merge
these regions when loading the dataset."""

import os
from torch.utils.data import Dataset
import torch
import SimpleITK as sitk
import torchvision.transforms as transforms
import numpy as np


def load_subject_data_norm_tensor(root, modality, brats2020_shuffled_path, start_num, end_num, start_slice, end_slice):
    with open(brats2020_shuffled_path, 'r') as file:
        numbers = [line.strip() for line in file.readlines()]

    datalist = []
    for i in numbers[start_num: end_num]:
        path = os.path.join(root, 'BraTS2021_' + i, 'BraTS2021_' + i + '_' + modality + '.nii.gz')
        img = sitk.GetArrayFromImage(sitk.ReadImage(path))
        img = img[:, 34:226, 40:-40]
        img_max = img.max()
        img_min = img.min()
        for i_slice in range(start_slice, end_slice):
            img_2d = img[i_slice]
            img_2d = (img_2d - img_min) / (img_max - img_min) * 2 - 1
            img_slice = transforms.ToTensor()(img_2d).to(torch.float32)
            img_slice_path = i + '_' + str(i_slice)
            data = [img_slice, img_slice_path]
            datalist.append(data)
    return datalist


#######################################################################################################################
# This function is responsible for splitting the segmentation labels and then returning the edema region labels and
# the merged tumor region labels.
def load_segmentation_and_divide(root, brats2020_shuffled_path, start_num, end_num, start_slice, end_slice):
    with open(brats2020_shuffled_path, 'r') as file:
        numbers = [line.strip() for line in file.readlines()]

        datalist = []
        tmp_tensor = torch.tensor([1, 4])
        for i in numbers[start_num: end_num]:
            path = os.path.join(root, 'BraTS2021_' + i, 'BraTS2021_' + i + '_seg.nii.gz')
            seg = sitk.GetArrayFromImage(sitk.ReadImage(path))
            seg = seg[:, 34:226, 40:-40]
            for i_slice in range(start_slice, end_slice):
                seg_2d = (seg[i_slice]).astype(np.float32)
                seg_2d = torch.from_numpy(seg_2d).unsqueeze(0)
                edema_region = torch.zeros_like(seg_2d)
                tumor_region = torch.zeros_like(seg_2d)
                edema_region[seg_2d == 2] = 1.0  # Edema region, label 2.
                tumor_region[torch.isin(seg_2d, tmp_tensor)] = 1.0  # Merged tumor region, label 1.
                data = [edema_region, tumor_region]
                datalist.append(data)

        return datalist


class LoadData(Dataset):
    def __init__(self, root, modalities, data_list, start_num=0, end_num=None, start_slice=0, end_slice=155):
        super(LoadData, self).__init__()

        self.data = {}
        self.modalities = modalities

        for modality in self.modalities:
            self.data[modality] = load_subject_data_norm_tensor(root, modality, data_list,
                                                                start_num, end_num, start_slice, end_slice)

        self.data['seg'] = load_segmentation_and_divide(root, data_list, start_num, end_num, start_slice, end_slice)

        print('num_Subjects:', len(self.data[self.modalities[0]]))

    def __getitem__(self, index):
        slice_data = {}
        for modality in self.modalities:
            slice_data[modality] = self.data[modality][index]
        slice_data['seg'] = self.data['seg'][index]

        return slice_data  # {'t1':[t1data, t1path], 't2':[t2data, t2path]}

    def __len__(self):
        return len(self.data[self.modalities[0]])
