from dataclasses import dataclass
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import torch
from skimage.measure import block_reduce
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from data_utils.anatomical_map import anatomical_map
from img_utils.zst_utils import load_zst16
from img_utils.img_process import process_image
import time

def patchify_3d(volume, patch_size):
    D, H, W = volume.shape
    pD, pH, pW = patch_size

    assert (
        D % pD == 0 and H % pH == 0 and W % pW == 0
    ), "Volume must be divisible by patch size"

    nD, nH, nW = D // pD, H // pH, W // pW

    # Reshape and reorder axes to gather patches
    patches = volume.reshape(nD, pD, nH, pH, nW, pW)
    patches = patches.transpose(0, 2, 4, 1, 3, 5)  # → (nD, nH, nW, pD, pH, pW)
    patches = patches.reshape(-1, pD * pH * pW)  # → (N, patch_volume)

    return patches

def apply_brain_window_level(data):
    lower_bound = 0 #window_center - window_width / 2
    upper_bound = 80 # window_center + window_width / 2

    # Apply window leveling
    window_leveled_data = np.clip(data, lower_bound, upper_bound)

    # Normalize to the range 0 to 1 (optional, can adjust based on output requirement)
    #window_leveled_data = (window_leveled_data - lower_bound) / window_width
    return window_leveled_data
def apply_st_window_level(data):
    lower_bound = -150 #window_center - window_width / 2
    upper_bound = 250 # window_center + window_width / 2

    # Apply window leveling
    window_leveled_data = np.clip(data, lower_bound, upper_bound)

    # Normalize to the range 0 to 1 (optional, can adjust based on output requirement)
    #window_leveled_data = (window_leveled_data - lower_bound) / window_width
    return window_leveled_data

def apply_lung_window_level(data):
    window_center= -600
    window_width = 1500
    lower_bound = -1350.0 #window_center - window_width / 2
    upper_bound = 150.0 # window_center + window_width / 2

    # Apply window leveling
    window_leveled_data = np.clip(data, lower_bound, upper_bound)

    # Normalize to the range 0 to 1 (optional, can adjust based on output requirement)
    #window_leveled_data = (window_leveled_data - lower_bound) / window_width
    return window_leveled_data
    
def apply_abd_window_level(data):
    window_center= -600
    window_width = 1500
    lower_bound = -160.0 #window_center - window_width / 2
    upper_bound = 240.0 # window_center + window_width / 2

    # Apply window leveling
    window_leveled_data = np.clip(data, lower_bound, upper_bound)

    # Normalize to the range 0 to 1 (optional, can adjust based on output requirement)
    #window_leveled_data = (window_leveled_data - lower_bound) / window_width
    return window_leveled_data
def unpatchify_3d(patches, volume_shape, patch_size):
    D, H, W = volume_shape
    pD, pH, pW = patch_size

    assert (
        D % pD == 0 and H % pH == 0 and W % pW == 0
    ), "Volume must be divisible by patch size"

    nD, nH, nW = D // pD, H // pH, W // pW
    N = nD * nH * nW
    assert patches.shape == (
        N,
        pD * pH * pW,
    ), "Patches shape doesn't match expected size"

    # Reshape to 6D
    patches = patches.reshape(nD, nH, nW, pD, pH, pW)
    patches = patches.transpose(0, 3, 1, 4, 2, 5)  # → (nD, pD, nH, pH, nW, pW)
    volume = patches.reshape(D, H, W)

    return volume


@dataclass
class VQABinaryDataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer
    with_label: bool

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        #start = time.time()
        result = VQAMaskDataset._get_inputs_from_batch(
            batch=batch,
            tokenizer=self.tokenizer,
            with_label=self.with_label,
        )
        #end = time.time()
        #print(f"Data collator time: {end - start:.4f}s")
        return result


class VQAMaskDataset(Dataset):
    def __init__(
        self,
        text_list: List[str],
        img_files: List[str],
        seg_files: List[str],
        organs: List[str],
        tokenizer: PreTrainedTokenizer,
        labels: Optional[List[int]] = None,
        transform=None,
        transform_imgonly=None,
    ):

        if img_files is not None:
            assert len(text_list) == len(img_files)
            self.img_files = img_files
        if seg_files is not None:
            assert len(text_list) == len(seg_files)
            self.seg_files = seg_files
        if labels is not None:
            assert len(text_list) == len(labels)
        self.organs = organs
        self.labels = labels
        self.text_list = text_list

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.transform = transform
        self.transform_imgonly = transform_imgonly
        self.tsseg_map = anatomical_map
        self.mask_pool_size = (8, 8, 8)
        self.patch_size = (4, 4, 4)

    def __len__(self):
        return len(self.text_list)

    def read_img(self, img_file):
        if img_file.endswith((".nii", ".nii.gz")):
            img = nib.load(img_file).get_fdata(dtype=np.float32)
            img = np.clip(img, -1000, 1000)
            return img
        elif img_file.endswith(".bin"):
            img = load_zst16(img_file).astype("float32")
        elif img_file.endswith(".npz"):
            img = np.load(img_file)['arr']
            #img = np.float16(img)
            #img = np.clip(img, -1000, 1000)
            #img = apply_abd_window_level(img)
            return img
        else:
            raise ValueError(
                f"Unsupported file type: {img_file}. Only .nii, .nii.gz, and .bin are supported."
            )
        img = img / 2000.0 + 0.5
        return img

    def get_img(self, img_file: str,seg_file: str, transform: None):
        img = self.read_img(img_file)
        
        
        img = process_image(img,None, False)
        
        img = np.float16(img) / 255.0 
        
        if transform is not None:
            img = np.expand_dims(img, 0)
            return transform(img)
        img = torch.tensor(img).unsqueeze(0)
        return img

    def get_img_mask(self, img_file: str, seg_file: str, org_name:str, transform: None, seg_indices):
        img = self.read_img(img_file)
        if seg_file.endswith((".nii", ".nii.gz")):
            seg = nib.load(seg_file).get_fdata(dtype=np.float32).astype("int")
        elif seg_file.endswith(".npz"):
            seg = np.load(seg_file)['arr'].astype("uint8")
        
        #img, seg, coordinate_list = process_image(img, seg, True)
        #img = apply_lung_window_level(img)
        
        img = np.clip(img, -1000, 1000)
        
        img = (img - img.min()) / (img.max()-img.min())
        img = np.float16(img) 
        
        mask = np.isin(seg, seg_indices).astype("float16")

        if transform is not None:
            img_mask = np.stack([img, mask], 0)
            img_mask = transform(img_mask)
            return img_mask[[0]], img_mask[[1]]

        img = torch.tensor(img).unsqueeze(0)  # [1, 256, 256, 128]
        mask = torch.tensor(mask).unsqueeze(0)  # [1, 256, 256, 128]
        return img, mask

    def get_seg_info(self, img_file: str, text: str):
        #seg_tag = text.split(":")[1]
        seg_tag = text.lower()
        
        # TODO: simplify
        if seg_tag == "Nodule":
            text = "Segment pulmonary nodules in the image."
            seg_indices = [1]
        else:
            seg_indices = self.tsseg_map[seg_tag]
            #text = f"Segment {seg_tag} in the image."
        return seg_indices

    def __getitem__(self, idx: int):
        text = self.text_list[idx]

        # TODO: simplify

        if self.img_files is not None:
            img_file = self.img_files[idx]
            seg_file = self.seg_files[idx]
            org1 = self.organs[idx]
            seg_indices = self.get_seg_info(img_file, org1)
            img, mask = self.get_img_mask(
                img_file,
                seg_file,
                org1,
                transform=self.transform,
                seg_indices=seg_indices,
            )
            if self.transform_imgonly is not None:
                img = self.transform_imgonly(img)

        if self.labels is not None:
            #print('seg shape',mask[0].numpy().shape)
            #print('img shape', img.shape)
            mask_label = block_reduce(
                mask[0].numpy(), block_size=self.mask_pool_size, func=np.max
            )
            patchify_label = patchify_3d(
                mask_label, self.patch_size
            )  # (nDxnHxnW, pDxpHxpW)
            #print('patchify_label size', patchify_label.shape)
           
        
        content = {"text": text, "image": img}
        if self.labels is not None:
            content["label"] = self.labels[idx]
            content["seg_label"] = patchify_label
        return content

    def collate_fn(self, batch: List[Dict]):
        return self._get_inputs_from_batch(
            batch=batch,
            tokenizer=self.tokenizer,
            with_label=(self.labels is not None),
            load_image=(self.img_files is not None),
        )

    @staticmethod
    def _get_inputs_from_batch(
        batch: List[Dict[str, torch.Tensor]],
        tokenizer: PreTrainedTokenizer,
        with_label: bool = True,
        load_image: bool = True,
    ):
        sentences = [item["text"] for item in batch]

        inputs = tokenizer(sentences, padding=True, return_tensors="pt")

        if load_image:
            images = [item["image"] for item in batch]
            inputs["images"] = torch.stack(images)

        if with_label:
            labels = np.array([item["label"] for item in batch], dtype=np.float16)
            inputs["labels"] = torch.from_numpy(labels[:, None])

            seg_labels = np.stack([item["seg_label"] for item in batch]).astype(
                np.float16
            )
            inputs["seg_label"] = torch.from_numpy(seg_labels)

            
        return inputs



