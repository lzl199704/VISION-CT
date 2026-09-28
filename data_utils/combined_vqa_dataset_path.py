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

def apply_lung_window_level(data):
    window_center= -600
    window_width = 1500
    lower_bound = window_center - window_width / 2
    upper_bound = window_center + window_width / 2

    # Apply window leveling
    window_leveled_data = np.clip(data, lower_bound, upper_bound)

    # Normalize to the range 0 to 1 (optional, can adjust based on output requirement)
    window_leveled_data = (window_leveled_data - lower_bound) / window_width
    return window_leveled_data
    
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
        # self.mask_pool_size = (8, 8, 8) # No longer needed for high-res
        # self.patch_size = (4, 4, 4) # No longer needed

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

    # organs that get the LUNG window; must stay identical to combined_vqa_dataset_semantic.py
    # (the eval dataset) so train-time and eval-time intensities match for chest rows.
    LUNG_WINDOW_ORGANS = ('chest', 'lung', 'heart', 'thoracic aorta')

    def get_img_mask(self, img_file: str, seg_file: str, transform: None, seg_indices, org_name: str = None):
        img = self.read_img(img_file)
        if seg_file.endswith((".nii", ".nii.gz")):
            seg = nib.load(seg_file).get_fdata(dtype=np.float32).astype("int")
        elif seg_file.endswith(".npz"):
            seg = np.load(seg_file)['arr'].astype("uint8")
        elif seg_file=='None':
            seg = np.zeros(img.shape, dtype=np.uint8)
        
        #img, seg, coordinate_list = process_image(img, seg, True)
        #img = apply_lung_window_level(img)
        # Window by the QUERIED ORGAN (not by file path): the old path rule ('chest_noncon' in
        # path) gave Sinai chest volumes the abdominal window while eval applied the lung
        # window. Organ-based selection reproduces the old behaviour for every abdomen path.
        if (org_name or '').lower() in self.LUNG_WINDOW_ORGANS:
            img = apply_lung_window_level(img)
        else:
            img = apply_abd_window_level(img)
        
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
                transform=self.transform,
                seg_indices=seg_indices,
                org_name=org1,
            )
            if self.transform_imgonly is not None:
                img = self.transform_imgonly(img)

        # Skip downsampling/patchifying logic here
        
        content = {"text": text, "image": img}
        if self.labels is not None:
            content["label"] = self.labels[idx]
            # Directly use the full resolution mask
            # Ensure mask is (1, D, H, W) or (D, H, W)
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            
            # If mask has channel dim, good. If not, add one?
            # get_img_mask returns (C, D, H, W) or so.
            # actually get_img_mask returns transformed tensor which likely has channel dim 0
            
            content["seg_label"] = mask 
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

            # Stack 3D volumes. Result: (B, C, D, H, W)
            seg_labels = np.stack([item["seg_label"] for item in batch]).astype(
                np.float16
            )
            inputs["seg_label"] = torch.from_numpy(seg_labels)

            
        return inputs
