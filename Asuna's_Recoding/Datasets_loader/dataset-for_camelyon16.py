import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import os
import glob
from PIL import Image
from skimage import io
import h5py


def calculate_slide_statistics(slide_paths, pos_region_threshold=0.5):
    """
    Calculate statistics for positive and negative slides.
    """
    pos_patch_count = 0
    total_pos_patches = 0
    neg_patch_count = 0

    for slide_path in tqdm(slide_paths, desc="Calculating slide statistics"):
        if "pos" in os.path.basename(slide_path):
            pos_patches = [
                p for p in glob.glob(os.path.join(slide_path, "*_pos*.jpg"))
                if float(p.split("_")[-1].split(".jpg")[0]) >= pos_region_threshold
            ]
            pos_patch_count += len(pos_patches)
            total_pos_patches += len(glob.glob(os.path.join(slide_path, "*.jpg")))
        else:
            neg_patch_count += len(glob.glob(os.path.join(slide_path, "*.jpg")))

    print(f"Total slides: {len(slide_paths)}")
    print(f"Positive patches in positive slides: {pos_patch_count} / {total_pos_patches}")
    print(f"Negative patches: {neg_patch_count}")
    return pos_patch_count, total_pos_patches, neg_patch_count


class CAMELYON16Dataset(torch.utils.data.Dataset):
    """
    Generalized dataset for CAMELYON16.
    """

    def __init__(self, root_dir, resolution="10x", train=True, transform=None,
                 downsample_ratio=1.0, pos_region_threshold=0.5, return_bag=False):
        self.root_dir = os.path.join(root_dir, "training" if train else "testing")
        self.transform = transform or transforms.Compose([transforms.ToTensor()])
        self.downsample_ratio = downsample_ratio
        self.pos_region_threshold = pos_region_threshold
        self.return_bag = return_bag

        # Load and filter slides
        slides = glob.glob(os.path.join(self.root_dir, "*"))
        pos_slides = [s for s in slides if "pos" in os.path.basename(s)]
        self.filtered_slides = self._filter_positive_slides(pos_slides)
        
        # Downsample slides if required
        if downsample_ratio < 1.0:
            np.random.shuffle(self.filtered_slides)
            self.filtered_slides = self.filtered_slides[:int(len(self.filtered_slides) * downsample_ratio)]

        # Preload patch information
        self.all_patches, self.patch_labels, self.slide_indices = self._preload_patches()

    def _filter_positive_slides(self, pos_slides):
        """
        Remove positive slides without valid positive patches.
        """
        valid_slides = []
        for slide in tqdm(pos_slides, desc="Filtering positive slides"):
            pos_patches = [
                p for p in glob.glob(os.path.join(slide, "*_pos*.jpg"))
                if float(p.split("_")[-1].split(".jpg")[0]) >= self.pos_region_threshold
            ]
            if pos_patches:
                valid_slides.append(slide)
        return valid_slides

    def _preload_patches(self):
        """
        Preload patch paths, labels, and indices for all slides.
        """
        patches = []
        labels = []
        slide_indices = []
        for idx, slide in enumerate(tqdm(self.filtered_slides, desc="Preloading patches")):
            patch_files = glob.glob(os.path.join(slide, "*.jpg"))
            for patch in patch_files:
                patches.append(patch)
                labels.append(int("pos" in patch))
                slide_indices.append(idx)
        return np.array(patches), np.array(labels), np.array(slide_indices)

    def __getitem__(self, index):
        """
        Get item by index.
        """
        patch_path = self.all_patches[index]
        patch_image = io.imread(patch_path)
        patch_label = self.patch_labels[index]

        patch_image = self.transform(Image.fromarray(patch_image))
        return patch_image, patch_label

    def __len__(self):
        return len(self.all_patches)