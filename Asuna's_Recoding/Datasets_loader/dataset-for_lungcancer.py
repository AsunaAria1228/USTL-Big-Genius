import os
import glob
import numpy as np
import pandas as pd
from random import sample
from sklearn.utils import shuffle
from tqdm import tqdm
from PIL import Image
from skimage import io
import torch
from torchvision import transforms

class TCGA_LungCancerDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None, downsample=0.2, preload=False, patch_downsample=1.0, scale=1):
        """
        A dataset for TCGA Lung Cancer data, supporting both train and test splits.
        """
        self.train = train
        self.transform = transform or transforms.Compose([transforms.ToTensor()])
        self.downsample = downsample
        self.patch_downsample = patch_downsample
        self.preload = preload
        self.scale = scale

        # Directories and test IDs
        dir_luad = "/home/qlh/Data3/TCGA_LungCancer/processed_LUAD/pyramid/qlh/"
        dir_lusc = "/home/qlh/Data3/TCGA_LungCancer/processed_LUSC/pyramid/qlh/"
        test_ids_file = "/home/qlh/Data3/TCGA_LungCancer/TEST_ID.csv"

        # Load slides and split into train/test
        all_slides = glob.glob(os.path.join(dir_luad, "*")) + glob.glob(os.path.join(dir_lusc, "*"))
        test_ids = pd.read_csv(test_ids_file)['0'].tolist()
        slides_train, slides_test = self._split_slides(all_slides, test_ids)
        self.slides = slides_train if train else slides_test

        # Downsample slides
        self.slides = self._downsample_slides(self.slides, downsample)
        self.num_slides = len(self.slides)

        # Process patches
        self._process_patches()

    def _split_slides(self, all_slides, test_ids):
        """Split slides into train and test sets based on IDs."""
        slides_train = [slide for slide in all_slides if slide.split('/')[-1] not in test_ids]
        slides_test = [slide for slide in all_slides if slide.split('/')[-1] in test_ids]
        return slides_train, slides_test

    def _downsample_slides(self, slides, downsample_ratio):
        """Randomly downsample the list of slides."""
        np.random.shuffle(slides)
        return slides[:int(len(slides) * downsample_ratio)]

    def _process_patches(self):
        """Process patches for all slides."""
        self.num_patches = 0
        self.all_patches = [] if not self.preload else np.zeros((self.num_patches, 512, 512, 3), dtype=np.uint8)
        self.slide_labels = []
        self.slide_indices = []
        self.slide_names = []

        for slide_idx, slide in tqdm(enumerate(self.slides), desc='Processing patches', ascii=True):
            patches = self._get_patches_from_slide(slide)
            for patch in patches:
                self._add_patch(patch, slide_idx, slide)

        self.num_patches = len(self.all_patches)
        self.slide_labels = np.array(self.slide_labels)
        self.slide_indices = np.array(self.slide_indices)
        self.slide_names = np.array(self.slide_names)
        print(f"[DATA INFO] num_slides: {self.num_slides}, num_patches: {self.num_patches}")

    def _get_patches_from_slide(self, slide):
        """Get patch file paths for a slide."""
        pattern = '*.jpeg' if self.scale == 1 else '*/*.jpeg'
        patches = glob.glob(os.path.join(slide, pattern))
        if self.patch_downsample < 1.0:
            patches = sample(patches, int(len(patches) * self.patch_downsample))
        return patches

    def _add_patch(self, patch_path, slide_idx, slide):
        """Add a patch to the dataset."""
        if self.preload:
            self.all_patches[self.num_patches] = io.imread(patch_path)
        else:
            self.all_patches.append(patch_path)
        label = int('LUSC' in slide.split('/')[-4])
        self.slide_labels.append(label)
        self.slide_indices.append(slide_idx)
        self.slide_names.append(slide.split('/')[-1])

    def __getitem__(self, index):
        """Retrieve a patch and its associated data."""
        if self.preload:
            patch_image = self.all_patches[index]
        else:
            patch_image = io.imread(self.all_patches[index])

        slide_label = self.slide_labels[index]
        slide_index = self.slide_indices[index]
        slide_name = self.slide_names[index]
        patch_image = self.transform(Image.fromarray(np.uint8(patch_image), 'RGB'))
        patch_label = 0  # No patch-level labels available in TCGA
        return patch_image, [patch_label, slide_label, slide_index, slide_name], index

    def __len__(self):
        return self.num_patches

def get_data_loaders(batch_size=64, downsample=0.2, num_workers=0, preload=False):
    """Get data loaders for train and validation datasets."""
    train_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor()
    ])
    train_dataset = TCGA_LungCancerDataset(train=True, transform=train_transforms, downsample=downsample, preload=preload)
    val_dataset = TCGA_LungCancerDataset(train=False, transform=transforms.ToTensor(), downsample=downsample, preload=preload)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                               num_workers=num_workers, drop_last=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                             num_workers=num_workers, drop_last=False, pin_memory=True)
    return train_loader, val_loader

if __name__ == '__main__':
    train_loader, val_loader = get_data_loaders(batch_size=64, downsample=1.0)
    for data in tqdm(train_loader, desc='Loading'):
        patch_images = data[0]
        labels = data[1]
        indices = data[-1]
    print("END")