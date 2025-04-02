import torch
import openslide
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import h5py
import vahadane

# Define mean and standard deviation for image normalization
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform_patch = transforms.Compose(
                    [# Apply transformations including normalization
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                    ]
                )

class Roi_Seg_Dataset(Dataset):
    """
    Dataset for extracting ROI (Region of Interest) patches from whole-slide images.
    
    Args:
        embed_type (str): Type of pre-trained model. Choose from 'ImageNet', 'RetCCL', 'ctranspath', 'simclr-ciga'.
        file_path (str): Path to the HDF5 file containing coordinates of extracted patches.
        slide_path (str): Path to the whole-slide image (WSI).
        wsi (OpenSlide object): OpenSlide object for reading regions from the WSI.
        levels (list): List of magnification levels at which features are extracted (0:20x,1:10x,2:5x).
        target_roi_size (int): The size of the ROI at 20x magnification.
        patch_size (int): Size of the patches used for feature extraction.
        is_stain_norm (bool): Whether to perform stain normalization.
        resize (bool): Whether to resize patches to 224x224 pixels.
    """
    
    def __init__(self, embed_type='ImageNet', file_path='', slide_path='', wsi=None, levels=[0,1,2],
                 target_roi_size=2048, patch_size=256, is_stain_norm=False, resize=False):
        self.file_path = file_path
        self.wsi = wsi
        self.levels = levels
        self.patch_size = patch_size
        self.slide_path = slide_path
        self.target_roi_size = target_roi_size
        self.downscale = 0
        self.resize = resize

        # Adjust mean and std for specific embedding types
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if embed_type == 'simclr-ciga':
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)

        self.transform_patch = transforms.Compose(
                    [# Apply normalization and tensor transformation
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                    ]
                )

        with h5py.File(self.file_path, 'r') as f:
            dset = f['coords']
            self.roi_level = f['coords'].attrs['patch_level']
            self.roi_size = f['coords'].attrs['patch_size']
            self.downscale = int(self.roi_size / self.target_roi_size)
            self.length = len(dset)
            patch_num_0 = (self.target_roi_size / self.patch_size) ** 2
            self.patch_nums = [int(patch_num_0 / (2 ** level)) for level in self.levels]
        
        # Select target image for stain normalization
        if target_roi_size == 512:
            target_image_dir = 'target_images/target_image_6e3_512.jpg'
        elif target_roi_size == 1024:
            target_image_dir = 'target_images/target_image_6e3_1024.jpg'
        else:
            target_image_dir = 'target_images/target_roi_6e3.jpg'

        if is_stain_norm:
            self.target_img = np.array(Image.open(target_image_dir))
            self.vhd = vahadane.vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=0, ITER=100)
            self.Wt, self.Ht = self.vhd.stain_separate(self.target_img)
        self.is_stain_norm = is_stain_norm

    def __len__(self):
        return self.length

    def stain_norm(self, src_img):
        """
        Perform stain normalization on the input image.
        
        Args:
            src_img (numpy array): Source image in shape (H, W, 3).
        
        Returns:
            img (numpy array): Stain-normalized image.
            flag (bool): Whether normalization was successfully applied.
        """
        std = np.std(src_img[:, :, 0].reshape(-1))
        if std < 5:
            return src_img, False
        else:
            Ws, Hs = self.vhd.stain_separate(src_img)
            img = self.vhd.SPCN(src_img, Ws, Hs, self.Wt, self.Ht)
        return img, True

    def __getitem__(self, idx):
        """
        Retrieve an image patch from the dataset.
        
        Args:
            idx (int): Index of the patch to retrieve.
        
        Returns:
            img_batch (Tensor): Batch of image patches.
            coord (tuple): Coordinates of the patch.
            available (Tensor): Availability status of the patch.
        """
        with h5py.File(self.file_path, 'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]

        try:
            img = self.wsi.read_region(coord, self.roi_level, (self.roi_size, self.roi_size)).convert('RGB')
        except:
            self.wsi = openslide.open_slide(self.slide_path)
            available = False
        else:
            available = True

        patch_num_all = np.sum(self.patch_nums)
        if not available:
            if self.resize:
                img_batch = torch.zeros((patch_num_all, 3, 224, 224))
            else:
                img_batch = torch.zeros((patch_num_all, 3, self.patch_size, self.patch_size))
        else:
            img_batch = []
            img_roi = img.resize((self.target_roi_size, self.target_roi_size))

            if self.is_stain_norm:
                img_roi, flag = self.stain_norm(np.array(img_roi))
            
            if not flag:
                img_roi = torch.zeros((patch_num_all, 3, self.patch_size, self.patch_size))
                available = False
            else:
                for level in self.levels:
                    roi_size_cur = int(self.target_roi_size / (2 ** level))
                    img_cur = Image.fromarray(np.array(img_roi)).resize((roi_size_cur, roi_size_cur))
                    imgarray = np.array(img_cur)
                    for i in range(0, roi_size_cur, self.patch_size):
                        for j in range(0, roi_size_cur, self.patch_size):
                            img_patch = imgarray[i:i+self.patch_size, j:j+self.patch_size, :]
                            if self.resize:
                                img_patch = Image.fromarray(img_patch).resize((224, 224))
                            img_patch = self.transform_patch(np.array(img_patch))
                            img_batch.append(img_patch)
                img_batch = torch.stack(img_batch) if available else img_roi

        return img_batch, coord, torch.tensor([available])



class Patch_Seg_Dataset(Dataset):
    """
    Basic Dataset for Patch Segmentation
    
    Args:
        file_path (str): Path to the .h5 file containing coordinates of patches.
        slide_path (str): Path to the whole slide image (WSI).
        wsi (OpenSlide object): OpenSlide object for reading regions from WSI.
        patch_size (int, optional): Size of the extracted patches. Default is 256.
        is_stain_norm (bool, optional): Whether to apply stain normalization. Default is False.
    """
    
    def __init__(self, file_path, slide_path, wsi, patch_size=256, is_stain_norm=False):
        self.file_path = file_path
        self.wsi = wsi
        self.target_patch_size = patch_size
        self.slide_path = slide_path
        self.downscale = 0

        with h5py.File(self.file_path, 'r') as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.downscale = int(self.patch_size / self.target_patch_size)
            self.length = len(dset)

        print(self.patch_size)

        # Define target image directory based on patch size
        if self.patch_size == 256:
            target_image_dir = 'target_images/target_image_6e3_256.jpg'
        elif self.patch_size == 512:
            target_image_dir = 'target_images/target_image_6e3_512.jpg'
        elif self.patch_size == 1024:
            target_image_dir = 'target_images/target_image_6e3_1024.jpg'

        # Perform stain normalization if required
        if is_stain_norm:
            self.target_img = np.array(Image.open(target_image_dir))
            self.vhd = vahadane.vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=0, ITER=100)
            self.Wt, self.Ht = self.vhd.stain_separate(self.target_img)
            # self.vhd.fast_mode = 1  # Enable fast mode for stain separation
        
        self.is_stain_norm = is_stain_norm

    def __len__(self):
        """
        Get the total number of patches.
        """
        return self.length
    
    def stain_norm(self, src_img):
        """
        Perform stain normalization on the input image.
        
        Args:
            src_img (numpy array): Input patch image.
        
        Returns:
            tuple: Normalized image and a boolean indicating whether normalization was applied.
        """
        std = np.std(src_img[:, :, 0].reshape(-1))
        if std < 10:
            return src_img, False
        else:
            Ws, Hs = self.vhd.stain_separate(src_img)
            img = self.vhd.SPCN(src_img, Ws, Hs, self.Wt, self.Ht)
            return img, True

    def __getitem__(self, idx):
        """
        Retrieve a single patch based on the index.
        
        Args:
            idx (int): Index of the patch to retrieve.
        
        Returns:
            tuple: Processed patch tensor, patch coordinates, and availability flag.
        """
        with h5py.File(self.file_path, 'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]
        
        # Attempt to read the image region from the WSI
        try:
            img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
            available = True
        except:
            # Reopen WSI in case of an error
            self.wsi = openslide.open_slide(self.slide_path)
            available = False

        # Handle unavailable image case
        if not available:
            img_patch = torch.ones((1, 3, self.target_patch_size, self.target_patch_size))
        else:
            img_patch = img.resize((self.target_patch_size, self.target_patch_size))

            flag = True
            if self.is_stain_norm:
                img_patch, flag = self.stain_norm(np.array(img_patch))

            if not flag:
                img_patch = torch.ones((1, 3, self.target_patch_size, self.target_patch_size))
                available = False
            else:
                # Transform patch for model input
                img_patch = transform_patch(img_patch)
                img_patch = img_patch.unsqueeze(0)

        return img_patch, coord, torch.tensor([available])

