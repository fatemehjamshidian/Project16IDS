import os
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img
from nilearn.image import new_img_like
from scipy.ndimage import zoom

folder_path = 'F:/team 16/new sort data/New folder/1'

def process_nifti_file(file_path):

    nifti_image = nib.load(file_path)
  
    nifti_data = nifti_image.get_fdata()

    nifti_data = np.squeeze(nifti_data)
 
    brain_mask = nib.load('Attention.nii')


    mask_data = brain_mask.get_fdata()

    if nifti_data.shape != mask_data.shape:

        zoom_factors = [n / m for n, m in zip(nifti_data.shape, mask_data.shape)]

        resampled_mask_data = zoom(mask_data, zoom_factors, order=0) 

        resampled_brain_mask = nib.Nifti1Image(resampled_mask_data, affine=nifti_image.affine)

        nib.save(resampled_brain_mask, 'resampled_brain_mask.nii.gz')

        mask_data = resampled_mask_data

    masked_nifti_data = np.multiply(nifti_data, mask_data)

    masked_nifti_image = nib.Nifti1Image(masked_nifti_data, nifti_image.affine)

    output_file = os.path.join(folder_path, 'masked_attention_' + os.path.basename(file_path))

    nib.save(masked_nifti_image, output_file)

for file_name in os.listdir(folder_path):
    if file_name.endswith('.nii'): 
        file_path = os.path.join(folder_path, file_name)
        process_nifti_file(file_path)


