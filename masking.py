"""
@author: Fatemeh Jamshidian
"""
import nilearn
from nilearn import image
from nilearn import maskers

func_img=nilearn.image.load_img('dsw01_mci_rs.nii')
mask_img=nilearn.image.load_img('working memory mask.nii')
masker=nilearn.maskers.NiftiMasker(mask_img)

masker.fit(func_img)
masked = masker.transform(func_img)
masked_img=masker.inverse_transform(masked).to_filename('masked_01_mci.nii')
