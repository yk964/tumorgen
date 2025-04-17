import numpy as np
import nibabel as nib
import os
import glob

dir = "/data/yike/nnunet/Dataset/nnUNet_raw/nnUNet_raw_data/Task215_regiontumor/"
sub_dir = ["imagesTr","imagesTs","labelsTr"]
for sub in sub_dir:
    print(dir+sub)
dst_dir = dir + "labelsTr"
file_list = glob.glob(dst_dir+"/*.nii.gz")
for file in file_list:
    img = nib.load(file)
    head = img.header
    affine = img.affine
    print(file)
    data=img.get_fdata()
    print(np.unique(data))
    data[data > 2] = 0
    new_data = np.round(data).astype(int)
    print(np.unique(new_data))
    new_image = nib.Nifti1Image(new_data,affine,head)
    nib.save(new_image,file)