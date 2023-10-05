"Created by Fatemeh Jamshidian"
import numpy as np
import pandas as pd
import nibabel as nib

def GreyLevelOfVoxels(mask_path,nii_path):
   
    mask_img = nib.load(mask_path)
    coordinates_array_ROI = np.transpose(np.nonzero(mask_img.get_fdata()))
    nifti_img = nib.load(nii_path)
    # Get the voxel data 
    voxel_data = nifti_img.get_fdata()
    voxel_coordinates_list_ROI = [tuple(coord) for coord in coordinates_array_ROI]
    voxel_values_ROI = {}
    for coord in voxel_coordinates_list_ROI:
        x, y, z = coord
        voxel_values_ROI[coord] = voxel_data[x, y, z]
    return voxel_values_ROI, voxel_coordinates_list_ROI


def AdjancencyMatrix(voxel_coordinates,voxel_values,data_path):
    img_nii= nib.load(data_path)
    nii_data = img_nii.get_fdata()
    nii_data_non_zero = nii_data[nii_data > 0]
    ########Define threshold
    max_value = np.max(nii_data)
    min_value = np.min(nii_data_non_zero)
    
    # Divide the range into 10 equal parts
    part_width=(max_value-min_value)/9
    
    
    #####Create an empty adjacency matrix
    num_voxels = len(voxel_coordinates)
    adjacency_matrix = np.zeros((num_voxels, num_voxels), dtype=int)
    for i, coord_i in enumerate(voxel_coordinates):
        for j, coord_j in enumerate(voxel_coordinates):
            value_i = voxel_values[coord_i]
            value_j = voxel_values[coord_j]
        
            if int((value_j-min_value)/part_width)==int((value_i-min_value)/part_width):
                adjacency_matrix[i, j] = 1
                
            else:
                adjacency_matrix[i, j] = 0
            
    #####Get adjacency_matrix as dataframe
    adjacency_df = pd.DataFrame(adjacency_matrix, index=voxel_coordinates, columns=voxel_coordinates)
    return adjacency_df

def LabeledMatrix(adjacency_dataframe, voxel_coordinates, label_dict):
    labels = [label_dict.get(coord, coord) for coord in voxel_coordinates]
    
    multi_index = pd.MultiIndex.from_tuples(list(zip(labels, voxel_coordinates)), names=['Label', 'Coordinate'])
    adjacency_dataframe.index = multi_index
    adjacency_dataframe.columns = multi_index
    adjacency_dataframe_labeled= adjacency_dataframe
    
    return adjacency_dataframe_labeled


def Total(list_of_ROI_path,list_of_ROI_name,path):
    ALL_Dict = dict(zip(list_of_ROI_path, list_of_ROI_name))
    ####
    dict_of_ROINumber_coordinate={}
    voxel_values_total={}
    voxel_coordinates_total=[]
    for ROI_path in list_of_ROI_path:
        voxel_values, voxel_coordinates=GreyLevelOfVoxels(ROI_path, path)
        dict_of_ROINumber_coordinate[ROI_path]=voxel_coordinates
        voxel_values_total={**voxel_values_total,**voxel_values}
        voxel_coordinates_total += [coordinate  for coordinate  in voxel_coordinates]
    #####
    labeled_dict = {}
    counter=0
    for Roi in ALL_Dict:
        for coord in voxel_coordinates_total:
            if coord in dict_of_ROINumber_coordinate[Roi]:
                labeled_dict[coord]=ALL_Dict[Roi]

    ########
    #######
    
    ad_mat=AdjancencyMatrix(voxel_coordinates_total, voxel_values_total, path)
    ##########
    label_mat_df=LabeledMatrix(ad_mat,  voxel_coordinates_total, labeled_dict)
    return label_mat_df

