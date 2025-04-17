import numpy as np
import cv2
import random
import elasticdeform
import SimpleITK as sitk
import glob
import os

from resize import resize_image_itk
from scipy.ndimage import gaussian_filter


def random_select(mask_scan):
    # we first find z index and then sample point with z slice
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

    # we need to strict number z's position (0.3 - 0.7 in the middle of liver)
    z = round(random.uniform(0.3, 0.7) * (z_end - z_start)) + z_start

    liver_mask = mask_scan[..., z]

    # erode the mask (we don't want the edge points)
    kernel = np.ones((5,5), dtype=np.uint8)
    liver_mask = cv2.erode(liver_mask, kernel, iterations=1)

    coordinates = np.argwhere(liver_mask == 1)
    random_index = np.random.randint(0, len(coordinates))
    xyz = coordinates[random_index].tolist() # get x,y
    xyz.append(z)
    potential_points = xyz

    return potential_points

def get_ellipsoid(x, y, z):
    """"
    x, y, z is the radius of this ellipsoid in x, y, z direction respectly.
    """
    sh = (4*x, 4*y, 4*z)
    out = np.zeros(sh, int)
    aux = np.zeros(sh)
    radii = np.array([x, y, z])
    com = np.array([2*x, 2*y, 2*z])  # center point

    # calculate the ellipsoid 
    bboxl = np.floor(com-radii).clip(0,None).astype(int)
    bboxh = (np.ceil(com+radii)+1).clip(None, sh).astype(int)
    "生成 Roi区域的 slice对象, 提取出roi区域"
    roi = out[tuple(map(slice,bboxl,bboxh))]
    roiaux = aux[tuple(map(slice,bboxl,bboxh))]

    logrid = *map(np.square,np.ogrid[tuple(
            map(slice,(bboxl-com)/radii,(bboxh-com-1)/radii,1j*(bboxh-bboxl)))]),
    dst = (1-sum(logrid)).clip(0,None)
    mask = dst>roiaux
    roi[mask] = 1
    np.copyto(roiaux,dst,where=mask)

    return out

def get_predefined_texture(mask_shape, sigma_a, sigma_b):
    # uniform noise generate
    a = np.random.uniform(0, 1, size=(mask_shape[0],mask_shape[1],mask_shape[2]))
    a_2 = gaussian_filter(a, sigma=sigma_a)
    scale = np.random.uniform(0.19, 0.21)
    base = np.random.uniform(0.04, 0.06)
    a =  scale * (a_2 - np.min(a_2)) / (np.max(a_2) - np.min(a_2)) + base

    # sample once
    random_sample = np.random.uniform(0, 1, size=(mask_shape[0],mask_shape[1],mask_shape[2]))
    b = (a > random_sample).astype(float)  # int type can't do Gaussian filter
    b = gaussian_filter(b, sigma_b)

    # Scaling and clipping
    u_0 = np.random.uniform(0.5, 0.55)
    threshold_mask = b > 0.12    # this is for calculte the mean_0.2(b2)
    beta = u_0 / (np.sum(b * threshold_mask) / threshold_mask.sum())
    Bj = np.clip(beta*b, 0, 1) # 目前是0-1区间

    return Bj

# Step 1: Random select (numbers) location for tumor.
def random_select(mask_scan):
    # we first find z index and then sample point with z slice
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

    # we need to strict number z's position (0.3 - 0.7 in the middle of liver)
    z = round(random.uniform(0.3, 0.7) * (z_end - z_start)) + z_start

    liver_mask = mask_scan[..., z]

    # erode the mask (we don't want the edge points)
    kernel = np.ones((5,5), dtype=np.uint8)
    liver_mask = cv2.erode(liver_mask, kernel, iterations=1)

    coordinates = np.argwhere(liver_mask == 1)
    random_index = np.random.randint(0, len(coordinates))
    xyz = coordinates[random_index].tolist() # get x,y
    xyz.append(z)
    potential_points = xyz

    return potential_points

# Step 2 : generate the ellipsoid
def get_ellipsoid(x, y, z):
    """"
    x, y, z is the radius of this ellipsoid in x, y, z direction respectly.
    """
    sh = (4*x, 4*y, 4*z)
    out = np.zeros(sh, int)
    aux = np.zeros(sh)
    radii = np.array([x, y, z])
    com = np.array([2*x, 2*y, 2*z])  # center point

    # calculate the ellipsoid 
    bboxl = np.floor(com-radii).clip(0,None).astype(int)
    bboxh = (np.ceil(com+radii)+1).clip(None, sh).astype(int)
    roi = out[tuple(map(slice,bboxl,bboxh))]
    roiaux = aux[tuple(map(slice,bboxl,bboxh))]
    logrid = *map(np.square,np.ogrid[tuple(
            map(slice,(bboxl-com)/radii,(bboxh-com-1)/radii,1j*(bboxh-bboxl)))]),
    dst = (1-sum(logrid)).clip(0,None)
    mask = dst>roiaux
    roi[mask] = 1
    np.copyto(roiaux,dst,where=mask)

    return out

def get_fixed_geo(mask_scan, tumor_type):

    enlarge_x, enlarge_y, enlarge_z = 160, 160, 160
    geo_mask = np.zeros((mask_scan.shape[0] + enlarge_x, mask_scan.shape[1] + enlarge_y, mask_scan.shape[2] + enlarge_z), dtype=np.int8)
    # texture_map = np.zeros((mask_scan.shape[0] + enlarge_x, mask_scan.shape[1] + enlarge_y, mask_scan.shape[2] + enlarge_z), dtype=np.float16)
    small_radius, medium_radius, large_radius = 6, 16, 26


    if tumor_type == 'small':
        num_tumor = random.randint(3,10)
        for _ in range(num_tumor):
            # Small tumor
            x = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            y = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            z = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            sigma = random.randint(1, 2)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

    if tumor_type == 'medium':
        num_tumor = random.randint(2, 5)
        for _ in range(num_tumor):
            # medium tumor
            x = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            y = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            z = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            sigma = random.randint(3, 6)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste medium tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

    if tumor_type == 'large':
        num_tumor = random.randint(1,3)
        for _ in range(num_tumor):
            # Large tumor
            x = random.randint(int(0.75*large_radius), int(1.25*large_radius))
            y = random.randint(int(0.75*large_radius), int(1.25*large_radius))
            z = random.randint(int(0.75*large_radius), int(1.25*large_radius))
            sigma = random.randint(5, 10)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture


    geo_mask = geo_mask[enlarge_x//2:-enlarge_x//2, enlarge_y//2:-enlarge_y//2, enlarge_z//2:-enlarge_z//2]
    # texture_map = texture_map[enlarge_x//2:-enlarge_x//2, enlarge_y//2:-enlarge_y//2, enlarge_z//2:-enlarge_z//2]
    geo_mask = (geo_mask * mask_scan) >= 1
    
    return geo_mask


def get_tumor(volume_scan, mask_scan, tumor_type, texture):
    geo_mask = get_fixed_geo(mask_scan, tumor_type)

    sigma      = np.random.uniform(1, 2)
    difference = np.random.uniform(65, 145)    #65, 145

    # blur the boundary
    geo_blur = gaussian_filter(geo_mask*1.0, sigma)
    abnormally = (volume_scan - texture * geo_blur * difference) * mask_scan
    # abnormally = (volume_scan - texture * geo_mask * difference) * mask_scan
    
    abnormally_full = volume_scan * (1 - mask_scan) + abnormally
    abnormally_mask = mask_scan + geo_mask

    return abnormally_full, abnormally_mask, geo_mask

# def SynthesisTumor(volume_scan, mask_scan, tumor_type, texture):
#     # for speed_generate_tumor, we only send the liver part into the generate program
#     x_start, x_end = np.where(np.any(mask_scan, axis=(1, 2)))[0][[0, -1]]
#     y_start, y_end = np.where(np.any(mask_scan, axis=(0, 2)))[0][[0, -1]]
#     z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

#     # shrink the boundary
#     x_start, x_end = max(0, x_start+1), min(mask_scan.shape[0], x_end-1)
#     y_start, y_end = max(0, y_start+1), min(mask_scan.shape[1], y_end-1)
#     z_start, z_end = max(0, z_start+1), min(mask_scan.shape[2], z_end-1)

#     liver_volume = volume_scan[x_start:x_end, y_start:y_end, z_start:z_end]
#     liver_mask   = mask_scan[x_start:x_end, y_start:y_end, z_start:z_end]

#     # input texture shape: 420, 300, 320
#     # we need to cut it into the shape of liver_mask
#     # for examples, the liver_mask.shape = 286, 173, 46; we should change the texture shape
#     x_length, y_length, z_length = x_end - x_start, y_end - y_start, z_end - z_start

#     print(x_length, y_length, z_length)

#     start_x = random.randint(0, texture.shape[0] - x_length - 1) # random select the start point, -1 is to avoid boundary check
#     start_y = random.randint(0, texture.shape[1] - y_length - 1) 
#     start_z = random.randint(0, texture.shape[2] - z_length - 1) 
#     cut_texture = texture[start_x:start_x+x_length, start_y:start_y+y_length, start_z:start_z+z_length]


#     liver_volume, liver_mask, geo_mask = get_tumor(liver_volume, liver_mask, tumor_type, cut_texture)
#     volume_scan[x_start:x_end, y_start:y_end, z_start:z_end] = liver_volume
#     mask_scan[x_start:x_end, y_start:y_end, z_start:z_end] = liver_mask

#     return volume_scan, mask_scan, geo_mask

def SynthesisTumor(volume_scan, mask_scan, tumor_type, texture):
    """
    综合生成肿瘤并将结果保存到文件。

    Parameters:
    - volume_scan (numpy.ndarray): 原始 CT 扫描数据。
    - mask_scan (numpy.ndarray): 原始肝脏掩码。
    - tumor_type (str): 肿瘤类型 ('small', 'medium', 'large')。
    - texture (numpy.ndarray): 仿真生成的纹理数据。

    Returns:
    - volume_scan: 带肿瘤的最终 CT 扫描。
    - mask_scan: 最终的掩码（肝脏 + 肿瘤）。
    - geo_mask: 肿瘤的几何形状掩码。
    """
    # 步骤 1：获取肝脏所在的体积范围
    x_start, x_end = np.where(np.any(mask_scan, axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(mask_scan, axis=(0, 2)))[0][[0, -1]]
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

    # 缩小边界范围以便后续操作
    x_start, x_end = max(0, x_start+1), min(mask_scan.shape[0], x_end-1)
    y_start, y_end = max(0, y_start+1), min(mask_scan.shape[1], y_end-1)
    z_start, z_end = max(0, z_start+1), min(mask_scan.shape[2], z_end-1)

    liver_volume = volume_scan[x_start:x_end, y_start:y_end, z_start:z_end]
    liver_mask = mask_scan[x_start:x_end, y_start:y_end, z_start:z_end]

    # 保存步骤 1 的结果
    save_results(
        liver_volume, liver_mask, np.zeros(liver_mask.shape),
        "output_stage_1_liver_region", "liver_region"
    )

    # 步骤 2：从纹理中裁剪到对应大小
    x_length, y_length, z_length = x_end - x_start, y_end - y_start, z_end - z_start
    print("Liver region shape:", x_length, y_length, z_length)

    start_x = random.randint(0, texture.shape[0] - x_length - 1) # 随机选择纹理起点
    start_y = random.randint(0, texture.shape[1] - y_length - 1)
    start_z = random.randint(0, texture.shape[2] - z_length - 1)
    cut_texture = texture[start_x:start_x+x_length, start_y:start_y+y_length, start_z:start_z+z_length]

    # 保存步骤 2 的结果
    save_results(
        np.zeros(liver_mask.shape), liver_mask, cut_texture,
        "output_stage_2_texture_cropped", "cropped_texture"
    )

    # 步骤 3：生成肿瘤
    liver_volume, liver_mask, geo_mask = get_tumor(liver_volume, liver_mask, tumor_type, cut_texture)
    geo_mask[geo_mask] == 1
    # 保存步骤 3的肿瘤生成结果
    save_results(
        liver_volume, liver_mask, geo_mask,
        "output_stage_3_tumor_applied", "tumor_applied"
    )

    # 步骤 4：将生成的肿瘤补回原体积
    volume_scan[x_start:x_end, y_start:y_end, z_start:z_end] = liver_volume
    mask_scan[x_start:x_end, y_start:y_end, z_start:z_end] = liver_mask

    # 保存最终结果（含完整体积）
    save_results(
        volume_scan, mask_scan, geo_mask,
        "output_stage_4_final_result", "final_case"
    )

    return volume_scan, mask_scan, geo_mask

def save_results(volume, mask, geo_mask, folder_path, file_prefix):
    """
    保存SynthesisTumor每个阶段的结果。

    Parameters:
    - volume (numpy.ndarray): CT图像数据。
    - mask (numpy.ndarray): 掩码数据（肝脏+肿瘤）。
    - geo_mask (numpy.ndarray): 生成的肿瘤几何掩码。
    - folder_path (str): 保存结果的文件夹路径。
    - file_prefix (str): 文件名前缀，用于区分不同阶段。
    """
    os.makedirs(folder_path, exist_ok=True)

    # 转换 NumPy 数组为 SimpleITK 图像
    volume_sitk = sitk.GetImageFromArray(volume.transpose(2, 1, 0))
    mask_sitk = sitk.GetImageFromArray(mask.transpose(2, 1, 0))
    geo_mask_sitk = sitk.GetImageFromArray(geo_mask.transpose(2, 1, 0))

    # 保存文件
    sitk.WriteImage(volume_sitk, os.path.join(folder_path, f"{file_prefix}_volume.nii.gz"))
    sitk.WriteImage(mask_sitk, os.path.join(folder_path, f"{file_prefix}_mask.nii.gz"))
    sitk.WriteImage(geo_mask_sitk, os.path.join(folder_path, f"{file_prefix}_geo_mask.nii.gz"))
    print(f"Results saved to {folder_path} with prefix {file_prefix}")



sigma_as = [3, 6, 9, 12, 15]  #3, 6, 9, 12, 15
sigma_bs = [4, 7]  #4, 7

tumor_types = ['small', 'medium', 'large']

splits = ['imagesTr','imagesTs']
# data_path = "data/" 

#填入需要保存的路径
data_path = '/data/yike/nnunet/raw/LITS_free192/'
image_list = glob.glob('/data/yike/nnunet/raw/LITS_free192/imagesTr/*.nii.gz')

for image_path in image_list:

    file_name = os.path.basename(image_path)
    label_path = data_path + 'labelsTr/' + file_name

    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)

    print(image_path)

    imgResampled = resize_image_itk(image, (192, 192, 64), resamplemethod = sitk.sitkLinear) #这里要注意：mask用最近邻插值，CT图像用线性插值
    labelResampled = resize_image_itk(label, (192, 192, 64), resamplemethod = sitk.sitkNearestNeighbor) #这里要注意：mask用最近邻插值，CT图像用线性插值


    if splits == 'imagesTs':

        sitk.WriteImage(imgResampled, data_path+'imagesTs/'+ file_name + image_path[len(image_path):])



    print(file_name,label_path)

    image = sitk.ReadImage(image_path)
    direction = image.GetDirection()


    image_data = sitk.GetArrayFromImage(imgResampled).transpose(2,1,0)
    label_data = sitk.GetArrayFromImage(labelResampled).transpose(2,1,0)


    predefined_texture_shape = image_data.shape
    textures = []

    for sigma_a in sigma_as:
        for sigma_b in sigma_bs:
            texture = get_predefined_texture(predefined_texture_shape, sigma_a, sigma_b)
            textures.append(texture)

    #将肿瘤mask值置为1
    ground_truth = np.zeros(label_data.shape)       
    print(ground_truth.shape)
    ground_truth[label_data == 2 ] = 2
    
    label_data[label_data == 2] = 0
    label_data[label_data == 1] = 1


    print(np.unique(label_data))
    print(image_data.shape)

    tumor_type = np.random.choice(tumor_types, p=[0.12,0.67,0.21])
    texture = random.choice(textures)


    image_new, label_new, geo_mask = SynthesisTumor(image_data, label_data, tumor_type, texture)

    print(tumor_type)

    # print(np.unique(label_new))
    # print(image_new.shape)
    label_new[ground_truth == 2 ] = 2
    image_new = image_new.transpose(2,1,0)
    label_new = label_new.transpose(2,1,0)



    image_new = sitk.GetImageFromArray(image_new)
    label_new= sitk.GetImageFromArray(label_new)


    image_new.SetDirection(direction)
    label_new.SetDirection(direction)


    os.makedirs("Dataset/nnUNet_raw/nnUNet_raw_data/Task999_freetumor3/imagesTr",exist_ok=True)
    os.makedirs("Dataset/nnUNet_raw/nnUNet_raw_data/Task999_freetumor3/labelsTr",exist_ok=True)

    sitk.WriteImage(image_new,"/data/yike/nnunet/Dataset/nnUNet_raw/nnUNet_raw_data/Task999_freetumor3/imagesTr/"+file_name)
    sitk.WriteImage(label_new,"/data/yike/nnunet/Dataset/nnUNet_raw/nnUNet_raw_data/Task999_freetumor3/labelssTr/"+file_name)
    # sitk.WriteImage(label,"/data/yike/nnunet/real_label.nii.gz")
    break
