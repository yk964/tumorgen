import numpy as np
import cv2
import random
import elasticdeform
import SimpleITK as sitk
import glob
import os
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_erosion, binary_dilation, map_coordinates

from skimage.morphology import ball

def segment_vessels(volume, hu_min=150, hu_max=300):
    """根据HU值范围分割血管区域"""
    return (volume >= hu_min) & (volume <= hu_max)

def generate_perlin_noise(shape, scale=20, octaves=3, persistence=0.5, lacunarity=2.0):
    """生成3D Perlin噪声场"""
    import noise
    world = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                world[i][j][k] = noise.pnoise3(i/scale, j/scale, k/scale,
                                              octaves=octaves,
                                              persistence=persistence,
                                              lacunarity=lacunarity,
                                              repeatx=1024,
                                              repeaty=1024,
                                              repeatz=1024)
    return world

def apply_complex_deformation(geo, D):
    """应用复合形变（高斯+Perlin）"""
    # 计算形变参数
    sigma_e = 0.5 + 0.1 * (D/30)**2
    lambda_val = 0.1 * (1 + D/20)
    
    # 高斯形变
    geo = elasticdeform.deform_random_grid(geo, sigma=sigma_e, points=3)
    
    # Perlin噪声形变
    if D >= 10:  # 仅在中晚期应用
        noise_field = generate_perlin_noise(geo.shape) * lambda_val
        x, y, z = np.indices(geo.shape)
        indices = np.stack([x + noise_field, 
                           y + noise_field,
                           z + noise_field], axis=-1)

        geo = map_coordinates(geo, indices.T, order=0, mode='constant')
    
    return geo

def gen_position(mask_scan, vessel_mask, R):
    """带血管碰撞检测的位置选择"""
    while True:
        # 原有选择逻辑
        z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]
        z = round(random.uniform(0.3, 0.7) * (z_end - z_start)) + z_start
        liver_mask = mask_scan[..., z]
        liver_mask = cv2.erode(liver_mask, np.ones((5,5), np.uint8), iterations=1)
        coordinates = np.argwhere(liver_mask == 1)
        if len(coordinates) == 0: continue
        random_index = np.random.randint(0, len(coordinates))
        x, y = coordinates[random_index]
        
        # 碰撞检测
        distance_map = distance_transform_edt(~vessel_mask[..., z])
        if distance_map[x, y] >= R:
            return [x, y, z]


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

def get_predefined_texture(shape, hu_liver):
    """改进的纹理生成（含坏死和脂肪变性）"""
    # 基础纹理
    base = hu_liver + np.random.normal(0, 15, shape)
    
    # 坏死核心（中心区域增加HU）
    center = tuple(d//2 for d in shape)
    dist_map = np.sqrt(sum((np.indices(shape) - np.array(center)[:,None,None,None])**2))
    necrosis_mask = dist_map < min(shape)//3
    base[necrosis_mask] += 30
    
    # 脂肪变性（随机斑块降低HU）
    fat_mask = np.random.rand(*shape) < 0.2
    base[fat_mask] -= np.random.uniform(10,30)
    
    return gaussian_filter(base, sigma=1.5)


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

def get_fixed_geo(mask_scan, tumor_type, vessel_mask=None):
    """改进版肿瘤几何生成函数（整合血管检测和复合形变）"""
    # ================= 医学参数配置 =================
    TUMOR_PARAMS = {
        'small': {
            'radius_range': (4,8), 
            'num': (3,10), 
            'vessel_safe_dist': 3.0
        },
        'medium': {
            'radius_range': (12,20),
            'num': (2,5),
            'vessel_safe_dist': 5.0
        },
        'large': {
            'radius_range': (20,30),
            'num': (1,3),
            'vessel_safe_dist': 8.0
        }
    }

    # ================= 主逻辑 =================
    params = TUMOR_PARAMS[tumor_type]
    enlarge = 160
    geo_mask = np.zeros(tuple(np.array(mask_scan.shape) + enlarge), dtype=np.uint8)
    
    # 生成多个肿瘤
    for _ in range(random.randint(*params['num'])):
        # 随机生成肿瘤尺寸
        rad = [random.randint(*params['radius_range']) for _ in range(3)]
        D = max(rad) * 2  # 计算等效直径
        
        # 生成基础椭球
        geo = get_ellipsoid(*rad)
        
        # 应用复合形变
        geo = apply_complex_deformation(geo, D)
        
        # 安全位置选择（带血管碰撞检测）
        try:
            point = gen_position(mask_scan, vessel_mask, params['vessel_safe_dist'])
        except Exception as e:
            continue
        
        # 位置调整及融合
        new_point = np.array(point) + enlarge//2
        slices = tuple(
            slice(int(new_point[i]-geo.shape[i]//2), 
                  int(new_point[i]+geo.shape[i]//2))
            for i in range(3)
        )
        geo_mask[slices] = np.maximum(geo_mask[slices], geo)

    # ================= 后处理 =================
    geo_mask = geo_mask[
        enlarge//2:-enlarge//2,
        enlarge//2:-enlarge//2,
        enlarge//2:-enlarge//2
    ]
    
    # 确保肿瘤位于肝脏内部（2mm安全边界）
    safe_liver = binary_erosion(mask_scan, ball(2))
    geo_mask = np.logical_and(geo_mask, safe_liver)
    
    return geo_mask.astype(np.uint8)

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

def SynthesisTumor(volume_scan, mask_scan, tumor_type, texture):
    """综合肿瘤生成流程"""
    # 血管分割
    vessel_mask = segment_vessels(volume_scan)
    
    # 生成几何（传递vessel_mask参数）
    geo_mask = get_fixed_geo(mask_scan, tumor_type, vessel_mask)
    
    # 生成病理纹理
    hu_liver = np.mean(volume_scan[mask_scan.astype(bool)])
    texture = get_predefined_texture(volume_scan.shape, hu_liver)
    
    # 边缘增强
    struct = ball(1)
    edge_mask = binary_dilation(geo_mask, struct) ^ binary_erosion(geo_mask, struct)
    enhanced = volume_scan + edge_mask * 25
    
    # 合成最终图像
    final_volume = enhanced * (1 - geo_mask) + texture * geo_mask

    return final_volume, geo_mask, geo_mask


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


