# main.py
import argparse
import os
import logging
import glob
import random
import SimpleITK as sitk
from utils.tumors_gen import SynthesisTumor, save_results

def setup_logging(output_dir):
    """配置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'processing.log')),
            logging.StreamHandler()
        ]
    )

def validate_paths(input_dir, output_dir):
    """验证输入输出路径有效性"""
    if not os.path.exists(input_dir):
        raise ValueError(f"输入路径不存在: {input_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"输出路径不可写: {output_dir}")

def process_single_case(image_path, label_path, output_dir):
    """处理单个病例"""
    try:
        # 读取原始数据
        image = sitk.ReadImage(image_path)
        label = sitk.ReadImage(label_path)

        # 转换为numpy数组并调整维度顺序
        image_array = sitk.GetArrayFromImage(image).transpose(2,1,0)
        label_array = sitk.GetArrayFromImage(label).transpose(2,1,0)

        # 生成肿瘤
        tumor_type = random.choices(
            ['small', 'medium', 'large'], 
            weights=[0.12, 0.67, 0.21]
        )[0]
        
        # 合成肿瘤（修改SynthesisTumor函数使其不需要texture参数）
        final_volume, final_mask, _ = SynthesisTumor(
            image_array, 
            label_array,
            tumor_type
        )

        # 保存结果
        case_id = os.path.basename(image_path).split('.')[0]
        output_folder = os.path.join(output_dir, case_id)
        save_results(
            final_volume,
            final_mask,
            final_mask,  # 这里使用mask作为geo_mask示例
            output_folder,
            case_id
        )
        
        return True
    except Exception as e:
        logging.error(f"处理病例失败: {image_path}\n错误信息: {str(e)}")
        return False

def main(input_dir, output_dir):
    """主处理流程"""
    # 初始化设置
    setup_logging(output_dir)
    validate_paths(input_dir, output_dir)
    
    # 获取数据路径
    image_paths = sorted(glob.glob(os.path.join(input_dir, 'imagesTr', '*.nii.gz')))
    label_paths = sorted(glob.glob(os.path.join(input_dir, 'labelsTr', '*.nii.gz')))
    
    # 检查数据一致性
    if len(image_paths) != len(label_paths):
        raise RuntimeError("图像与标签数量不匹配")

    # 处理所有病例
    success_count = 0
    for img_path, lbl_path in zip(image_paths, label_paths):
        logging.info(f"正在处理: {os.path.basename(img_path)}")
        if process_single_case(img_path, lbl_path, output_dir):
            success_count += 1
    
    # 生成统计报告
    logging.info("\n处理完成!")
    logging.info(f"成功处理病例数: {success_count}/{len(image_paths)}")
    logging.info(f"输出目录: {output_dir}")

if __name__ == "__main__":
    # 配置命令行参数
    parser = argparse.ArgumentParser(description='肝脏肿瘤数据生成工具')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='输入数据集路径，需包含imagesTr和labelsTr目录')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='输出结果路径')
    
    args = parser.parse_args()
    
    # 检查目录结构
    required_dirs = ['imagesTr', 'labelsTr']
    for d in required_dirs:
        if not os.path.exists(os.path.join(args.input, d)):
            raise ValueError(f"输入目录缺少必要子目录: {d}")

    # 运行主程序
    main(args.input, args.output)