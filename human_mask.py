#!/usr/bin/env python3
"""
Human Mask Generator
专门用于生成人物黑白掩码的脚本

功能：
- 自动处理指定目录中的所有图像
- 检测和分割人物
- 生成黑白二值掩码
- 保存为同名.jpg文件到输出目录

使用方法：
python human_mask_lightweight.py
"""

import os
import sys
import glob
import numpy as np
import torch
import torchvision
from PIL import Image
import cv2
from tqdm import tqdm
import supervision as sv
import argparse
from pathlib import Path

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'groundingdino'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'segment_anything'))

# Grounding DINO - 使用更简单的inference接口
from groundingdino.util.inference import Model

# Segment Anything
from segment_anything import sam_model_registry, SamPredictor


class HumanMaskGenerator:
    """人物掩码生成器"""
    
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.grounding_dino_model = None
        self.sam_predictor = None
        self._load_models()
    
    def _load_models(self):
        """加载模型"""
        print("正在加载GroundingDINO模型...")
        config_path = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
        checkpoint_path = "models/groundingdino_swint_ogc.pth"
        
        if not os.path.exists(checkpoint_path):
            print(f"错误: GroundingDINO模型文件不存在: {checkpoint_path}")
            sys.exit(1)
        
        # 使用更简单的Model类进行推理
        self.grounding_dino_model = Model(
            model_config_path=config_path,
            model_checkpoint_path=checkpoint_path
        )
        
        print("正在加载SAM模型...")
        sam_checkpoint = "models/sam_vit_h_4b8939.pth"
        
        if not os.path.exists(sam_checkpoint):
            print(f"错误: SAM模型文件不存在: {sam_checkpoint}")
            sys.exit(1)
        
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
        print("模型加载完成!")
    
    def _load_image(self, image_path):
        """加载图像"""
        image_pil = Image.open(image_path).convert("RGB")
        # 直接返回PIL图像，supervision会自动处理
        return image_pil, None
    
    def _detect_humans(self, image_pil, image):
        """检测人物 - 使用与demo相同的方法"""
        # 将PIL图像转换为OpenCV格式用于supervision
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # 使用与demo相同的简单检测方法
        detections = self.grounding_dino_model.predict_with_classes(
            image=image_cv,
            classes=["person"],  # 只使用person，与demo保持一致
            box_threshold=0.25,
            text_threshold=0.25
        )
        
        if len(detections) > 0:
            # 应用NMS去重 - 使用与demo相同的阈值
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy), 
                torch.from_numpy(detections.confidence), 
                0.8  # 使用与demo相同的NMS阈值
            ).numpy().tolist()
            
            # 过滤检测结果
            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]
            
            # 转换为torch格式
            if len(detections) > 0:
                H, W = image_pil.size[1], image_pil.size[0]
                boxes_torch = torch.from_numpy(detections.xyxy) / torch.tensor([W, H, W, H])
                phrases = [f"person_{i}" for i in range(len(detections))]
                return boxes_torch, phrases
        
        return None, None
    
    def _generate_masks(self, image_pil, boxes_filt):
        """生成掩码 - 使用与demo相同的方法"""
        if boxes_filt is None or len(boxes_filt) == 0:
            return None
        
        # 设置图像
        self.sam_predictor.set_image(np.array(image_pil))
        
        # 转换框格式
        H, W = image_pil.size[1], image_pil.size[0]
        boxes_xyxy = boxes_filt * torch.Tensor([W, H, W, H])
        
        masks_list = []
        for box in boxes_xyxy:
            # 使用与demo相同的SAM预测方法
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box.cpu().numpy(),
                multimask_output=True,  # 使用与demo相同的multimask_output=True
            )
            # 选择最佳掩码
            best_mask = masks[np.argmax(scores)]
            masks_list.append(best_mask)
        
        return masks_list
    
    def _save_binary_mask(self, masks_list, output_path):
        """保存黑白掩码 - 人物为黑色，背景为白色（颠倒）"""
        if masks_list is None or len(masks_list) == 0:
            # 如果没有检测到人物，创建全黑掩码
            # 需要获取原图尺寸
            return False
        
        # 合并所有掩码 - 颠倒黑白：人物为黑色，背景为白色
        combined_mask = np.ones_like(masks_list[0], dtype=np.uint8) * 255  # 先设为全白
        for mask in masks_list:
            binary_mask = (mask > 0.5).astype(np.uint8)
            combined_mask[binary_mask > 0] = 0  # 人物区域设为黑色
        
        # 保存为JPG格式
        cv2.imwrite(output_path, combined_mask)
        return True
    
    def process_image(self, image_path, output_path):
        """处理单张图像"""
        try:
            # 加载图像
            image_pil, image = self._load_image(image_path)
            
            # 检测人物
            boxes_filt, pred_phrases = self._detect_humans(image_pil, image)
            
            if boxes_filt is None:
                # 没有检测到人物，创建全白掩码（颠倒后）
                H, W = image_pil.size[1], image_pil.size[0]
                white_mask = np.ones((H, W), dtype=np.uint8) * 255
                cv2.imwrite(output_path, white_mask)
                return False, "未检测到人物"
            
            # 生成掩码
            masks_list = self._generate_masks(image_pil, boxes_filt)
            
            # 保存掩码
            success = self._save_binary_mask(masks_list, output_path)
            
            if success:
                return True, f"检测到 {len(pred_phrases)} 个人物"
            else:
                return False, "保存失败"
                
        except Exception as e:
            return False, f"处理错误: {str(e)}"
    
    def process_directory(self, input_dir, output_dir):
        """处理整个目录"""
        # 支持的图像格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        
        # 获取所有图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
        
        # 去重（Windows文件系统不区分大小写）
        image_files = list(set(image_files))
        image_files.sort()  # 排序以便有序处理
        
        if not image_files:
            print(f"在 {input_dir} 中未找到图像文件")
            return
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理统计
        success_count = 0
        failed_count = 0
        
        # 批量处理
        for image_path in tqdm(image_files, desc="处理图像"):
            # 生成输出文件名（保持原名，但改为.jpg）
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{image_name}.jpg")
            
            # 处理图像
            success, message = self.process_image(image_path, output_path)
            
            if success:
                success_count += 1
                print(f"✓ {os.path.basename(image_path)}: {message}")
            else:
                failed_count += 1
                print(f"✗ {os.path.basename(image_path)}: {message}")
        
        print(f"\n处理完成!")
        print(f"成功: {success_count} 个")
        print(f"失败: {failed_count} 个")
        print(f"输出目录: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Human Mask Generator - 人物掩码生成工具')
    parser.add_argument('input_dir', nargs='?', default='data', 
                       help='输入图像目录路径 (默认: data)')
    parser.add_argument('output_dir', nargs='?', default='outputs', 
                       help='输出掩码目录路径 (默认: outputs)')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("人物掩码生成器")
    print("=" * 50)
    
    # 获取输入和输出目录
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        print("请确保输入目录存在并包含图像文件")
        return
    
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"检测目标: 人物")
    print()
    
    # 创建生成器
    generator = HumanMaskGenerator()
    
    # 处理目录
    generator.process_directory(input_dir, output_dir)


if __name__ == "__main__":
    main()
