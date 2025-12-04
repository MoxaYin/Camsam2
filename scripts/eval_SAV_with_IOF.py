"""
SAV-test 数据集推理脚本（SAM2+IOF 配置版本）

功能特性：
1. 加载训练好的伪装参数
2. 冻结 CamSAM2 新增的伪装模块（EOF、OPG、伪装 token）
3. 保持 SAM2 主干和 IOF 模块完全激活
4. 使用 combined_mask 输出模式（只使用 IOF，不使用伪装抑制）
5. 保存分割结果为 PNG 格式，不进行指标评估

这是 SAM2 + IOF 配置，用于在 SAV-test 上进行推理和结果保存

数据结构理解：
- Annotations_6fps/video_0001/000/0.png = 对象000在帧0的GT
- Annotations_6fps/video_0001/001/0.png = 对象001在帧0的GT
- 每个对象在各自的文件夹中包含完整的帧序列
- 所有对象使用同一个JPEGImages_24fps中的视频帧来分割
"""
import os
import argparse
import numpy as np
import torch
from PIL import Image
import cv2
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sam2.build_sam import build_camsam2_video_predictor


def get_device():
    """获取设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.float32).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print("\n注意: MPS 支持处于试验阶段，性能可能下降")

    return device


def freeze_only_camouflaged_modules(model):
    """
    只冻结 CamSAM2 新增的模块（EOF、OPG 和伪装 token），保持 SAM2+IOF 激活
    
    冻结的模块包括：
    - decamouflaged_token (伪装token)
    - decamouflaged_mlp (伪装MLP)
    - OPG 原型生成 (kmeans 相关在前向中动态执行，此处冻结特征提取部分)
    - EOF 边界增强部分
    
    保持激活的模块：
    - SAM2 主干 (所有层)
    - IOF 模块 (compress_hiera_feat, embedding_encoder 等)
    """
    # 首先解冻所有参数（设置为可训练）
    for param in model.parameters():
        param.requires_grad = True
    
    # 冻结 CamSAM2 新增模块
    frozen_modules = [
        'decamouflaged_token',
        'decamouflaged_mlp',
    ]
    
    frozen_count = 0
    for name, module in model.named_modules():
        # 检查是否是需要冻结的模块
        for frozen_module in frozen_modules:
            if frozen_module in name:
                print(f"🔒 冻结模块: {name}")
                for param in module.parameters():
                    param.requires_grad = False
                frozen_count += 1
                break
    
    # 计算冻结/激活参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\n📊 参数统计:")
    print(f"   总参数数:     {total_params:,}")
    print(f"   激活参数:     {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"   冻结参数:     {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
    print(f"\n📋 激活模块配置:")
    print(f"   ✅ SAM2 主干 (所有层)")
    print(f"   ✅ IOF 模块 (compress_hiera_feat, embedding_encoder 等)")
    print(f"   🔒 冻结: 伪装 token 和 MLP (EOF/OPG)")


def inference_single_object(video_name, object_id, annotations_path, images_path, 
                            output_path, predictor, output_mode):
    """
    对视频中的单个分割对象进行推理
    
    参数：
        video_name: 视频名称（如 'video_0001'）
        object_id: 对象 ID（如 0, 1, 2）
        annotations_path: Annotations_6fps 的路径
        images_path: JPEGImages_24fps 的路径
        output_path: 输出路径
        predictor: SAM2/CamSAM2 预测器
        output_mode: 输出模式
    
    返回：
        True 如果成功，False 如果出错
    """
    try:
        # 路径
        object_annotation_dir = os.path.join(annotations_path, video_name, f"{object_id:03d}")
        video_images_dir = os.path.join(images_path, video_name)
        
        if not os.path.exists(object_annotation_dir):
            print(f"      ⚠️  对象目录不存在: {object_annotation_dir}")
            return False
        
        if not os.path.exists(video_images_dir):
            print(f"      ⚠️  视频目录不存在: {video_images_dir}")
            return False
        
        # 加载该对象的所有帧 GT（用于获取第一帧提示）
        frames_data = {}
        frame_files = sorted([f for f in os.listdir(object_annotation_dir)
                             if f.endswith('.png')],
                            key=lambda x: int(os.path.splitext(x)[0]))
        
        for frame_file in frame_files:
            frame_id = int(os.path.splitext(frame_file)[0])
            frame_mask = np.array(Image.open(os.path.join(object_annotation_dir, frame_file)))
            frames_data[frame_id] = frame_mask
        
        if not frames_data:
            print(f"      ⚠️  对象无有效帧数据")
            return False
        
        # 获取视频中的所有帧
        frame_files = sorted([f for f in os.listdir(video_images_dir)
                             if f.endswith(('.jpg', '.jpeg', '.JPG', '.JPEG'))],
                            key=lambda x: int(os.path.splitext(x)[0]))
        total_frames = len(frame_files)
        
        # 获取第一帧的 mask 作为提示
        first_frame_ids = sorted(frames_data.keys())
        first_frame_id = first_frame_ids[0]
        first_frame_mask = frames_data[first_frame_id]
        
        # 初始化推理状态
        inference_state = predictor.init_state(video_path=video_images_dir, output_mode=output_mode)
        predictor.reset_state(inference_state)
        
        # 添加 mask 提示（在第一帧）
        ann_obj_id = 1
        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=first_frame_id,
            obj_id=ann_obj_id,
            mask=first_frame_mask
        )
        
        # 推理整个视频
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        # 收集推理结果
        prediction_to_eval = []
        for frame_idx in range(total_frames):
            if frame_idx in video_segments:
                for out_obj_id, out_mask in video_segments[frame_idx].items():
                    prediction_to_eval.append(out_mask[0])
                    break
            else:
                prediction_to_eval.append(np.zeros_like(first_frame_mask))
        
        prediction_to_eval = np.array(prediction_to_eval)
        
        # 保存预测结果（PNG 格式）
        save_dir = os.path.join(output_path, video_name, f"{object_id:03d}")
        os.makedirs(save_dir, exist_ok=True)
        
        for i, pred_mask in enumerate(prediction_to_eval):
            save_file = os.path.join(save_dir, f"{i:05d}.png")
            cv2.imwrite(save_file, (pred_mask * 255).astype(np.uint8))
        
        print(f"      ✅ 对象 {object_id:03d}: 已保存 {len(prediction_to_eval)} 帧")
        return True
        
    except Exception as e:
        print(f"      ❌ 对象 {object_id:03d} 出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def inference_video_all_objects(video_name, annotations_path, images_path, 
                                output_path, predictor, output_mode):
    """
    对视频中的所有分割对象进行推理
    
    返回：
        True 如果至少一个对象成功，False 如果全部失败
    """
    print(f"\n🎬 处理视频: {video_name}")
    
    video_annotation_dir = os.path.join(annotations_path, video_name)
    
    if not os.path.exists(video_annotation_dir):
        print(f"   ❌ 标注目录不存在: {video_annotation_dir}")
        return False
    
    # 获取该视频中的所有对象 ID
    object_dirs = sorted([d for d in os.listdir(video_annotation_dir) 
                         if os.path.isdir(os.path.join(video_annotation_dir, d))],
                        key=lambda x: int(x) if x.isdigit() else 999)
    object_ids = [int(obj_dir) for obj_dir in object_dirs]
    
    if not object_ids:
        print(f"   ⚠️  未找到分割对象")
        return False
    
    print(f"   📊 找到 {len(object_ids)} 个分割对象: {object_ids}")
    
    success_count = 0
    for object_id in object_ids:
        success = inference_single_object(
            video_name=video_name,
            object_id=object_id,
            annotations_path=annotations_path,
            images_path=images_path,
            output_path=output_path,
            predictor=predictor,
            output_mode=output_mode
        )
        if success:
            success_count += 1
    
    return success_count > 0


def parse_args():
    """命令行参数"""
    parser = argparse.ArgumentParser("SAV-test 评估（SAM2+IOF 配置）", add_help=True)
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_t.yaml",
                        help="模型配置文件")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/sam2_hiera_tiny.pt",
                        help="SAM2 模型权重")
    parser.add_argument("--camsam2_extra", type=str, required=True,
                        help="CamSAM2 伪装模块权重（必需）")
    parser.add_argument("--output_mode", type=str, default="combined_mask",
                        choices=["original_sam2_mask", "combined_mask"],
                        help="输出模式（combined_mask 包含 IOF）")
    parser.add_argument("--annotations_path", type=str, required=True,
                        help="Annotations_6fps 路径")
    parser.add_argument("--images_path", type=str, required=True,
                        help="JPEGImages_24fps 路径")
    parser.add_argument("--output_path", type=str, required=True,
                        help="输出路径")
    parser.add_argument("--freeze_except_iof", action="store_true", default=True,
                        help="冻结伪装模块（EOF/OPG），只保持 SAM2+IOF 激活（默认 True）")
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    device = get_device()
    
    print(f"\n{'='*80}")
    print(f"🎯 CamSAM2 SAV-test 推理（SAM2+IOF 配置）")
    print(f"{'='*80}")
    print(f"✅ SAM2 模型: {args.model_cfg}")
    print(f"✅ SAM2 权重: {args.ckpt_path}")
    print(f"✅ CamSAM2 权重: {args.camsam2_extra}")
    print(f"✅ 输出模式: {args.output_mode}")
    print(f"✅ 冻结伪装模块: {args.freeze_except_iof}")
    print(f"{'='*80}\n")
    
    # 构建预测器
    print("📦 构建模型...")
    predictor = build_camsam2_video_predictor(
        args.model_cfg, 
        args.ckpt_path, 
        device=device, 
        camsam2_extra=args.camsam2_extra
    )
    
    # 冻结伪装模块
    if args.freeze_except_iof and args.output_mode == "combined_mask":
        print("\n🔒 配置模块冻结...")
        # 直接访问 predictor 的 sam_mask_decoder
        if hasattr(predictor, 'sam_mask_decoder'):
            freeze_only_camouflaged_modules(predictor.sam_mask_decoder)
        else:
            print("⚠️  找不到 sam_mask_decoder，跳过冻结")
            # 尝试列出 predictor 中所有属性
            print("   可用属性:", [attr for attr in dir(predictor) if not attr.startswith('_')][:10])
    
    predictor.eval()
    
    # 获取所有视频
    print(f"\n📂 扫描数据集...")
    videos = sorted([v for v in os.listdir(args.annotations_path) 
                    if os.path.isdir(os.path.join(args.annotations_path, v))])
    
    print(f"📹 找到 {len(videos)} 个视频\n")
    
    # 推理每个视频的所有对象
    os.makedirs(args.output_path, exist_ok=True)
    success_count = 0
    total_count = 0
    
    for video_name in videos:
        success = inference_video_all_objects(
            video_name=video_name,
            annotations_path=args.annotations_path,
            images_path=args.images_path,
            output_path=args.output_path,
            predictor=predictor,
            output_mode=args.output_mode
        )
        if success:
            success_count += 1
        total_count += 1
    
    # 生成完成报告
    print("\n" + "="*80)
    print("📊 推理完成！")
    print(f"{'='*80}")
    print(f"✅ 成功处理视频: {success_count}/{total_count}")
    print(f"📁 输出路径: {args.output_path}")
    print(f"✅ 所有分割结果已保存为 PNG 格式")
    print("\n提示：使用以下命令计算指标:")
    print(f"   python scripts/eval_jf.py --pred_dir {args.output_path} --gt_dir {args.annotations_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
