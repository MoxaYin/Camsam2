"""
SAV-test 数据集评估脚本（SAM2.1 + Memory Bank 清理）

基于 SAM2.1 大模型，无需 CamSAM2 参数

数据结构理解：
- Annotations_6fps/video_0001/000/0.png = 对象000在帧0的GT
- Annotations_6fps/video_0001/001/0.png = 对象001在帧0的GT
- 每个对象在各自的文件夹中包含完整的帧序列
- 所有对象使用同一个JPEGImages_24fps中的视频帧来分割

改进策略：
1. Memory Bank 清理机制（Memory Bank Clearing）
   - 监控掩码面积变化
   - 当某帧掩码面积突然缩小 70% 以上时，强制清空旧 memory
   - 清理后继续推理，等待目标重新出现
   - 防止长视频后期因目标消失导致后续全黑的问题
"""
import os
import argparse
import numpy as np
import torch
from PIL import Image
import cv2
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sam2.build_sam import build_sam2_video_predictor


# ========================= 辅助函数 =========================

def resize_frame(frame, target_size):
    """缩放帧到目标尺寸"""
    h, w = frame.shape[:2]
    new_h, new_w = target_size
    if (h, w) != (new_h, new_w):
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return frame


def resize_mask(mask, target_size):
    """缩放掩码到目标尺寸"""
    # 确保 target_size 的值都是有效的整数
    target_h, target_w = target_size
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid target size: {target_size}")
    
    # 处理空掩码
    if mask is None or mask.size == 0:
        return np.zeros((target_h, target_w), dtype=bool)
    
    # 转换为 uint8，避免 OpenCV 布尔类型不兼容问题
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    elif mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    
    h, w = mask.shape[:2]
    
    # 验证输入掩码的尺寸
    if h <= 0 or w <= 0:
        return np.zeros((target_h, target_w), dtype=bool)
    
    if (h, w) != (target_h, target_w):
        # OpenCV resize 需要 (width, height) 格式
        try:
            resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            return resized > 127
        except cv2.error as e:
            print(f"Warning: cv2.resize failed with {h}x{w} -> {target_h}x{target_w}: {e}")
            return np.zeros((target_h, target_w), dtype=bool)
    
    return mask > 127


def get_mask_area(mask):
    """计算掩码的面积（前景像素数）"""
    return np.sum(mask > 0)


# ========================= 主要推理函数 =========================

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


def get_all_objects_in_video(video_annotation_dir):
    """
    获取视频中的所有分割对象 ID
    
    参数：
        video_annotation_dir: 视频的标注目录（如 Annotations_6fps/video_0001/）
    
    返回：
        sorted list of object IDs（例如 [0, 1, 2]）
    """
    object_dirs = sorted([d for d in os.listdir(video_annotation_dir) 
                         if os.path.isdir(os.path.join(video_annotation_dir, d))],
                        key=lambda x: int(x) if x.isdigit() else 999)
    return [int(obj_dir) for obj_dir in object_dirs]


def load_object_frames(object_dir):
    """
    加载单个分割对象的所有帧数据
    
    参数：
        object_dir: 对象目录（如 Annotations_6fps/video_0001/000/）
    
    返回：
        {frame_id: mask_array, ...}
        例如：{0: mask_array, 1: mask_array, ...}
    """
    frames_data = {}
    
    frame_files = sorted([f for f in os.listdir(object_dir)
                         if f.endswith('.png')],
                        key=lambda x: int(os.path.splitext(x)[0]))
    
    for frame_file in frame_files:
        frame_id = int(os.path.splitext(frame_file)[0])
        frame_mask = np.array(Image.open(os.path.join(object_dir, frame_file)))
        frames_data[frame_id] = frame_mask
    
    return frames_data


def get_frame_files(video_images_dir):
    """
    获取视频帧文件列表（按顺序）
    """
    frame_files = sorted([f for f in os.listdir(video_images_dir)
                         if f.endswith(('.jpg', '.jpeg', '.JPG', '.JPEG'))],
                        key=lambda x: int(os.path.splitext(x)[0]))
    return frame_files


def eval_single_object(video_name, object_id, annotations_path, images_path, 
                       output_path, predictor, 
                       enable_memory_clear=True, memory_clear_threshold=0.3):
    """
    对视频中的单个分割对象进行推理和评估
    
    参数：
        video_name: 视频名称（如 'video_0001'）
        object_id: 对象 ID（如 0, 1, 2）
        annotations_path: Annotations_6fps 的路径
        images_path: JPEGImages_24fps 的路径
        output_path: 输出路径
        predictor: SAM2 视频预测器
        enable_multiscale: 启用多尺度推理
        enable_memory_clear: 启用 Memory Bank 清理机制
        memory_clear_threshold: Memory 清理阈值 (0.3 = 70% 缩小)
    
    返回：
        metrics dict 或 None（如果出错）
    """
    try:
        # 路径
        object_annotation_dir = os.path.join(annotations_path, video_name, f"{object_id:03d}")
        video_images_dir = os.path.join(images_path, video_name)
        
        if not os.path.exists(object_annotation_dir):
            print(f"      ⚠️  对象目录不存在: {object_annotation_dir}")
            return None
        
        if not os.path.exists(video_images_dir):
            print(f"      ⚠️  视频目录不存在: {video_images_dir}")
            return None
        
        # 加载该对象的所有帧 GT
        object_frames = load_object_frames(object_annotation_dir)
        
        if not object_frames:
            print(f"      ⚠️  对象无有效帧数据")
            return None
        
        # 获取视频中的所有帧
        frame_files = get_frame_files(video_images_dir)
        total_frames = len(frame_files)
        
        # 获取第一帧的 mask 作为提示
        first_frame_ids = sorted(object_frames.keys())
        first_frame_id = first_frame_ids[0]
        first_frame_mask = object_frames[first_frame_id]
        
        # 验证第一帧掩码的有效性
        if first_frame_mask is None or first_frame_mask.size == 0:
            print(f"      ⚠️  第一帧掩码无效")
            return None
        
        h_orig, w_orig = first_frame_mask.shape[:2]
        if h_orig <= 0 or w_orig <= 0:
            print(f"      ⚠️  第一帧掩码尺寸无效: {h_orig}x{w_orig}")
            return None
        
        # 单尺度推理
        prediction_to_eval = _inference_single_scale(
            predictor, video_images_dir, first_frame_id, 
            first_frame_mask, total_frames,
            enable_memory_clear, memory_clear_threshold
        )
        
        # 注：指标评估已移除，将在单独的脚本中进行
        
        # 保存预测结果
        save_dir = os.path.join(output_path, video_name, f"{object_id:03d}")
        os.makedirs(save_dir, exist_ok=True)
        
        # 确保 prediction_to_eval 是正确格式
        if prediction_to_eval.dtype != np.uint8:
            # 如果是布尔值或浮点数，转换为 uint8 (0 或 255)
            if prediction_to_eval.dtype == bool:
                prediction_to_eval = prediction_to_eval.astype(np.uint8) * 255
            elif prediction_to_eval.dtype in [np.float32, np.float64]:
                prediction_to_eval = (prediction_to_eval > 0.5).astype(np.uint8) * 255
            else:
                prediction_to_eval = prediction_to_eval.astype(np.uint8)
        
        for i, pred_mask in enumerate(prediction_to_eval):
            save_file = os.path.join(save_dir, f"{i:05d}.png")
            # pred_mask 已经是 uint8 (0-255)，可以直接保存
            cv2.imwrite(save_file, pred_mask)
        
        print(f"      ✅ 对象 {object_id:03d}: 推理完成")
        print(f"         📁 结果保存到: {save_dir}")
        
        return True
        
    except Exception as e:
        print(f"      ❌ 对象 {object_id:03d} 出错: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 保存错误日志
        try:
            save_dir = os.path.join(output_path, video_name, f"{object_id:03d}")
            os.makedirs(save_dir, exist_ok=True)
            error_log = os.path.join(save_dir, "ERROR.log")
            with open(error_log, 'w') as f:
                f.write(f"Object {object_id} inference failed\n")
                f.write(f"Error: {str(e)}\n")
        except:
            pass
        
        return None


def _inference_single_scale(predictor, video_images_dir, first_frame_id, 
                            first_frame_mask, total_frames,
                            enable_memory_clear, memory_clear_threshold):
    """
    单尺度推理（可选：启用 Memory Bank 清理机制）
    
    Memory Bank 清理逻辑：
    - 监控掩码面积变化
    - 当掩码面积缩小超过阈值时，清理Memory并从该帧重新初始化
    - 清理后继续推理后续帧，等待目标重新出现
    
    错误恢复逻辑：
    - 如果在清理后出现 AssertionError，使用最近有效的 mask 重新初始化
    - 从下一帧继续推理，而不是停止
    """
    # 初始化推理状态
    inference_state = predictor.init_state(video_path=video_images_dir)
    predictor.reset_state(inference_state)
    
    # 添加 mask 提示（在第一帧）
    ann_obj_id = 1
    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=first_frame_id,
        obj_id=ann_obj_id,
        mask=first_frame_mask
    )
    
    # 推理整个视频，支持 Memory Bank 清理和错误恢复
    video_segments = {}
    prev_mask_area = None
    memory_cleared = False
    memory_clear_frame = None  # 记录 Memory Clear 触发的帧号
    frames_since_clear = 0
    failed_frames = 0
    next_start_frame = None  # 用于错误恢复后从指定帧开始
    last_valid_mask_before_clear = None  # 记录清理前最后一个有效的mask
    
    # 外层循环用于错误恢复重试
    while True:
        try:
            # 如果有 next_start_frame，从该帧开始继续推理
            if next_start_frame is not None:
                print(f"        Recovery: continuing from frame {next_start_frame}...")
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                    inference_state, start_frame_idx=next_start_frame
                ):
                    # 安全性检查
                    if len(out_obj_ids) == 0:
                        failed_frames += 1
                        continue
                    
                    mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                    
                    # 确保掩码是 2D 数组
                    if mask.ndim != 2:
                        if mask.ndim == 1:
                            failed_frames += 1
                            continue
                        elif mask.ndim == 3:
                            mask = mask[0]
                        else:
                            failed_frames += 1
                            continue
                    
                    video_segments[out_frame_idx] = {out_obj_ids[0]: mask}
                    
                    # ========== Memory Bank 清理后的恢复逻辑 ==========
                    # 如果目标在清理后重新出现，使用清理前的有效mask重新初始化
                    if memory_cleared and last_valid_mask_before_clear is not None:
                        curr_mask_area = get_mask_area(mask)
                        
                        # 目标重新出现：面积从消失状态恢复到有意义的大小
                        if curr_mask_area > 100:  # 简单启发式：超过100像素认为是真正的重新出现
                            print(
                                f"        [Memory Reinit] Frame {out_frame_idx}: Target reappeared (area={curr_mask_area}), reinitializing with previous valid mask..."
                            )
                            # 用清理前的有效mask重新初始化状态
                            predictor.reset_state(inference_state)
                            _, _, _ = predictor.add_new_mask(
                                inference_state=inference_state,
                                frame_idx=memory_clear_frame,  # 用清理时的帧号
                                obj_id=ann_obj_id,
                                mask=last_valid_mask_before_clear,  # 用清理前的有效mask
                            )
                            memory_cleared = False
                            memory_clear_frame = None
                            last_valid_mask_before_clear = None
                            print("        [Memory Reinit] Reinitialization completed, continuing inference...")
                            # 继续从当前帧推理
                            prev_mask_area = curr_mask_area
                            continue
                        else:
                            # 还是黑图，继续等待
                            frames_since_clear += 1
                            continue
                
                # 正常完成，跳出外层循环
                next_start_frame = None
                break
            else:
                # 第一次推理，从头开始
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                    # 安全性检查
                    if len(out_obj_ids) == 0:
                        failed_frames += 1
                        continue
                    
                    mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                    
                    # 确保掩码是 2D 数组
                    if mask.ndim != 2:
                        if mask.ndim == 1:
                            failed_frames += 1
                            continue
                        elif mask.ndim == 3:
                            mask = mask[0]
                        else:
                            failed_frames += 1
                            continue
                    
                    video_segments[out_frame_idx] = {out_obj_ids[0]: mask}
                    
                    # ========== Memory Bank 清理机制 ==========
                    if enable_memory_clear:
                        curr_mask_area = get_mask_area(mask)
                        
                        if not memory_cleared and prev_mask_area is not None:
                            area_ratio = curr_mask_area / (prev_mask_area + 1e-8)
                            
                            # 掩码面积缩小超过阈值 - 记录清理前的有效mask并继续推理
                            if area_ratio < memory_clear_threshold:
                                print(
                                    f"        [Memory Clear] Frame {out_frame_idx}: Area drop {area_ratio:.2%}, preparing for memory clear..."
                                )
                                
                                # 记录清理前最后的有效mask（前一帧），用于目标重新出现时的恢复
                                last_valid_frame_before_clear = max(video_segments.keys()) - 1 if len(video_segments) > 1 else out_frame_idx
                                if last_valid_frame_before_clear in video_segments:
                                    last_valid_mask_before_clear = next(iter(video_segments[last_valid_frame_before_clear].values()))
                                
                                # 清空旧 memory，但这次不立即重新初始化
                                predictor.reset_state(inference_state)
                                memory_cleared = True
                                memory_clear_frame = out_frame_idx
                                frames_since_clear = 0
                                print("        [Memory Clear] Memory cleared, waiting for target to reappear...")
                                # 重要：继续推理，不重新初始化，这样可以继续追踪消失的目标
                                prev_mask_area = curr_mask_area
                                continue
                        
                        # 清理后等待目标重新出现
                        if memory_cleared:
                            frames_since_clear += 1
                            prev_mask_area = curr_mask_area
                        else:
                            prev_mask_area = curr_mask_area
                
                # 正常完成，跳出外层循环
                break
        
        except AssertionError as e:
            print(f"        ⚠️  AssertionError during propagation: {str(e)[:100]}")
            
            # 尝试从最后有效的帧恢复
            if len(video_segments) > 0:
                last_valid_frame = max(video_segments.keys())
                last_valid_mask = next(iter(video_segments[last_valid_frame].values()))
                
                # 如果已经到达或超过最后一帧，则结束
                if last_valid_frame + 1 >= total_frames:
                    print(f"        ✓ Already reached end of video, using available frames")
                    break
                
                try:
                    print(
                        f"        🔄 Recovery: reinit from frame {last_valid_frame}, continue from {last_valid_frame + 1}"
                    )
                    predictor.reset_state(inference_state)
                    _, _, _ = predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=last_valid_frame,
                        obj_id=ann_obj_id,
                        mask=last_valid_mask,
                    )
                    next_start_frame = last_valid_frame + 1
                    memory_cleared = True
                    frames_since_clear = 0
                    # 继续 while 循环重试
                    continue
                except Exception as recovery_error:
                    print(f"        ❌ Recovery failed: {str(recovery_error)[:100]}, stopping inference")
                    break
            else:
                print(f"        ❌ No valid frames to recover from, stopping inference")
                break
        
        except Exception as e:
            print(f"        ⚠️  Error during propagation: {str(e)[:100]}")
            
            # 尝试从最后有效的帧恢复
            if len(video_segments) > 0:
                last_valid_frame = max(video_segments.keys())
                last_valid_mask = next(iter(video_segments[last_valid_frame].values()))
                
                if last_valid_frame + 1 >= total_frames:
                    print(f"        ✓ Already reached end of video, using available frames")
                    break
                
                try:
                    print(
                        f"        🔄 Recovery: reinit from frame {last_valid_frame}, continue from {last_valid_frame + 1}"
                    )
                    predictor.reset_state(inference_state)
                    _, _, _ = predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=last_valid_frame,
                        obj_id=ann_obj_id,
                        mask=last_valid_mask,
                    )
                    next_start_frame = last_valid_frame + 1
                    memory_cleared = True
                    frames_since_clear = 0
                    continue
                except Exception as recovery_error:
                    print(f"        ❌ Recovery failed: {str(recovery_error)[:100]}, stopping inference")
                    break
            else:
                print(f"        ❌ No valid frames to recover from, stopping inference")
                break
    
    # 收集推理结果
    prediction_to_eval = []
    for frame_idx in range(total_frames):
        if frame_idx in video_segments:
            for out_obj_id, out_mask in video_segments[frame_idx].items():
                prediction_to_eval.append(out_mask)
                break
        else:
            prediction_to_eval.append(np.zeros_like(first_frame_mask))
    
    if failed_frames > 0:
        print(f"        Info: {failed_frames} frames skipped due to inference issues")
    
    return np.array(prediction_to_eval)


def eval_video_all_objects(video_name, annotations_path, images_path, 
                           output_path, predictor, 
                           enable_memory_clear=True,
                           memory_clear_threshold=0.3):
    """
    对视频中的所有分割对象进行推理和评估
    
    返回：
        {object_id: metrics_dict, ...}
    """
    print(f"\n🎬 处理视频: {video_name}")
    
    video_annotation_dir = os.path.join(annotations_path, video_name)
    
    if not os.path.exists(video_annotation_dir):
        print(f"   ❌ 标注目录不存在: {video_annotation_dir}")
        return {}
    
    # 获取该视频中的所有对象 ID
    object_ids = get_all_objects_in_video(video_annotation_dir)
    
    if not object_ids:
        print(f"   ⚠️  未找到分割对象")
        return {}
    
    print(f"   📊 找到 {len(object_ids)} 个分割对象: {object_ids}")
    
    video_results = {}
    
    for object_id in object_ids:
        metrics = eval_single_object(
            video_name=video_name,
            object_id=object_id,
            annotations_path=annotations_path,
            images_path=images_path,
            output_path=output_path,
            predictor=predictor,
            enable_memory_clear=enable_memory_clear,
            memory_clear_threshold=memory_clear_threshold
        )
        
        if metrics is not None:
            video_results[object_id] = metrics
    
    return video_results


def parse_args():
    """命令行参数"""
    parser = argparse.ArgumentParser("SAV-test 评估（Memory Bank 清理）", add_help=True)
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_t.yaml",
                        help="模型配置文件")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/sam2_hiera_tiny.pt",
                        help="SAM2 模型权重")
    parser.add_argument("--camsam2_extra", type=str, required=False,
                        help="CamSAM2 模块权重")
    parser.add_argument("--output_mode", type=str, default="combined_mask",
                        choices=["original_sam2_mask", "combined_mask"],
                        help="输出模式")
    parser.add_argument("--annotations_path", type=str, required=True,
                        help="Annotations_6fps 路径")
    parser.add_argument("--images_path", type=str, required=True,
                        help="JPEGImages_24fps 路径")
    parser.add_argument("--output_path", type=str, required=True,
                        help="输出路径")
    parser.add_argument("--enable_memory_clear", action="store_true", default=True,
                        help="启用 Memory Bank 清理机制")
    parser.add_argument("--disable_memory_clear", action="store_true",
                        help="禁用 Memory Bank 清理机制")
    parser.add_argument("--memory_clear_threshold", type=float, default=0.3,
                        help="Memory 清理阈值（面积缩小比例，默认 0.3 = 70%）")
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    device = get_device()
    
    # 处理改进策略开关
    enable_memory_clear = args.enable_memory_clear and not args.disable_memory_clear
    memory_clear_threshold = args.memory_clear_threshold
    
    # 显示配置
    print(f"\n{'='*80}")
    print(f"SAV-test 评估（SAM2.1 + Memory Bank 清理机制）")
    print(f"{'='*80}")
    print(f"模型配置: {args.model_cfg}")
    print(f"模型权重: {args.ckpt_path}")
    print(f"输出路径: {os.path.abspath(args.output_path)}")
    print(f"\n改进策略:")
    print(f"  • Memory Bank 清理: {'启用 (阈值={:.1%})'.format(1-memory_clear_threshold) if enable_memory_clear else '禁用'}")
    print(f"{'='*80}\n")
    
    # 构建 SAM2 视频预测器
    print("📦 加载 SAM2 模型...")
    predictor = build_sam2_video_predictor(
        args.model_cfg, args.ckpt_path, device=device
    )
    print("✅ SAM2 模型加载完成\n")
    
    # 获取所有视频
    videos = sorted([v for v in os.listdir(args.annotations_path) 
                    if os.path.isdir(os.path.join(args.annotations_path, v))])
    
    print(f"📹 找到 {len(videos)} 个视频")
    
    # 评估每个视频的所有对象
    all_results = {}
    for video_name in videos:
        video_results = eval_video_all_objects(
            video_name=video_name,
            annotations_path=args.annotations_path,
            images_path=args.images_path,
            output_path=args.output_path,
            predictor=predictor,
            enable_memory_clear=enable_memory_clear,
            memory_clear_threshold=memory_clear_threshold
        )
        all_results[video_name] = video_results
    
    # 推理完成
    print("\n" + "="*80)
    print("📊 推理完成！")
    
    # 列出输出目录结构
    print(f"\n📂 输出目录结构:")
    print(f"   {os.path.abspath(args.output_path)}/")
    try:
        video_count = 0
        object_count = 0
        for root, dirs, files in os.walk(args.output_path):
            level = root.replace(args.output_path, '').count(os.sep)
            if level == 0:
                video_count = len([d for d in dirs if d.startswith('sav_')])
            if level == 1 and 'sav_' in os.path.basename(root):
                object_count += len(dirs)
        
        print(f"   📹 视频数: {video_count}")
        print(f"   📊 对象总数: {object_count}")
        print(f"   💾 掩码文件已保存")
    except:
        pass
    
    print(f"\n✅ 所有推理结果已保存到: {os.path.abspath(args.output_path)}")
    print("💡 提示: 请使用单独的脚本计算推理指标（IoU, BIoU, TIoU 等）")
    print("="*80)


if __name__ == "__main__":
    main()
