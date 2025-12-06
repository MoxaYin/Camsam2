#!/usr/bin/env python3
"""
行人视频分割脚本 (Pedestrian Video Segmentation)

功能：
1. 读取 MP4 视频文件
2. 使用 YOLO 检测第一帧中的所有行人
3. 使用 SAM2 对检测到的行人进行分割
4. 在整个视频中追踪和分割这些行人
5. 保存分割结果（视频或图像序列）

依赖：
- torch
- opencv-python
- ultralytics (YOLO)
- numpy
- PIL

使用方法：
python eval_person.py --video_path input.mp4 --output_path output/
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
import tempfile
import shutil

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sam2.build_sam import build_sam2_video_predictor


# ========================= 配置 =========================

# YOLO 行人类别 ID (COCO 数据集中 person = 0)
PERSON_CLASS_ID = 0

# 颜色调色板 (用于可视化不同行人)
COLORS = [
    (255, 0, 0),     # 红
    (0, 255, 0),     # 绿
    (0, 0, 255),     # 蓝
    (255, 255, 0),   # 青
    (255, 0, 255),   # 品红
    (0, 255, 255),   # 黄
    (128, 0, 0),     # 深红
    (0, 128, 0),     # 深绿
    (0, 0, 128),     # 深蓝
    (128, 128, 0),   # 橄榄
    (128, 0, 128),   # 紫
    (0, 128, 128),   # 青绿
    (255, 128, 0),   # 橙
    (255, 0, 128),   # 粉红
    (128, 255, 0),   # 黄绿
]


# ========================= 辅助函数 =========================

def get_device():
    """获取计算设备"""
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
        print("注意: MPS 支持处于试验阶段，性能可能下降")

    return device


def extract_frames_from_video(video_path, output_dir):
    """
    从视频中提取所有帧并保存为 JPEG 图像

    参数:
        video_path: 视频文件路径
        output_dir: 输出目录

    返回:
        frame_count: 总帧数
        fps: 视频帧率
        frame_size: (width, height)
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {frame_width}x{frame_height}, {fps:.2f} fps, {total_frames} 帧")

    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="提取视频帧")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 保存帧 (BGR -> RGB -> JPEG)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_path = os.path.join(output_dir, f"{frame_idx:05d}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    print(f"已提取 {frame_idx} 帧到 {output_dir}")
    return frame_idx, fps, (frame_width, frame_height)


def detect_persons_yolo(frame_path, confidence_threshold=0.5):
    """
    使用 YOLO 检测帧中的所有行人

    参数:
        frame_path: 帧图像路径
        confidence_threshold: 置信度阈值

    返回:
        detections: list of dict, 每个 dict 包含 'bbox' (xyxy) 和 'confidence'
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("请安装 ultralytics: pip install ultralytics")

    # 加载 YOLO 模型
    model = YOLO("yolov8n.pt")  # 使用 nano 模型，快速且准确

    # 运行检测
    results = model(frame_path, verbose=False)

    detections = []
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = boxes.conf[i].item()

            # 只保留行人类别 (class 0 = person)
            if cls_id == PERSON_CLASS_ID and conf >= confidence_threshold:
                xyxy = boxes.xyxy[i].cpu().numpy()
                detections.append({
                    'bbox': xyxy,  # [x1, y1, x2, y2]
                    'confidence': conf
                })

    print(f"检测到 {len(detections)} 个行人")
    return detections


def detect_persons_grounding_dino(frame_path, confidence_threshold=0.3):
    """
    使用 Grounding DINO 检测帧中的所有行人（备选方案）

    参数:
        frame_path: 帧图像路径
        confidence_threshold: 置信度阈值

    返回:
        detections: list of dict
    """
    try:
        from groundingdino.util.inference import load_model, predict
        from groundingdino.util import box_ops
    except ImportError:
        print("Grounding DINO 未安装，使用 YOLO 替代")
        return detect_persons_yolo(frame_path, confidence_threshold)

    # 这里可以实现 Grounding DINO 的检测逻辑
    # 目前返回空列表，作为备选方案
    return []


def create_mask_from_bbox(bbox, frame_shape):
    """
    从边界框创建一个简单的矩形掩码

    参数:
        bbox: [x1, y1, x2, y2]
        frame_shape: (height, width)

    返回:
        mask: numpy array (H, W), bool
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=bool)

    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    mask[y1:y2, x1:x2] = True
    return mask


def overlay_masks_on_frame(frame, masks, alpha=0.5):
    """
    将多个掩码叠加到帧上进行可视化

    参数:
        frame: numpy array (H, W, 3), BGR
        masks: dict {obj_id: mask_array}
        alpha: 透明度

    返回:
        overlayed_frame: numpy array (H, W, 3), BGR
    """
    overlayed = frame.copy()

    for obj_id, mask in masks.items():
        color = COLORS[obj_id % len(COLORS)]

        # 确保掩码是 2D
        if mask.ndim == 3:
            mask = mask[0]

        # 创建彩色掩码
        colored_mask = np.zeros_like(frame)
        colored_mask[mask] = color

        # 混合
        overlayed = np.where(
            mask[:, :, np.newaxis],
            cv2.addWeighted(overlayed, 1 - alpha, colored_mask, alpha, 0),
            overlayed
        )

    return overlayed


def save_masks_as_images(masks_all_frames, output_dir, num_objects, frame_shape=None):
    """
    保存所有帧的掩码为图像（所有行人合并到同一帧）

    参数:
        masks_all_frames: dict {frame_idx: {obj_id: mask}}
        output_dir: 输出目录
        num_objects: 对象数量
        frame_shape: (height, width) 帧尺寸，用于初始化空掩码
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存掩码 - 每帧一个图像，所有行人合并
    for frame_idx, frame_masks in tqdm(masks_all_frames.items(), desc="保存掩码"):
        # 获取帧尺寸
        if frame_shape is None:
            # 从第一个掩码获取尺寸
            first_mask = next(iter(frame_masks.values()))
            if first_mask.ndim == 3:
                first_mask = first_mask[0]
            h, w = first_mask.shape
        else:
            h, w = frame_shape

        # 创建合并掩码 (所有行人合并到一起)
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for obj_id, mask in frame_masks.items():
            # 确保掩码是 2D
            if mask.ndim == 3:
                mask = mask[0]
            # 合并掩码 (使用不同的像素值区分不同行人，或者简单合并)
            # 这里使用简单合并：所有行人区域都是255
            combined_mask[mask > 0] = 255

        # 保存合并后的掩码
        mask_path = os.path.join(output_dir, f"{frame_idx:05d}.png")
        cv2.imwrite(mask_path, combined_mask)


def create_output_video(frames_dir, masks_all_frames, output_path, fps, frame_size):
    """
    创建带有分割掩码叠加的输出视频

    参数:
        frames_dir: 原始帧目录
        masks_all_frames: dict {frame_idx: {obj_id: mask}}
        output_path: 输出视频路径
        fps: 帧率
        frame_size: (width, height)
    """
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for frame_idx, frame_file in enumerate(tqdm(frame_files, desc="生成输出视频")):
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)

        if frame_idx in masks_all_frames:
            frame = overlay_masks_on_frame(frame, masks_all_frames[frame_idx])

        out.write(frame)

    out.release()
    print(f"输出视频已保存到: {output_path}")


# ========================= 主要推理函数 =========================

def segment_persons_in_video(
    video_path,
    output_path,
    model_cfg="sam2_hiera_t.yaml",
    ckpt_path="checkpoints/sam2_hiera_tiny.pt",
    detection_method="yolo",
    confidence_threshold=0.5,
    save_masks=True,
    save_video=True,
    keep_temp_frames=False
):
    """
    对视频中的所有行人进行分割

    参数:
        video_path: 输入视频路径
        output_path: 输出目录
        model_cfg: SAM2 模型配置文件
        ckpt_path: SAM2 模型权重路径
        detection_method: 行人检测方法 ('yolo' 或 'grounding_dino')
        confidence_threshold: 检测置信度阈值
        save_masks: 是否保存掩码图像
        save_video: 是否保存带掩码的视频
        keep_temp_frames: 是否保留临时帧文件
    """
    device = get_device()

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 创建临时目录存放视频帧
    if keep_temp_frames:
        frames_dir = os.path.join(output_path, "frames")
    else:
        temp_dir = tempfile.mkdtemp()
        frames_dir = temp_dir

    try:
        # Step 1: 提取视频帧
        print("\n" + "="*60)
        print("Step 1: 提取视频帧")
        print("="*60)
        num_frames, fps, frame_size = extract_frames_from_video(video_path, frames_dir)

        if num_frames == 0:
            raise ValueError("视频中没有帧")

        # Step 2: 检测第一帧中的行人
        print("\n" + "="*60)
        print("Step 2: 检测第一帧中的行人")
        print("="*60)
        first_frame_path = os.path.join(frames_dir, "00000.jpg")

        if detection_method == "yolo":
            detections = detect_persons_yolo(first_frame_path, confidence_threshold)
        else:
            detections = detect_persons_grounding_dino(first_frame_path, confidence_threshold)

        if len(detections) == 0:
            print("警告: 未检测到任何行人，尝试降低置信度阈值")
            detections = detect_persons_yolo(first_frame_path, confidence_threshold=0.3)

        if len(detections) == 0:
            raise ValueError("未检测到任何行人")

        # 获取第一帧尺寸
        first_frame = np.array(Image.open(first_frame_path))
        frame_height, frame_width = first_frame.shape[:2]

        # Step 3: 加载 SAM2 模型
        print("\n" + "="*60)
        print("Step 3: 加载 SAM2 模型")
        print("="*60)
        predictor = build_sam2_video_predictor(model_cfg, ckpt_path, device=device)
        print("SAM2 模型加载完成")

        # Step 4: 初始化推理状态
        print("\n" + "="*60)
        print("Step 4: 初始化视频分割")
        print("="*60)
        inference_state = predictor.init_state(video_path=frames_dir)
        predictor.reset_state(inference_state)

        # Step 5: 为每个检测到的行人添加提示
        print("\n" + "="*60)
        print("Step 5: 添加行人检测提示")
        print("="*60)

        for obj_id, detection in enumerate(detections):
            bbox = detection['bbox']
            print(f"  行人 {obj_id}: bbox={bbox.astype(int).tolist()}, conf={detection['confidence']:.2f}")

            # 使用边界框作为提示（SAM2 支持 box prompt）
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                box=bbox,
            )

        print(f"已添加 {len(detections)} 个行人提示")

        # Step 6: 在整个视频中传播分割
        print("\n" + "="*60)
        print("Step 6: 视频分割传播")
        print("="*60)

        masks_all_frames = {}

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            frame_masks = {}

            for i, obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy()

                # 确保掩码是 2D
                if mask.ndim == 3:
                    mask = mask[0]

                frame_masks[obj_id] = mask

            masks_all_frames[out_frame_idx] = frame_masks

        print(f"分割完成，共处理 {len(masks_all_frames)} 帧")

        # Step 7: 保存结果
        print("\n" + "="*60)
        print("Step 7: 保存结果")
        print("="*60)

        if save_masks:
            masks_dir = os.path.join(output_path, "masks")
            save_masks_as_images(masks_all_frames, masks_dir, len(detections),
                                 frame_shape=(frame_height, frame_width))
            print(f"掩码已保存到: {masks_dir}")

        if save_video:
            video_output_path = os.path.join(output_path, "segmented_video.mp4")
            create_output_video(frames_dir, masks_all_frames, video_output_path, fps, frame_size)

        # 统计信息
        print("\n" + "="*60)
        print("分割完成！")
        print("="*60)
        print(f"  输入视频: {video_path}")
        print(f"  检测到行人数: {len(detections)}")
        print(f"  处理帧数: {num_frames}")
        print(f"  输出目录: {output_path}")

        return masks_all_frames

    finally:
        # 清理临时文件
        if not keep_temp_frames and 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="行人视频分割 - 使用 SAM2 对视频中的行人进行分割",
        add_help=True
    )

    # 输入输出
    parser.add_argument(
        "--video_path", "-v",
        type=str,
        required=True,
        help="输入视频路径 (MP4 格式)"
    )
    parser.add_argument(
        "--output_path", "-o",
        type=str,
        required=True,
        help="输出目录路径"
    )

    # 模型配置
    parser.add_argument(
        "--model_cfg",
        type=str,
        default="sam2_hiera_t.yaml",
        help="SAM2 模型配置文件 (默认: sam2_hiera_t.yaml)"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/sam2_hiera_tiny.pt",
        help="SAM2 模型权重路径"
    )

    # 检测配置
    parser.add_argument(
        "--detection_method",
        type=str,
        default="yolo",
        choices=["yolo", "grounding_dino"],
        help="行人检测方法 (默认: yolo)"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="检测置信度阈值 (默认: 0.5)"
    )

    # 输出选项
    parser.add_argument(
        "--no_masks",
        action="store_true",
        help="不保存掩码图像"
    )
    parser.add_argument(
        "--no_video",
        action="store_true",
        help="不保存输出视频"
    )
    parser.add_argument(
        "--keep_frames",
        action="store_true",
        help="保留提取的视频帧"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 检查输入文件
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"视频文件不存在: {args.video_path}")

    # 运行分割
    segment_persons_in_video(
        video_path=args.video_path,
        output_path=args.output_path,
        model_cfg=args.model_cfg,
        ckpt_path=args.ckpt_path,
        detection_method=args.detection_method,
        confidence_threshold=args.confidence_threshold,
        save_masks=not args.no_masks,
        save_video=not args.no_video,
        keep_temp_frames=args.keep_frames
    )


if __name__ == "__main__":
    main()
