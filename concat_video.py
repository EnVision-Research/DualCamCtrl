import os
import cv2
from natsort import natsorted
from joblib import Parallel, delayed

# 根目录
root_dir = "/hpc2hdd/home/hongfeizhang/dataset/ttr"
fps = 10  # 帧率，可自定义
num_jobs = -1  # 并行线程数

def images_to_video(image_dir, output_path):
    images = natsorted([img for img in os.listdir(image_dir) if img.endswith(".png")])
    if not images:
        print(f"[Skip] No images in {image_dir}")
        return
    first_image_path = os.path.join(image_dir, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"[Error] Cannot read image: {first_image_path}")
        return
    video_size = (frame.shape[1], frame.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, video_size)
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[Warning] Failed to load {img_path}, skipping.")
            continue
        out.write(img)
    out.release()
    print(f"[Saved] {output_path}")

def process_scene(scene_root):
    scene_name = os.path.basename(scene_root)
    left_path = os.path.join(scene_root, "image_left")
    right_path = os.path.join(scene_root, "image_right")
    output_left = os.path.join(scene_root, "video_left")
    output_right = os.path.join(scene_root, "video_right")
    os.makedirs(output_left, exist_ok=True)
    os.makedirs(output_right, exist_ok=True)
    left_video = os.path.join(output_left, f"{scene_name}_left.mp4")
    right_video = os.path.join(output_right, f"{scene_name}_right.mp4")
    print(f"[Processing] {scene_name}")
    images_to_video(left_path, left_video)
    images_to_video(right_path, right_video)

# 查找所有含有 image_left 和 image_right 的目录
scene_dirs = []
for root, dirs, files in os.walk(root_dir):
    if "image_left" in dirs and "image_right" in dirs:
        scene_dirs.append(root)

# 自然排序
scene_dirs = natsorted(scene_dirs)
print("Found scenes:")
for s in scene_dirs:
    print(f"  - {s}")

# 并行处理
Parallel(n_jobs=num_jobs)(delayed(process_scene)(scene) for scene in scene_dirs)
