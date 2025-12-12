import imageio

import open3d.visualization.rendering as rendering

import open3d as o3d
import numpy as np
import json
import os
from PIL import Image
os.environ["OPEN3D_RENDERING_HEADLESS"] = "true"


class ImagePoseDatasetDict:
    def __init__(self, json_paths, transform=None):

        self.transform = transform

        self.meta_json_paths = []
        for json_path in json_paths:
            self.meta_json_paths.append(json_path)
        print(f"self.meta_json_paths: {self.meta_json_paths[:10]}")

    def __getitem__(self, idx):
        total_len = len(self.meta_json_paths)
        idx = (idx+total_len) % total_len
        return SceneImagePoseDataset(self.meta_json_paths[idx], transform=self.transform)

    def __len__(self):
        return len(self.meta_json_paths)


class SceneImagePoseDataset:
    def __init__(self, json_path, transform=None):
        self.json_path = json_path
        self.transform = transform
        self.base_dir = os.path.dirname(json_path)
        self.camera_data = CameraDataset(json_path)

        for frame in self.camera_data.frames:
            frame.file_path = os.path.join(
                self.base_dir, frame.file_path.replace('images', 'images_4'))

    def __len__(self):
        return len(self.camera_data)

    def __getitem__(self, idx):
        frame = self.camera_data[idx]
        image = Image.open(frame.file_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'pose': frame.pose,
            'path': frame.file_path,
            'camera_position': frame.camera_position(),
            'forward_vector': frame.forward_vector()
        }

    def get_all_paths(self):
        return [frame.file_path for frame in self.camera_data.frames]


class CameraDataset:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.w = data['w']
        self.h = data['h']
        self.fl_x = data['fl_x']
        self.fl_y = data['fl_y']
        self.cx = data['cx']
        self.cy = data['cy']
        self.k1 = data['k1']
        self.k2 = data['k2']
        self.p1 = data['p1']
        self.p2 = data['p2']
        self.camera_model = data.get('camera_model', 'OPENCV')

        self.frames = [CameraFrame(f) for f in data['frames']]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]


class CameraFrame:
    def __init__(self, frame_dict):
        self.file_path = frame_dict['file_path']
        self.colmap_id = frame_dict.get('colmap_im_id', None)
        self.transform_matrix = np.array(
            frame_dict['transform_matrix'], dtype=np.float32)  # 4x4

    @property
    def pose(self):
        return self.transform_matrix

    def rotation_matrix(self):
        return self.transform_matrix[:3, :3]

    def translation_vector(self):
        return self.transform_matrix[:3, 3]

    def camera_position(self):
        R = self.rotation_matrix()
        t = self.translation_vector()
        return -R.T @ t  

    def forward_vector(self):
        return self.rotation_matrix()[:, 2]

    def up_vector(self):
        return self.rotation_matrix()[:, 1]

    def right_vector(self):
        return self.rotation_matrix()[:, 0]


def draw_camera_poses(c2ws, scale=0.1, save_path=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, c2w in enumerate(c2ws):
        origin = c2w[:3, 3]
        z_axis = c2w[:3, 2] * scale
        ax.quiver(*origin, *z_axis, color='b')  # camera forward
        ax.scatter(*origin, color='r', s=1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera Poses")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def create_image_plane(pose, image_path, scale=0.2):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((160, 120))  

    img_np = np.asarray(img).astype(np.float32) / 255.0
    h, w, _ = img_np.shape

    x = np.linspace(-scale, scale, w)
    y = np.linspace(-scale * h / w, scale * h / w, h)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)

    points = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
    colors = img_np.reshape(-1, 3)

    R = pose[:3, :3]
    t = pose[:3, 3]
    points_world = (R @ points.T).T + t[None, :]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points_world)
    pc.colors = o3d.utility.Vector3dVector(colors)
    return pc


def save_rotating_scene_video(scene_dataset, output_video_path, downsample=5, image_scale=0.3,
                              width=1920, height=1080, num_views=60, fps=20):
    renderer = rendering.OffscreenRenderer(width, height)
    scene = renderer.scene

    scene.set_background([1.0, 1.0, 1.0, 1.0])  # 白背景
    scene.scene.set_sun_light([0.577, -0.577, -0.577], [1.0, 1.0, 1.0], 75000)
    scene.scene.enable_sun_light(True)

    camera_positions = []
    camera_forwards = []

    for i in range(0, len(scene_dataset), downsample):
        sample = scene_dataset[i]
        c2w = sample['pose']
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        forward = R[:, 2]   
        camera_positions.append(t)
        camera_forwards.append(forward)

    camera_positions = np.stack(camera_positions)
    camera_forwards = np.stack(camera_forwards)

    avg_pos = camera_positions.mean(axis=0)
    avg_forward = camera_forwards.mean(axis=0)
    avg_target = avg_pos + avg_forward  

    radius = 2.0  
    up = np.array([0, 1, 0], dtype=np.float32)
    bounds = scene.bounding_box
    center = bounds.get_center()
    extent = bounds.get_extent()
    radius = np.linalg.norm(extent) * 0.3

    video_frames = []
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        x = np.cos(angle) * radius
        z = np.sin(angle) * radius
        eye = avg_target + np.array([x, 0.4 * radius, z])  

        renderer.setup_camera(
            60.0,
            avg_pos.astype(np.float32),  
            eye.astype(np.float32),      
            up
        )

        img = renderer.render_to_image()
        img_np = np.asarray(img)
        video_frames.append(img_np)

    imageio.mimsave(output_video_path, video_frames, fps=fps)
    print(f"🎥 Saved rotating scene video to: {output_video_path}")


def get_all_camera_poses(dataset_dict):

    all_poses = []
    for i in range(len(dataset_dict)):
        scene_dataset = dataset_dict[i]
        for j in range(len(scene_dataset)):
            sample = scene_dataset[j]
            all_poses.append(sample['pose'])
    return np.stack(all_poses)


