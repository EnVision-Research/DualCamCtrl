from PIL import Image
import os
import json
import numpy as np


class ImagePoseDatasetDict:
    def __init__(self, json_paths, transform=None):
        """
        json_paths: 所有 JSON 路径的列表
        transform: torchvision.transforms
        """
        self.transform = transform
        self.scene_datasets = {}  # key: json_path, value: SceneImagePoseDataset

        for json_path in json_paths:
            self.scene_datasets[json_path] = SceneImagePoseDataset(
                json_path, transform)

    def __getitem__(self, json_path):
        return self.scene_datasets[json_path]

    def keys(self):
        return list(self.scene_datasets.keys())

    def __len__(self):
        return sum(len(scene) for scene in self.scene_datasets.values())


class SceneImagePoseDataset:
    def __init__(self, json_path, transform=None):
        self.json_path = json_path
        self.transform = transform
        self.base_dir = os.path.dirname(json_path)
        self.camera_data = CameraDataset(json_path)

        # 替换为完整路径
        for frame in self.camera_data.frames:
            frame.file_path = os.path.join(self.base_dir, frame.file_path)

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

        # 相机内参
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

        # 所有帧
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
        """返回 4x4 的位姿矩阵"""
        return self.transform_matrix

    def rotation_matrix(self):
        return self.transform_matrix[:3, :3]

    def translation_vector(self):
        return self.transform_matrix[:3, 3]

    def camera_position(self):
        """返回相机在世界坐标系中的位置"""
        R = self.rotation_matrix()
        t = self.translation_vector()
        return -R.T @ t  # 世界坐标下相机中心

    def forward_vector(self):
        """相机朝向（z轴）"""
        return self.rotation_matrix()[:, 2]

    def up_vector(self):
        return self.rotation_matrix()[:, 1]

    def right_vector(self):
        return self.rotation_matrix()[:, 0]


# def get_
if __name__ == "__main__":
    import os
    import natsort
    # 示例用法
    total_json = []
    meta_json_path = '/hpc2hdd/home/hongfeizhang/hongfei_workspace/DiffSynth-Studio/camera_data_paths.json'
    if not os.path.exists(meta_json_path):

        data_root = '/hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/DL3DV-10K/DL3DV-ALL-960P'
        print(
            f"Meta JSON file not found: {meta_json_path}, scanning directory: {data_root}")
        for root, dirs, files in os.walk(data_root):
            for file in files:
                if file.endswith('.json'):
                    json_path = os.path.join(root, file)
                    print(f"Processing JSON file: {json_path}")
                    total_json.append(json_path)
        print(f"Total JSON files found: {len(total_json)}")
        total_json = natsort.natsorted(total_json)
        with open('/hpc2hdd/home/hongfeizhang/hongfei_workspace/DiffSynth-Studio/camera_data_paths.json', 'w') as f:
            for json_path in total_json:
                f.write(json_path + '\n')
    else:
        print(f"Meta JSON file found: {meta_json_path}, loading paths...")
        with open(meta_json_path, 'r') as f:
            total_json = [line.strip() for line in f.readlines()]
    print(f"len(total_json): {len(total_json)}")
    dataset = ImagePoseDatasetDict(total_json)
    print(f"共加载 {len(dataset.keys())} 个场景")
    # for scene in dataset.keys():
    #     print(f"{scene} -> 帧数: {len(dataset[scene])}")

    # # 取第一个场景的第0帧
    # scene_path = dataset.keys()[0]
    # sample = dataset.get_sample(scene_path, 0)
    # print("路径:", sample['path'])
    # print("位姿:\n", sample['pose'])
    # print("图像尺寸:", sample['image'].shape)
