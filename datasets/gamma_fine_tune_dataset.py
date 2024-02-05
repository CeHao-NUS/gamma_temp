import copy
import pdb
import time
from typing import Union, List
from pathlib import Path
import glob
import torch.utils.data as data
import os
import open3d as o3d
import numpy as np
from tqdm import tqdm
from .data_utilts import point_cloud_center_and_scale, translate_joint_base_world_to_camera, translate_joint_direc_world_to_camera, sample_point, \
    translate_pc_world_to_camera, view_point_cloud_parts_and_joints, view_point_cloud_parts_and_center, voxel_sample_points, view_object_joint

class GammaDataset(data.Dataset):
    """
    objcet link dataset
    """
    def __init__(self,
                 root: Union[Path, str, List[Path], List[str]],
                 noise: bool = True,
                 point_num: int = 10000,
                 ):
        self._root: List[Path] = [Path(r).expanduser() for r in root]
        self._noise = noise
        self._point_num = point_num
        self._files = []
        self._file_num = 0
        self.color_jitter = 0.3
        for root_path in self._root:
            object_file_dir = str(root_path)
            object_pcd_files = glob.glob(object_file_dir + "/*npz")
            self._files = self._files + object_pcd_files
            self._file_num = self._file_num + len(object_pcd_files)

    def __len__(self):
        return self._file_num

    def cal_joint_to_part_offset(self, pcd, joint_base, joint_direction):
        joint_axis = joint_direction.reshape((3, 1))
        vec1 = pcd - joint_base
        # project to joint axis
        proj_len = np.dot(vec1, joint_axis)
        # np.clip(proj_len, a_min=self.epsilon, a_max=None, out=proj_len)
        proj_vec = proj_len * joint_axis.transpose()
        orthogonal_vec = - vec1 + proj_vec
        heatmap = np.linalg.norm(orthogonal_vec, axis=1).reshape(-1, 1)
        unitvec = orthogonal_vec / heatmap
        heatmap = 1.0 - heatmap
        heatmap[heatmap < 0] = 0
        proj_vec = orthogonal_vec
        return heatmap, unitvec, proj_vec

    def add_random_noise_to_random_points(self, numpy_point_cloud, max_noise_std=0.03):
        noisy_point_cloud = numpy_point_cloud.copy()
        noise = np.random.normal(0, max_noise_std, size=numpy_point_cloud.shape)
        selected_indices = np.random.choice(noise.shape[0], int(noise.shape[0]*np.random.uniform(0.05, 0.3)), replace=False)
        noisy_point_cloud[selected_indices] = noisy_point_cloud[selected_indices] + noise[selected_indices]
        return noisy_point_cloud

    def simulate_point_cloud_missing_points(self, numpy_point_cloud, missing_probability=0.1):
        missing_mask = np.random.rand(len(numpy_point_cloud)) < missing_probability
        indexs = ~missing_mask
        missing_point_cloud = numpy_point_cloud[indexs]
        return missing_point_cloud, indexs

    def radius_based_denoising_numpy(self, numpy_point_cloud, nb_points=30, radius=0.05):
        if numpy_point_cloud.shape[0] > self._point_num:
            numpy_point_cloud, index_voxel = voxel_sample_points(numpy_point_cloud, point_number=self._point_num)
        else:
            index_voxel = None
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(numpy_point_cloud)
        cl, index_denoise = cloud.remove_statistical_outlier(nb_neighbors=nb_points, std_ratio=1.5)
        numpy_point_cloud = numpy_point_cloud[index_denoise]
        return numpy_point_cloud, index_voxel, index_denoise

    def __getitem__(self, file_id: int, down_sample=True):
        file_id %= self._file_num
        file_path = self._files[file_id]
        cat_id = file_path.split("/")[-1].split("_")[0]
        file = np.load(file_path, allow_pickle=True)
        coord_world = file["per_coord_world"]
        joint_bases = file["joint_bases"]
        joint_axises = file["joint_axises"]
        segment_mask = file["segment_mask"]
        instance_mask = file["instance_mask"]
        rgb = file["per_point_rgb"]
        instance_mask = instance_mask - 1
        assert instance_mask.max() == joint_bases.shape[0]
        if self._noise:
            # coord_world = self.add_random_noise_to_random_points(coord_world)
            coord_world, indexs = self.simulate_point_cloud_missing_points(coord_world)
            segment_mask = segment_mask[indexs]
            instance_mask = instance_mask[indexs]
            rgb = rgb[indexs]
            coord_world, index_org = voxel_sample_points(coord_world, point_number=int(self._point_num*1.5))
            segment_mask = segment_mask[index_org]
            instance_mask = instance_mask[index_org]
            rgb = rgb[index_org] / 255.0
            rgb += np.random.randn(1, 3) * self.color_jitter
            rgbs = np.clip(rgb, 0, 1)
        else:
            coord_world, index_org = voxel_sample_points(coord_world, point_number=int(self._point_num*1.5))
            segment_mask = segment_mask[index_org]
            instance_mask = instance_mask[index_org]
            rgbs = rgb[index_org] / 255.0
        # view_object_joint(coord_world, instance_mask, joint_bases, joint_axises)
        joint_ends = joint_axises + joint_bases
        if self._noise:
            random_scale = np.random.uniform(-0.05, 0.05)
        else:
            random_scale = 0.01
        point_could_center, center, scale = point_cloud_center_and_scale(coord_world, random_scale=random_scale)
        joint_bases = (joint_bases - center) / scale
        joint_ends = (joint_ends - center) / scale
        joint_directions = joint_ends - joint_bases
        joint_directions = joint_directions / np.linalg.norm(joint_directions, axis=1, keepdims=True)
        joint_trans = np.zeros((coord_world.shape[0], 3), dtype=np.float32)
        joint_dirs = np.zeros((coord_world.shape[0], 3), dtype=np.float32)
        joint_offset_unitvecs = np.zeros((coord_world.shape[0], 3), dtype=np.float32)
        joint_offset_heatmaps = np.zeros((coord_world.shape[0], 1), dtype=np.float32)
        joint_proj_vecs = np.zeros((coord_world.shape[0], 3), dtype=np.float32)
        instance_regions = np.zeros((coord_world.shape[0], 9), dtype=np.float32)

        for instance_id in np.unique(instance_mask):
            # print("instance_id: {}".format(instance_id))
            indices = np.where(instance_mask == instance_id)[0]
            if indices.shape[0] == 0:
                print("no instance: {}".format(instance_id))
                continue
            if instance_id == 0:
                joint_trans[indices] = np.array([0, 0, 0])
                joint_dirs[indices] = np.array([0, 0, 0])
                joint_offset_unitvecs[indices] = np.array([0, 0, 0])
                joint_offset_heatmaps[indices] = 0
                joint_proj_vecs[indices] = np.array([0, 0, 0])
            else:
                joint_trans[indices] = joint_bases[instance_id - 1]
                joint_dirs[indices] = joint_directions[instance_id - 1]
                part_pcd = point_could_center[indices, :3]
                heatmap, unitvec, proj_vec = self.cal_joint_to_part_offset(part_pcd, joint_bases[instance_id - 1], joint_directions[instance_id - 1])
                joint_offset_unitvecs[indices] = unitvec
                joint_offset_heatmaps[indices] = heatmap
                joint_proj_vecs[indices] = proj_vec
            xyz_i = point_could_center[indices, :3]
            min_i = xyz_i.min(0)
            max_i = xyz_i.max(0)
            mean_i = xyz_i.mean(0)
            instance_regions[indices, 0:3] = mean_i
            instance_regions[indices, 3:6] = min_i
            instance_regions[indices, 6:9] = max_i
        point_center_offset = instance_regions[:, :3] - point_could_center
        point_cloud_dim_min = point_could_center.min(axis=0)
        point_cloud_dim_max = point_could_center.max(axis=0)
        feat = np.hstack((point_could_center, rgbs))
        return {
            "cat_id": cat_id,
            "coords": point_could_center,
            "scale": scale,
            "center": center,
            "point_center_offsets": point_center_offset,
            "feats": feat,
            "point_num": point_center_offset.shape[0],
            "sem_labels": segment_mask,
            "instance_labels": instance_mask,
            "joint_directions": joint_dirs,
            "joint_proj_vecs": joint_proj_vecs,
            "file_id": file_path,
            "point_cloud_dim_min": point_cloud_dim_min,
            "point_cloud_dim_max": point_cloud_dim_max,
        }

if __name__ == '__main__':
    # root = ["/hdd/gapartnet_drawer_door_handle"]
    root = ["/data/Pl/multi_view_cabinet_rgb_test/"]
    dataset = GammaDataset(root=root)
    print("data total: ", dataset.__len__())
    for i in range(dataset.__len__()):
        print("data num: ", i)
        data = dataset.__getitem__(i, down_sample=True)
        # print(data["coord"].shape)
        # print("heat_map_max: ", data["joint_offset_heatmaps"].max(),  data["joint_offset_heatmaps"].min(), data["scale"])


