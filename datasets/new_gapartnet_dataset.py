import pdb
import time
from pathlib import Path
import glob
import torch.utils.data as data
from typing import Union, List, Tuple
import torch
import numpy as np
import copy
import json
import open3d as o3d
import shutil
from tqdm import tqdm
import os
from .data_utilts import point_cloud_center_and_scale, translate_joint_base_world_to_camera, translate_joint_direc_world_to_camera, sample_point, \
    translate_pc_world_to_camera, view_point_cloud_parts_and_joints, view_point_cloud_parts_and_center, voxel_sample_points, view_point_cloud_parts
from .config import NEW_TARGET_GAPART_SEM_IDS_TO_NEW_ids, NEW_TARGET_GAPARTS

class NewGAPartNetDataset(data.Dataset):
    def __init__(self,
                 root_dir: str = "",
                 ancsh_root_dir: str = "",
                 noise: bool = False,
                 point_num: int = 20000,
                 data_type: str = "val",
                 data_check: bool = False,
                 ancsh: bool = False,
                 ):
        self._root_dir = root_dir
        self._ancsh_root_dir = ancsh_root_dir
        self._noise = noise

        self._file_num = 0
        self._point_num = int(point_num*1.25)
        self._data_type = data_type
        self._files = []
        self._ancsh_files = []
        self._file_num = 0
        self.color_jitter = 0.3
        self.data_check = data_check
        self.ancsh = ancsh
        with open("./datasets/new_train_test_cat.json", "r") as fp:
            self.new_cat_data = json.load(fp)
            new_object_files, ancsh_files = self.get_file_path(self.new_cat_data, "new_objects")
            self._files += new_object_files
            if ancsh_files is not None:
                self._ancsh_files += ancsh_files
        with open("./datasets/akb48_train_test_cat.json", "r") as fp:
            self.akb48_cat_data = json.load(fp)
            akb48_object_files, ancsh_files = self.get_file_path(self.akb48_cat_data, "akb48_objects")
            self._files += akb48_object_files
            if ancsh_files is not None:
                self._ancsh_files += ancsh_files
        with open("./datasets/ori_train_test_cat.json", "r") as fp:
            self.ori_cat_data = json.load(fp)
            ori_object_files, ancsh_files= self.get_file_path(self.ori_cat_data, "origin_objects")
            self._files += ori_object_files
            if ancsh_files is not None:
                self._ancsh_files += ancsh_files

        if self.ancsh:
            assert len(self._ancsh_files) == self._file_num
        self._data_check()
        self._file_num = len(self._files)
        print("data num:", self._file_num, " data type:", self._data_type)

    def __len__(self):
        return self._file_num

    def _data_check(self):
        correct_files = []
        for idx in range(len(self._files)):
            file_path = self._files[idx]
            if not os.path.exists(file_path):
                continue
            if not os.path.exists(file_path + "/data/object_part_joint.npz"):
                shutil.rmtree(file_path)
            file = np.load(file_path + "/data/object_part_joint.npz", allow_pickle=True)
            segment_mask = file["segment_mask"]
            instance_mask = file["instance_mask"]
            coord_world = file["per_coord_world"]
            if (instance_mask > 0).sum() < 500:
                continue

            for sem_id in np.unique(segment_mask):
                if sem_id == 0:
                    continue
                else:
                    segment_mask[segment_mask == sem_id] = NEW_TARGET_GAPART_SEM_IDS_TO_NEW_ids[sem_id]
            for instance_id in np.unique(instance_mask):
                part_num = (instance_mask == instance_id).sum()
                if part_num > 100:
                    continue
                else:
                    if part_num < 50:
                        break
                    instance_type = segment_mask[instance_mask == instance_id][0]
                    type_name = NEW_TARGET_GAPARTS[int(instance_type - 1)]
                    if type_name in ["slider_drawer", "hinge_door", "slider_lid", " revolute_seat", "hinge_lid"]:
                        break
            correct_files.append(file_path)
        self._files = correct_files
        self._file_num = len(self._files)
        print("*" * 100)
        print("data check finished")
        print("data type:{}, num:{}".format(self._data_type, self._file_num))

    def get_file_path(self, json_file, type_file_name="akb48_objects"):
        object_files = []
        if self.ancsh:
            ancsh_files = []
        if self._data_type == "train":
            object_datas = json_file["seen_category"]
            for object_cat in object_datas:
                cat_ids = object_datas[object_cat]["seen"]
                for cat_id in cat_ids:
                    files = glob.glob(self._root_dir + "/" + type_file_name + "/" + object_cat + "_" + str(cat_id) + "/*")
                    if len(files) == 0:
                        object_cat = object_cat.lower()
                        files = glob.glob(self._root_dir + "/" + type_file_name + "/" + object_cat + "_" + str(cat_id) + "/*")
                    for file in files:
                        scene_id = int(file.split("/")[-1].split("_")[0])
                        if scene_id < 16:
                            object_files.append(file)
                            if self.ancsh:
                                cat_type = file.split("/")[-3]
                                cat_ids = file.split("/")[-2]
                                ancsh_files.append(self._ancsh_root_dir + "/" + cat_type + "/" + cat_ids + "/0_0/" )

        if self._data_type == "val":
            object_datas = json_file["seen_category"]
            for object_cat in object_datas:
                cat_ids = object_datas[object_cat]["seen"]
                for cat_id in cat_ids:
                    files = glob.glob(self._root_dir + "/" + type_file_name + "/" + object_cat + "_" + str(cat_id) + "/*")
                    if len(files) == 0:
                        object_cat = object_cat.lower()
                        files = glob.glob(self._root_dir + "/" + type_file_name + "/" + object_cat + "_" + str(cat_id) + "/*")
                    for file in files:
                        scene_id = int(file.split("/")[-1].split("_")[0])
                        if scene_id >= 17:
                            object_files.append(file)
                            if self.ancsh:
                                cat_type = file.split("/")[-3]
                                cat_ids = file.split("/")[-2]
                                ancsh_files.append(self._ancsh_root_dir + "/" + cat_type + "/" + cat_ids + "/0_0/")

        if self._data_type == "seen_category":
            object_datas = json_file["seen_category"]
            for object_cat in object_datas:
                cat_ids = object_datas[object_cat]["unseen"]
                for cat_id in cat_ids:
                    files = glob.glob(self._root_dir + "/" + type_file_name + "/" + object_cat + "_" + str(cat_id) + "/*")
                    if len(files) == 0:
                        object_cat = object_cat.lower()
                        files = glob.glob(self._root_dir + "/" + type_file_name + "/" + object_cat + "_" + str(cat_id) + "/*")
                    if self.ancsh:
                        for file in files:
                            cat_type = file.split("/")[-3]
                            cat_ids = file.split("/")[-2]
                            ancsh_files.append(self._ancsh_root_dir + "/" + cat_type + "/" + cat_ids + "/0_0/")
                    object_files += files

        if self._data_type == "unseen_category":
            object_datas = json_file["unseen_category"]
            for object_cat in object_datas:
                cat_ids = object_datas[object_cat]
                for cat_id in cat_ids:
                    files = glob.glob(self._root_dir + "/" + type_file_name + "/" + object_cat + "_" + str(cat_id) + "/*")
                    if len(files) == 0:
                        object_cat = object_cat.lower()
                        files = glob.glob(self._root_dir + "/" + type_file_name + "/" + object_cat + "_" + str(cat_id) + "/*")
                    if self.ancsh:
                        for file in files:
                            cat_type = file.split("/")[-3]
                            cat_ids = file.split("/")[-2]
                            ancsh_files.append(self._ancsh_root_dir + "/" + cat_type + "/" + cat_ids + "/0_0/")
                    object_files += files
        if self.ancsh:
            return object_files, ancsh_files
        else:
            return object_files, None

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

    def cal_query_point_heatmap(self, part_pcd, threshold=0.8, decay_rate=2.0):
        part_center = np.mean(part_pcd, axis=0)
        center_point = part_pcd[np.argmin(np.linalg.norm(part_pcd - part_center, axis=1))]
        part_to_center = np.linalg.norm(part_pcd - center_point, axis=1)
        heatmap_max = part_to_center.max()
        query_heatmap = part_to_center / (heatmap_max + 1e-6)
        query_heatmap = np.exp(-decay_rate*query_heatmap)
        query_heatmap = query_heatmap.reshape(-1, 1)
        query_heatmap[query_heatmap >= threshold] = threshold
        query_heatmap[query_heatmap < threshold] = 0
        return query_heatmap


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
        object_cat_id = file_path.split("/")[-2]
        if not os.path.exists(file_path + "/data/object_part_joint.npz"):
            shutil.rmtree(file_path, ignore_errors=True)
            print("remove empty directory: ", file_path)
            return
        file = np.load(file_path + "/data/object_part_joint.npz", allow_pickle=True)
        coord_world = file["per_coord_world"]
        joint_bases = file["joint_bases"]
        joint_axises = file["joint_axises"]
        segment_mask = file["segment_mask"]
        instance_mask = file["instance_mask"]
        per_point_npcs = file["per_point_npcs"]
        per_point_rgb = file["per_point_rgb"]
        extrinsic = file["extrinsic"]
        num_instances = np.unique(instance_mask).shape[0]
        if self.ancsh:
            ancsh_json_file_path = self._ancsh_files[file_id] + "/object_and_part_bboxs.json"
            with open(ancsh_json_file_path, "r") as f:
                ancsh_json_file = json.load(f)
            select_part_link_name = ancsh_json_file["select_part_link_names"]
            part_reset_bboxs = ancsh_json_file["object_and_part_bboxs"][:-1]
            object_reset_bbox = ancsh_json_file["object_and_part_bboxs"][-1]
            if not len(select_part_link_name) == len(part_reset_bboxs) == len(joint_bases) == len(joint_axises) == (num_instances - 1):
                print("select_part_link_name: ", len(select_part_link_name))
                print("joint_bases: ",  len(joint_bases))
                print("num_instances: ", num_instances)
                if self.data_check:
                    shutil.rmtree(file_path, ignore_errors=True)
                    return None
        else:
            assert len(joint_bases) == len(joint_axises) == (num_instances - 1)

        if self.data_check:
            if not coord_world.shape[0] == per_point_rgb.shape[0] == segment_mask.shape[0] == instance_mask.shape[0] == segment_mask.shape[0] == per_point_npcs.shape[0]:
                print("point num errors data: ", file_path)
                shutil.rmtree(file_path, ignore_errors=True)
                return None

        # change semantic id
        for sem_id in np.unique(segment_mask):
            if sem_id == 0:
                continue
            else:
                segment_mask[segment_mask == sem_id] = NEW_TARGET_GAPART_SEM_IDS_TO_NEW_ids[sem_id]

        if self._noise:
            coord_world, indexs = self.simulate_point_cloud_missing_points(coord_world)
            segment_mask = segment_mask[indexs]
            instance_mask = instance_mask[indexs]
            per_point_npcs = per_point_npcs[indexs]
            per_point_rgb = per_point_rgb[indexs]
            # noise
            coord_world, index_org = voxel_sample_points(coord_world, point_number=int(self._point_num))
            segment_mask = segment_mask[index_org]
            instance_mask = instance_mask[index_org]
            per_point_rgb = per_point_rgb[index_org] / 255.0
            per_point_rgb += np.random.randn(1, 3) * self.color_jitter
            per_point_rgb = np.clip(per_point_rgb, 0, 1)
            per_point_npcs = per_point_npcs[index_org]
        else:
            coord_world, index_org = voxel_sample_points(coord_world, point_number=int(self._point_num))
            segment_mask = segment_mask[index_org]
            instance_mask = instance_mask[index_org]
            per_point_rgb = per_point_rgb[index_org] / 255.0
            per_point_rgb += np.random.randn(1, 3) * self.color_jitter
            per_point_rgb = np.clip(per_point_rgb, 0, 1)
            per_point_npcs = per_point_npcs[index_org]
        joint_ends = joint_axises + joint_bases
        pcd_camera = translate_pc_world_to_camera(coord_world, extrinsic)
        joint_bases = translate_joint_base_world_to_camera(joint_bases, extrinsic)
        joint_ends = translate_joint_direc_world_to_camera(joint_ends, extrinsic)
        if self._noise:
            random_scale = np.random.uniform(-0.05, 0.05)
        else:
            random_scale = 0.01
        point_could_center, center, scale = point_cloud_center_and_scale(pcd_camera, random_scale=random_scale)
        joint_bases = (joint_bases - center) / scale
        joint_ends = (joint_ends - center) / scale
        joint_directions = joint_ends - joint_bases
        joint_directions = joint_directions / np.linalg.norm(joint_directions, axis=1, keepdims=True)
        num_instances = int(instance_mask.max()) + 1
        joint_trans = np.zeros((pcd_camera.shape[0], 3), dtype=np.float32)
        joint_dirs = np.zeros((pcd_camera.shape[0], 3), dtype=np.float32)
        joint_offset_unitvecs = np.zeros((pcd_camera.shape[0], 3), dtype=np.float32)
        joint_offset_heatmaps = np.zeros((pcd_camera.shape[0], 1), dtype=np.float32)
        joint_proj_vecs = np.zeros((pcd_camera.shape[0], 3), dtype=np.float32)
        instance_regions = np.zeros((pcd_camera.shape[0], 9), dtype=np.float32)
        query_heatmaps = np.zeros((pcd_camera.shape[0], 1), dtype=np.float32)
        for instance_id in range(num_instances):
            indices = np.where(instance_mask == instance_id)[0]
            if indices.shape[0] == 0:
                # print("no instance: {}".format(instance_id))
                if self.data_check:
                    if instance_id > 0:
                        print("no instance error: {}".format(file_path))
                        shutil.rmtree(file_path, ignore_errors=True)
                    return None
                continue
            if instance_id == 0:
                joint_trans[indices] = np.array([0, 0, 0])
                joint_dirs[indices] = np.array([0, 0, 0])
                joint_offset_unitvecs[indices] = np.array([0, 0, 0])
                joint_offset_heatmaps[indices] = 0
                joint_proj_vecs[indices] = np.array([0, 0, 0])
                query_heatmaps[indices] = 0
            else:
                joint_trans[indices] = joint_bases[instance_id - 1]
                joint_dirs[indices] = joint_directions[instance_id - 1]
                part_pcd = point_could_center[indices, :3]
                heatmap, unitvec, proj_vec = self.cal_joint_to_part_offset(part_pcd, joint_bases[instance_id - 1], joint_directions[instance_id - 1])
                part_query_heatmap = self.cal_query_point_heatmap(part_pcd, threshold=0.7)
                joint_offset_unitvecs[indices] = unitvec
                joint_offset_heatmaps[indices] = heatmap
                joint_proj_vecs[indices] = proj_vec
                query_heatmaps[indices] = part_query_heatmap
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
        segment_mask = segment_mask.astype(np.int64)
        instance_mask = instance_mask.astype(np.int64)
        # view_point_cloud_heatmap(point_cloud=point_could_center, heat_map=query_heatmaps)
        # view_point_cloud_parts_and_joints(point_cloud=point_could_center, mask=instance_mask,
        #                                   joint_directions=joint_dirs, joint_proj_vecs=joint_proj_vecs,
        #                                   point_to_center=point_center_offset)
        assert point_could_center.shape[0] == per_point_rgb.shape[0] == segment_mask.shape[0] == instance_mask.shape[0] == joint_dirs.shape[0] == joint_proj_vecs.shape[0]
        feat = np.concatenate([point_could_center, per_point_rgb], axis=-1)
        return {
            "coords": point_could_center,
            "per_point_npcs": per_point_npcs,
            "scales": scale,
            "centers": center,
            "point_center_offsets": point_center_offset,
            "feats": feat,
            "point_nums": point_center_offset.shape[0],
            "sem_labels": segment_mask,
            "instance_labels": instance_mask,
            "joint_directions": joint_dirs,
            "joint_offset_heatmaps": joint_offset_heatmaps,
            "joint_proj_vecs": joint_proj_vecs,
            "cat_id": object_cat_id,
            "point_cloud_dim_min": point_cloud_dim_min,
            "point_cloud_dim_max": point_cloud_dim_max,
            "query_heatmaps": query_heatmaps
        }

if __name__ == '__main__':
    root = "/data/gapartnet_data/"
    ancsh_root = "/data/gapartnet_reset_bbox"
    # ancsh_root = "/aidata/qiaojun/train_data/gapartnet_data/gapartnet_reset_bbox/"
    # root = "/aidata/qiaojun/train_data/gapartnet_data/"
    data_types = ["train", "val", "seen_category", "unseen_category"]
    for data_type in data_types:
        dataset = NewGAPartNetDataset(root_dir=root, ancsh_root_dir=ancsh_root,ancsh=False, data_type=data_type, noise=False, point_num=20000, data_check=True)
        print("data total: ", dataset.__len__())
        time.sleep(10)
        sem_labels = []
        for id in range(dataset.__len__()):
            data = dataset.__getitem__(id)
            print("data type: {} data id: {} ".format(data_type, id))
            if data is None:
                continue
            sem_label = data["sem_labels"]
            for id in np.unique(sem_label):
                if id not in sem_labels:
                    sem_labels.append(id)
                    print("sem labels: ", sem_labels)



