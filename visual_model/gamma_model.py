import pdb

from .pointnet2 import Point2SemSegSSGNetfeat, Point2SemSegMSGNetfeat
import torch.nn as nn
import torch
from .losses import iou, focal_loss, dice_loss, iou_evel
import numpy as np
import collections
from scipy.optimize import linear_sum_assignment
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from .pointnet2_utils import index_points, index_masks
from .pcd_utils import point_cloud_center_and_scale, radius_based_denoising_numpy, view_point_cloud_parts_and_joint, \
    voxel_sample_points, view_point_cloud_parts


class gamma_model_net(nn.Module):
    def __init__(self, pointnet_type, point_dim=3, num_classes=3, num_point=10000, device="cuda"):
        super(gamma_model_net, self).__init__()
        self.pointnet_type = pointnet_type
        self.point_dim = point_dim
        self.num_classes = num_classes
        self.num_point = num_point
        self.device = device
        self.point_feat = Point2SemSegMSGNetfeat(in_channel=self.point_dim)

        self.sem_head = nn.Sequential(
            nn.Conv1d(128, self.num_classes, kernel_size=1, padding=0, bias=False))

        self.offset_feature = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, padding=0, bias=False),
            nn.ReLU(True),
            nn.Conv1d(128, 128, kernel_size=1, padding=0, bias=False),
            nn.ReLU(True)
        )
        self.offset_head = nn.Linear(128, 3)
        self.joint_feature = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, padding=0, bias=False),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv1d(128, 128, kernel_size=1, padding=0, bias=False),
            nn.ReLU(True),
        )
        self.joint_direction = nn.Linear(128, 3)
        self.joint_proj_vec = nn.Linear(128, 3)

    def forward(self, point_cloud):
        pc_feature = self.point_feat(point_cloud)
        sem_logits = self.sem_head(pc_feature).transpose(1, 2)
        point_center_offsets_feature = self.offset_feature(pc_feature).transpose(1, 2)
        point_center_offsets = torch.tanh(self.offset_head(point_center_offsets_feature))
        joint_feature = self.joint_feature(pc_feature).transpose(1, 2)
        joint_proj_vecs = torch.tanh(self.joint_proj_vec(joint_feature))
        joint_directions = torch.tanh(self.joint_direction(joint_feature))
        pre_data = dict()
        pre_data["sem_logits"] = sem_logits
        pre_data["point_center_offsets"] = point_center_offsets
        pre_data["joint_directions"] = joint_directions
        pre_data["joint_proj_vecs"] = joint_proj_vecs
        pre_data["point_center_offsets_feature"] = point_center_offsets_feature
        pre_data["joint_feature"] = joint_feature
        pre_data["pc_feature"] = pc_feature.transpose(1, 2)
        return pre_data

    def offset_loss(self, pre_offset, gt_offsets, valid_mask):
        gt_offsets = gt_offsets.reshape(-1, 3)
        pre_offset = pre_offset.reshape(-1, 3)
        pt_diff = pre_offset - gt_offsets
        pt_dist = torch.sum(pt_diff.abs(), dim=-1)
        loss_pt_offset_dist = pt_dist[valid_mask].mean()
        gt_offsets_norm = torch.norm(gt_offsets, dim=1).reshape(-1, 1)
        gt_offsets = gt_offsets / (gt_offsets_norm + 1e-8)
        pre_offsets_norm = torch.norm(pre_offset, dim=1).reshape(-1, 1)
        pre_offset = pre_offset / (pre_offsets_norm + 1e-8)
        dir_diff = -(gt_offsets * pre_offset).sum(dim=-1)
        loss_offset_dir = dir_diff[valid_mask].mean()
        loss_offset = loss_offset_dir + loss_pt_offset_dist
        return loss_offset

    def get_offset_and_direction_orthogonal_points(self, pre_pro_offset, pre_direction, deg_threshold=5):
        pre_pro_offset = pre_pro_offset.reshape(-1, 3)
        pre_direction = pre_direction.reshape(-1, 3)
        dot_product = np.sum(pre_pro_offset * pre_direction, axis=1)
        norms_product = np.linalg.norm(pre_pro_offset, axis=1) * np.linalg.norm(pre_direction, axis=1)
        cosine_of_angle = dot_product / norms_product
        angles_rad = np.arccos(cosine_of_angle)
        angles_deg = abs(np.rad2deg(angles_rad) - 90)
        orthogonal_point_indexs = angles_deg < deg_threshold
        return orthogonal_point_indexs

    def offset_and_direction_orthogonal_loss(self, pre_pro_offset, pre_direction, valid_mask):
        pre_pro_offset = pre_pro_offset.reshape(-1, 3)
        pre_direction = pre_direction.reshape(-1, 3)
        dot_product = torch.sum(pre_pro_offset * pre_direction, dim=1)
        dot_product_abs = torch.abs(dot_product)
        orthogonal_loss = dot_product_abs[valid_mask].mean()
        norms_product = torch.norm(pre_pro_offset, dim=1) * torch.norm(pre_direction, dim=1)
        cosine_of_angle = dot_product / norms_product
        angles_rad = torch.acos(cosine_of_angle[valid_mask])
        angles_deg = torch.rad2deg(angles_rad)
        return orthogonal_loss, angles_deg

    def loss_sem_seg(self, sem_logits: torch.Tensor, sem_labels: torch.Tensor, ) -> torch.Tensor:
        loss = focal_loss(sem_logits, sem_labels, alpha=None, gamma=2.0, reduction="mean")
        loss += dice_loss(sem_logits[:, :, None, None], sem_labels[:, None, None], )
        return loss

    def get_loss(self, data_dict, sem_only=False, ignore_label=0):
        coord = data_dict["coords"].to(self.device, dtype=torch.float)
        if coord.shape[1] > self.num_point:
            fps_pcs_idx = furthest_point_sample(coord, self.num_point).cpu().numpy()
            feat_per_point = index_points(data_dict["feats"], fps_pcs_idx).transpose(1, 2).to(self.device, dtype=torch.float)
            data_dict["sem_labels"] = index_masks(data_dict["sem_labels"], fps_pcs_idx)
            data_dict["point_center_offsets"] = index_points(data_dict["point_center_offsets"], fps_pcs_idx)
            data_dict["joint_directions"] = index_points(data_dict["joint_directions"], fps_pcs_idx)
            data_dict["joint_proj_vecs"] = index_points(data_dict["joint_proj_vecs"], fps_pcs_idx)
        else:
            feat_per_point = data_dict["feats"].transpose(1, 2).to(self.device, dtype=torch.float)
        pred = self.forward(feat_per_point)
        valid_mask = data_dict["sem_labels"] != ignore_label
        valid_mask = valid_mask.reshape(-1)
        # part sem
        sem_mask_loss = self.loss_sem_seg(pred["sem_logits"].reshape(-1, self.num_classes),
                                               data_dict["sem_labels"].reshape(-1).to(self.device, dtype=torch.long))
        point_center_offset_loss = self.offset_loss(pred["point_center_offsets"],
                                                    data_dict["point_center_offsets"].to(self.device, dtype=torch.float),
                                                    valid_mask=valid_mask)
        # joint pose
        joint_direction_loss = self.offset_loss(pred["joint_directions"],
                                                data_dict["joint_directions"].to(self.device, dtype=torch.float),
                                               valid_mask=valid_mask)
        joint_proj_vec_loss = self.offset_loss(pred["joint_proj_vecs"],
                                               data_dict["joint_proj_vecs"].to(self.device, dtype=torch.float),
                                               valid_mask=valid_mask)
        orthogonal_loss, angles_deg = self.offset_and_direction_orthogonal_loss(pred["joint_proj_vecs"],
                                                                                pred["joint_directions"],
                                                                                valid_mask=valid_mask)
        # error
        if sem_only:
            total_loss = sem_mask_loss
        else:
            total_loss = sem_mask_loss + point_center_offset_loss + joint_direction_loss + joint_proj_vec_loss
        loss_dict = dict()
        result_dict = dict()
        sem_preds = torch.argmax(pred["sem_logits"].reshape(-1, self.num_classes).detach(), dim=-1)
        sem_all_accu = (sem_preds == data_dict["sem_labels"].reshape(-1).to(self.device, dtype=torch.long)).sum().float() / (sem_preds.shape[0])
        loss_dict["sem_loss"] = sem_mask_loss
        loss_dict["total_loss"] = total_loss
        loss_dict["point_center_offset_loss"] = point_center_offset_loss
        loss_dict["joint_direction_loss"] = joint_direction_loss
        loss_dict["joint_proj_vec_loss"] = joint_proj_vec_loss
        loss_dict["orthogonal_loss"] = orthogonal_loss
        loss_dict["orthogonal_angles_deg"] = (90 - angles_deg).mean()
        result_dict["sem_all_accu"] = sem_all_accu
        return loss_dict, result_dict

    def offset_angle(self, pre_offsets, gt_offsets, function_mask):
        gt_offsets_norm = torch.norm(gt_offsets, dim=2).reshape(-1, 1)
        gt_offsets = gt_offsets.reshape(-1, 3) / (gt_offsets_norm + 1e-8)
        pre_offsets_norm = torch.norm(pre_offsets, dim=2).reshape(-1, 1)
        offsets = pre_offsets.reshape(-1, 3) / (pre_offsets_norm + 1e-8)
        function_mask = function_mask.reshape(-1)
        dir_diff = (gt_offsets * offsets).sum(dim=-1)
        cos_theta = torch.clip(dir_diff, -1.0, 1.0)
        cos_theta = cos_theta[function_mask < 2]
        angles_rad = torch.acos(cos_theta)
        angles_deg = torch.rad2deg(angles_rad)
        return angles_deg

    def joint_direction_angle(self, pre_directions, gt_directions, function_mask):
        gt_directions_norm = torch.norm(gt_directions, dim=2).reshape(-1, 1)
        gt_directions = gt_directions.reshape(-1, 3) / (gt_directions_norm + 1e-8)
        pre_directions_norm = torch.norm(pre_directions, dim=2).reshape(-1, 1)
        pre_directions = pre_directions.reshape(-1, 3) / (pre_directions_norm + 1e-8)
        function_mask = function_mask.reshape(-1)
        dir_diff = (pre_directions * gt_directions).sum(dim=-1)
        cos_theta = torch.clip(dir_diff, -1.0, 1.0)
        cos_theta = cos_theta[function_mask < 2]
        angles_rad = torch.acos(cos_theta)
        angles_deg = torch.rad2deg(angles_rad)
        return angles_deg

    def joint_offset_direction_angle(self, pre_directions, gt_directions, function_mask):
        gt_directions_norm = torch.norm(gt_directions, dim=2).reshape(-1, 1)
        gt_directions = gt_directions.reshape(-1, 3) / (gt_directions_norm + 1e-8)
        pre_directions_norm = torch.norm(pre_directions, dim=2).reshape(-1, 1)
        pre_directions = pre_directions.reshape(-1, 3) / (pre_directions_norm + 1e-8)
        function_mask = function_mask.reshape(-1)
        dir_diff = (pre_directions * gt_directions).sum(dim=-1)
        cos_theta = torch.clip(dir_diff, -1.0, 1.0)
        cos_theta = cos_theta[function_mask < 2]
        angles_rad = torch.acos(cos_theta)
        angles_deg = torch.rad2deg(angles_rad)
        return angles_deg

    @torch.no_grad()
    def online_inference(self, camera_pcd, view_res=False, denoise=True, cluster_eps=0.1, num_point_min=500,
                         joint_type_to_name=True):
        if denoise:
            camera_pcd = radius_based_denoising_numpy(camera_pcd)
        camera_pcd, voxel_centroids = voxel_sample_points(camera_pcd, point_number=int(self.num_point * 1.5))
        point_cloud, center, scale = point_cloud_center_and_scale(camera_pcd)
        camcs_per_point = torch.from_numpy(point_cloud).to(self.device, dtype=torch.float).unsqueeze(0)
        fps_pcs_idx = furthest_point_sample(camcs_per_point, self.num_point).cpu().numpy()
        camera_pcd = camera_pcd[fps_pcs_idx[0]]
        camcs_per_point = index_points(camcs_per_point, fps_pcs_idx).transpose(1, 2)
        pred = self.forward(camcs_per_point)
        function_mask = torch.argmax(pred["function_mask"].reshape(-1, self.num_classes).detach(), dim=-1)
        function_mask = function_mask.detach().cpu().numpy()
        pre_offset = pred["point_center_offset"].detach().cpu().numpy()[0]
        pre_joint_offset_heatmap = pred["joint_offset_heatmaps"].detach().cpu().numpy()[0]
        pre_joint_offset_direction = pred["joint_offset_unitvecs"].detach().cpu().numpy()[0]
        pre_joint_direction = pred["joint_directions"].detach().cpu().numpy()[0]
        point_cloud = camcs_per_point.detach().cpu().numpy()[0].transpose(1, 0)
        function_mask_ids = np.unique(function_mask)
        results = []
        instance_id = 0
        instance_labels = np.zeros_like(function_mask)
        for function_mask_id in function_mask_ids:
            if function_mask_id == 2:
                continue
            if (function_mask == function_mask_id).sum() < 30:
                continue
            part_indexs = np.where(function_mask == function_mask_id)[0]
            part_pcd = point_cloud[function_mask == function_mask_id]
            pre_part_joint_offset_direction = pre_joint_offset_direction[function_mask == function_mask_id]
            pre_part_joint_offset_heatmap = pre_joint_offset_heatmap[function_mask == function_mask_id]
            pre_part_joint_direction = pre_joint_direction[function_mask == function_mask_id]
            pre_pcd_to_joint = part_pcd + (1 - pre_part_joint_offset_heatmap) * pre_part_joint_offset_direction
            pre_part_offset = pre_offset[function_mask == function_mask_id]
            pre_pcd_to_center = part_pcd + pre_part_offset
            pcd_feature = np.hstack((pre_pcd_to_joint, pre_pcd_to_center))
            part_labels = self.sklearn_cluster(pcd_feature, eps=cluster_eps)
            part_num = part_labels.max() + 1
            part_instance_mask = np.zeros_like(part_labels)
            for part_id in range(part_num):
                pre_joint_axis = np.median(pre_part_joint_direction[part_labels == part_id], axis=0)
                pre_joint_axis = pre_joint_axis / np.linalg.norm(pre_joint_axis)
                pred_joint_pts = pre_pcd_to_joint[part_labels == part_id]
                pred_joint_pt = np.median(pred_joint_pts, axis=0)
                joint_end = pred_joint_pt + pre_joint_axis
                pred_joint_pt = pred_joint_pt * scale + center
                joint_end = joint_end * scale + center
                pre_joint_axis = joint_end - pred_joint_pt
                pre_joint_axis = pre_joint_axis / np.linalg.norm(pre_joint_axis)
                instance_id += 1
                part_instance_mask[part_labels == part_id] = instance_id
                if (part_labels == part_id).sum() > num_point_min:
                    if joint_type_to_name:
                        if function_mask_id == 0:
                            joint_type = "revolute"
                        if function_mask_id == 1:
                            joint_type = "primastic"
                    else:
                        joint_type = function_mask_id
                    results.append(
                        {"joint_type": joint_type, "joint_translation": pred_joint_pt, "instance_id": instance_id,
                         "joint_direction": pre_joint_axis, "part_point_num": (part_labels == part_id).sum()})
            instance_labels[part_indexs] = part_instance_mask
        if view_res:
            view_point_cloud_parts_and_joint(camera_pcd, instance_labels, results)
        return results, instance_labels, camera_pcd

    @torch.no_grad()
    def evaluate(self, data_dict, cluster="joint_and_center", ignore_label=0):
        coord = data_dict["coords"].to(self.device, dtype=torch.float)
        if coord.shape[1] > self.num_point:
            fps_pcs_idx = furthest_point_sample(coord, self.num_point).cpu().numpy()
            feat_per_point = index_points(data_dict["feats"], fps_pcs_idx).transpose(1, 2).to(self.device, dtype=torch.float)
            data_dict["sem_labels"] = index_masks(data_dict["sem_labels"], fps_pcs_idx)
            data_dict["instance_labels"] = index_masks(data_dict["instance_labels"], fps_pcs_idx)
            data_dict["point_center_offsets"] = index_points(data_dict["point_center_offsets"], fps_pcs_idx)
            data_dict["joint_directions"] = index_points(data_dict["joint_directions"], fps_pcs_idx)
            data_dict["joint_proj_vecs"] = index_points(data_dict["joint_proj_vecs"], fps_pcs_idx)
            data_dict["coords"] = index_points(data_dict["coords"], fps_pcs_idx)
        else:
            feat_per_point = data_dict["feats"].transpose(1, 2).to(self.device, dtype=torch.float)
        coord_per_point = data_dict["coords"]
        pred = self.forward(feat_per_point)
        valid_mask = data_dict["sem_labels"] != ignore_label
        valid_mask = valid_mask.reshape(-1)
        # finish sample
        object_cat_id = data_dict["file_id"][0].split("/")[-1].split(".")[0]
        scale = data_dict["scale"][0].cpu().numpy()
        center = data_dict["center"][0].cpu().numpy()

        sem_preds = torch.argmax(pred["sem_logits"].reshape(-1, self.num_classes).detach(), dim=-1)
        sem_preds = sem_preds.detach().cpu().numpy()

        pre_function_mask = torch.argmax(pred["sem_logits"].reshape(-1, self.num_classes).detach(), dim=-1)
        pre_function_mask = pre_function_mask.detach().cpu().numpy()
        pre_joint_proj_vecs = pred["joint_proj_vecs"].detach().cpu().numpy()[0]
        pre_joint_directions = pred["joint_directions"].detach().cpu().numpy()[0]
        pre_point_center_offsets = pred["point_center_offsets"].detach().cpu().numpy()[0]

        gt_instance_labels = data_dict["instance_labels"].detach().cpu().numpy()[0]
        gt_function_masks = data_dict["sem_labels"].detach().cpu().numpy()[0]
        gt_joint_proj_vecs = data_dict["joint_proj_vecs"].detach().cpu().numpy()[0]
        gt_joint_directions = data_dict["joint_directions"].detach().cpu().numpy()[0]
        pcd = coord_per_point.detach().cpu().numpy()[0]
        # cal error
        sem_accu = (sem_preds == gt_function_masks).sum() / gt_function_masks.shape[0]
        pixel_accu = (sem_preds[valid_mask] == gt_function_masks[valid_mask]).sum() / gt_function_masks[valid_mask].shape[0]
        result = []
        angle_errors = {}
        trans_errors = {}
        ious = {}
        joint_type_accus = {}
        instance_errors = dict()
        pre_instance_id = 0
        pre_instance_labels = np.zeros_like(pre_function_mask)
        function_mask_ids = np.unique(pre_function_mask)
        for function_mask_id in function_mask_ids:
            if function_mask_id == ignore_label:
                continue
            if (pre_function_mask == function_mask_id).sum() < 30:
                continue
            part_indexs = np.where(pre_function_mask == function_mask_id)[0]
            part_pcd = pcd[pre_function_mask == function_mask_id]
            gt_instance_parts = gt_instance_labels[pre_function_mask == function_mask_id]
            pre_part_point_center_offsets = pre_point_center_offsets[pre_function_mask == function_mask_id]
            pre_part_joint_proj_vecs = pre_joint_proj_vecs[pre_function_mask == function_mask_id]
            pre_part_joint_directions = pre_joint_directions[pre_function_mask == function_mask_id]
            pre_pcd_to_center = part_pcd + pre_part_point_center_offsets
            pre_pcd_to_joint = part_pcd + pre_part_joint_proj_vecs
            if cluster == "joint_and_center":
                pcd_feature = np.hstack((pre_pcd_to_joint, pre_pcd_to_center))
            if cluster == "center":
                pcd_feature = pre_pcd_to_center
            if cluster == "joint":
                pcd_feature = pre_pcd_to_joint
            part_labels = self.sklearn_cluster(pcd_feature, eps=0.08)
            part_num = part_labels.max() + 1
            part_instance_mask = np.zeros_like(part_labels)
            for part_id in range(part_num):
                pre_instance_id = pre_instance_id + 1
                pre_joint_axis = np.median(pre_part_joint_directions[part_labels == part_id], axis=0)
                pre_joint_axis = pre_joint_axis / np.linalg.norm(pre_joint_axis)
                pred_joint_pts = pre_pcd_to_joint[part_labels == part_id]
                pred_joint_pt = np.median(pred_joint_pts, axis=0)
                result.append({"joint_type": function_mask_id, "joint_translation": pred_joint_pt,
                               "joint_direction": pre_joint_axis})
                gt_instance_part = gt_instance_parts[part_labels == part_id]
                instance_counter = collections.Counter(gt_instance_part)
                instance_keys = instance_counter.keys()
                max_part_num = 0
                instance_id = 0
                for instance_key in instance_keys:
                    part_num = instance_counter[instance_key]
                    if part_num > max_part_num:
                        max_part_num = part_num
                        instance_id = instance_key
                gt_part_types = gt_function_masks[gt_instance_labels == instance_id]
                joint_type_counter = collections.Counter(gt_part_types)
                joint_type_keys = joint_type_counter.keys()
                max_part_num = 0
                gt_part_type = ignore_label
                for joint_type_key in joint_type_keys:
                    part_num = joint_type_counter[joint_type_key]
                    if part_num > max_part_num:
                        max_part_num = part_num
                        gt_part_type = joint_type_key
                if gt_part_type == ignore_label:
                    continue
                elif gt_part_type == function_mask_id:
                    joint_type_acc = 1
                else:
                    joint_type_acc = 0
                part_instance_mask[part_labels == part_id] = pre_instance_id
                gt_part_joint_direction = gt_joint_directions[gt_instance_labels == instance_id]
                gt_joint_axis = np.median(gt_part_joint_direction, axis=0)
                if np.linalg.norm(gt_joint_axis) == 0:
                    continue
                gt_joint_axis = gt_joint_axis / np.linalg.norm(gt_joint_axis)
                gt_part_joint_proj_vecs = gt_joint_proj_vecs[gt_instance_labels == instance_id]
                part_pcd = pcd[gt_instance_labels == instance_id]
                gt_joint_pts = gt_part_joint_proj_vecs + part_pcd
                gt_joint_pt = np.median(gt_joint_pts, axis=0)
                # gt
                joint_end = gt_joint_pt + gt_joint_axis
                gt_joint_pt = gt_joint_pt * scale + center
                joint_end = joint_end * scale + center
                gt_joint_axis = joint_end - gt_joint_pt
                gt_joint_axis = gt_joint_axis / np.linalg.norm(gt_joint_axis)
                # pre
                joint_end = pred_joint_pt + pre_joint_axis
                pred_joint_pt = pred_joint_pt * scale + center
                joint_end = joint_end * scale + center
                pre_joint_axis = joint_end - pred_joint_pt
                pre_joint_axis = pre_joint_axis / np.linalg.norm(pre_joint_axis)
                angle_error = self.online_direction_error(gt_joint_axis, pre_joint_axis)
                if angle_error > 90:
                    angle_error = 180 - angle_error
                trans_error = self.online_dist_between_3d_lines(pred_joint_pt, gt_joint_pt, pre_joint_axis, gt_joint_axis)
                instance_errors[instance_id] = {"pre_pcd_num": (part_labels == part_id).sum(),
                                                "joint_type_acc": joint_type_acc,
                                                "real_pcd_num": (gt_instance_labels == instance_id).sum(),
                                                "gt_instance": instance_id, "angle_error": angle_error,
                                                "trans_error": trans_error,
                                                "joint_type": function_mask_id}
                print("joint_type", function_mask_id, "angle_error", angle_error, "trans_error", trans_error)
            pre_instance_labels[part_indexs] = part_instance_mask
        # view_point_cloud_parts(coord_per_point.reshape(-1, 3), pre_instance_labels)
        mean_iou, part_ious, part_types, match_dict = self.calculate_point_cloud_part_iou(pre_instance_labels,
                                                                                          gt_instance_labels,
                                                                                          gt_function_masks)
        if len(part_ious) > 0:
            for part_iou, part_type in zip(part_ious, part_types):
                if part_type == ignore_label:
                    continue
                if part_type not in ious.keys():
                    ious[part_type] = []
                ious[part_type].append(part_iou)
        else:
            mean_iou = 0

        match_ids = match_dict.keys()
        for match_id in match_ids:
            if match_id in instance_errors.keys():
                instance_error = instance_errors[match_id]
                if instance_error["joint_type"] not in trans_errors.keys():
                    trans_errors[instance_error["joint_type"]] = []
                    angle_errors[instance_error["joint_type"]] = []
                    joint_type_accus[instance_error["joint_type"]] = []
                trans_errors[instance_error["joint_type"]].append(instance_error["trans_error"])
                angle_errors[instance_error["joint_type"]].append(instance_error["angle_error"])
                joint_type_accus[instance_error["joint_type"]].append(instance_error["joint_type_acc"])
        object_result = {"angle_errors": angle_errors, "trans_errors": trans_errors, "object_id": object_cat_id,
                         "sem_accu": sem_accu, "pixel_accu": pixel_accu,  "mean_iou": mean_iou, "ious": ious,
                         "joint_type_accu": joint_type_accus}
        print("angle_errors:{} trans_errors:{}".format(angle_errors, trans_errors))
        return object_result

    def gt_mask_evaluation(self, data_dict, ignore_label=0):
        coord = data_dict["coords"].to(self.device, dtype=torch.float)
        if coord.shape[1] > self.num_point:
            fps_pcs_idx = furthest_point_sample(coord, self.num_point).cpu().numpy()
            feat_per_point = index_points(data_dict["feats"], fps_pcs_idx).transpose(1, 2).to(self.device, dtype=torch.float)
            data_dict["sem_labels"] = index_masks(data_dict["sem_labels"], fps_pcs_idx)
            data_dict["instance_labels"] = index_masks(data_dict["instance_labels"], fps_pcs_idx)
            data_dict["point_center_offsets"] = index_points(data_dict["point_center_offsets"], fps_pcs_idx)
            data_dict["joint_directions"] = index_points(data_dict["joint_directions"], fps_pcs_idx)
            data_dict["joint_proj_vecs"] = index_points(data_dict["joint_proj_vecs"], fps_pcs_idx)
            data_dict["coords"] = index_points(data_dict["coords"], fps_pcs_idx)
        else:
            feat_per_point = data_dict["feats"].transpose(1, 2).to(self.device, dtype=torch.float)
        coord_per_point = data_dict["coords"]
        pred = self.forward(feat_per_point)
        valid_mask = data_dict["sem_labels"] != ignore_label
        valid_mask = valid_mask.reshape(-1)
        # finish sample
        object_cat_id = data_dict["file_id"][0].split("/")[-1].split(".")[0]
        scale = data_dict["scale"][0].cpu().numpy()
        center = data_dict["center"][0].cpu().numpy()
        sem_preds = torch.argmax(pred["sem_logits"].reshape(-1, self.num_classes).detach(), dim=-1)
        sem_preds = sem_preds.detach().cpu().numpy()
        pre_joint_proj_vecs = pred["joint_proj_vecs"].detach().cpu().numpy()[0]
        pre_joint_directions = pred["joint_directions"].detach().cpu().numpy()[0]

        gt_joint_proj_vecs = data_dict["joint_proj_vecs"].detach().cpu().numpy()[0]
        gt_joint_directions = data_dict["joint_directions"].detach().cpu().numpy()[0]
        gt_instance_labels = data_dict["instance_labels"].detach().cpu().numpy()[0]
        gt_function_masks = data_dict["sem_labels"].detach().cpu().numpy()[0]
        pcd = coord_per_point.detach().cpu().numpy()[0]
        # cal error
        sem_accu = (sem_preds == gt_function_masks).sum() / gt_function_masks.shape[0]
        pixel_accu = (sem_preds[valid_mask] == gt_function_masks[valid_mask]).sum() / gt_function_masks[valid_mask].shape[0]
        target_part_instances = gt_instance_labels[valid_mask]
        instance_ids = np.unique(target_part_instances)
        instance_errors = {}
        for instance_id in instance_ids:
            instance_mask = gt_instance_labels == instance_id
            pre_instance_mask = gt_function_masks[valid_mask]
            counts = np.bincount(pre_instance_mask)
            pre_part_type = np.argmax(counts)
            gt_part_type = gt_function_masks[instance_mask][0]
            instance_pcd = pcd[instance_mask]
            pre_instance_joint_proj_vecs = pre_joint_proj_vecs[instance_mask]
            pre_instance_joint_directions = pre_joint_directions[instance_mask]
            pre_joint_axis = np.median(pre_instance_joint_directions, axis=0)
            pre_joint_axis = pre_joint_axis / np.linalg.norm(pre_joint_axis)
            pred_joint_pts = pre_instance_joint_proj_vecs + instance_pcd
            pred_joint_pt = np.median(pred_joint_pts, axis=0)
            gt_instance_joint_proj_vecs = gt_joint_proj_vecs[instance_mask]
            gt_instance_joint_directions = gt_joint_directions[instance_mask]
            gt_joint_axis = np.median(gt_instance_joint_directions, axis=0)
            gt_joint_axis = gt_joint_axis / np.linalg.norm(gt_joint_axis)
            gt_joint_pts = gt_instance_joint_proj_vecs + instance_pcd
            gt_joint_pt = np.median(gt_joint_pts, axis=0)

            joint_end = gt_joint_pt + gt_joint_axis
            gt_joint_pt = gt_joint_pt * scale + center
            joint_end = joint_end * scale + center
            gt_joint_axis = joint_end - gt_joint_pt
            gt_joint_axis = gt_joint_axis / np.linalg.norm(gt_joint_axis)
            # pre
            joint_end = pred_joint_pt + pre_joint_axis
            pred_joint_pt = pred_joint_pt * scale + center
            joint_end = joint_end * scale + center
            pre_joint_axis = joint_end - pred_joint_pt
            pre_joint_axis = pre_joint_axis / np.linalg.norm(pre_joint_axis)
            angle_error = self.online_direction_error(gt_joint_axis, pre_joint_axis)
            if angle_error > 90:
                angle_error = 180 - angle_error
            trans_error = self.online_dist_between_3d_lines(pred_joint_pt, gt_joint_pt, pre_joint_axis, gt_joint_axis)
            instance_errors[instance_id] = {"joint_type_acc": pre_part_type == gt_part_type,
                                            "real_pcd_num": (gt_instance_labels == instance_id).sum(),
                                            "gt_instance": instance_id, "angle_error": angle_error,
                                            "trans_error": trans_error,
                                            "joint_type": pre_part_type}

    def online_dist_between_3d_lines(self, p1, p2, e1, e2):
        orth_vect = np.cross(e1, e2)
        p = p1 - p2
        if np.linalg.norm(orth_vect) == 0:
            dist = np.linalg.norm(np.cross(p, e1)) / np.linalg.norm(e1)
        else:
            dist = np.linalg.norm(np.dot(orth_vect, p)) / np.linalg.norm(orth_vect)
        return dist

    def online_direction_error(self, e1, e2):
        cos_theta = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_radian = np.arccos(cos_theta) * 180 / np.pi
        return angle_radian

    def open3d_part(self, point_cloud, eps=0.2, visua=False):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=30, print_progress=True))
        if visua:
            colors = plt.get_cmap("tab20")(labels / (labels.max() + 1))
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

            o3d.visualization.draw_geometries([pcd])
        return labels

    def sklearn_cluster(self, point_cloud, eps=0.1, min_samples=100, visua=False):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(point_cloud)
        return labels

    def calculate_point_cloud_part_iou(self, predicted_clusters, ground_truth_clusters, function_masks, ignore_mask=0):
        predict_instance_ids = np.unique(predicted_clusters)
        gt_instance_ids = np.unique(ground_truth_clusters[ground_truth_clusters != ignore_mask])
        iou_matrix = np.zeros((len(gt_instance_ids), len(predict_instance_ids)))
        part_type_matrix = np.zeros((len(gt_instance_ids), len(predict_instance_ids)))
        for i in range(len(gt_instance_ids)):
            for j in range(len(predict_instance_ids)):
                intersection = (predicted_clusters == predict_instance_ids[j]) & (ground_truth_clusters == gt_instance_ids[i])
                union = (predicted_clusters == predict_instance_ids[j]) | (ground_truth_clusters == gt_instance_ids[i])
                iou = intersection.sum() / union.sum()
                iou_matrix[i, j] = iou
                joint_type_counter = collections.Counter(function_masks[ground_truth_clusters == gt_instance_ids[i]])
                joint_type_keys = joint_type_counter.keys()
                max_part_num = 0
                part_type = 2
                for joint_type_key in joint_type_keys:
                    part_num = joint_type_counter[joint_type_key]
                    if part_num > max_part_num:
                        max_part_num = part_num
                        part_type = joint_type_key
                part_type_matrix[i, j] = part_type
        row_inds, col_inds = linear_sum_assignment(-iou_matrix)  # 最小化IoU
        match_dict = {}
        for row_ind, col_ind in zip(row_inds, col_inds):
            match_dict[predict_instance_ids[col_ind]] = gt_instance_ids[row_ind]
        mean_iou = iou_matrix[row_inds, col_inds].sum() / len(row_inds)
        part_iou = iou_matrix[row_inds, col_inds]
        part_type = part_type_matrix[row_inds, col_inds]
        return mean_iou, part_iou, part_type, match_dict



