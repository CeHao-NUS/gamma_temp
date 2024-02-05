import glob
import os
import pdb

from visual_model.gamma_model import gamma_model_net
import argparse
import torch
import numpy as np

def main(args):
    model = gamma_model_net(pointnet_type=args.pointnet_type, num_point=int(args.num_points), device=args.device).to(args.device)
    assert os.path.exists(args.model_path)
    print("load model from path:", args.model_path)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    files = glob.glob("/data/physic_logic_data/multi_view_cabinet_rgb_test/*.npz")
    for path in files:
        file = np.load(path, allow_pickle=True)
        pcd_world = file["per_coord_world"]
        # extrinsic = file["extrinsic"]
        # pcd_camera = translate_pc_world_to_camera(pcd_world, extrinsic)
        # view_object_mask(pcd_camera)
        results, instance_labels, camera_pcd = model.online_inference(camera_pcd=pcd_world, view_res=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bbox object')
    parser.add_argument('--model_path', type=str, default="./train_process/gamma/2023-12-28-23_pointnet2_msg_16/best.pth")
    parser.add_argument('--pointnet_type', type=str, default="pointnet2_msg")
    parser.add_argument('--num_points', type=int, default=10000)
    parser.add_argument("--cluster", type=str, help="instructions file path", default="joint_and_center")
    parser.add_argument("--device", type=str, help="cuda or cpu", default="cuda")
    args = parser.parse_args()
    main(args)