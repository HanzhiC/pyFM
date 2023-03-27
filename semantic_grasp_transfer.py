import sys, os, time, math, pickle
import numpy as np
import torch
import open3d as o3d
from utils import *
import numpy as np
from tqdm import tqdm
import os, sys, yaml, configargparse, pickle
import torch
import trimesh
import h5py
from transformations import rotation_matrix

# Grasp file: Mug_62634df2ad8f19b87d1b7935311a2ed0_0.02328042176991366.h5
# This is the grasp file for real-world mug with scale of 0.0232.
category = 'mug'
for file in os.listdir('datasets/acronym/{0}'.format(category)):
    if file.endswith('h5'):
        grasp_file = file
        break

# see grasp pose ==> load results [N, 4, 4] correspond to grasps, [N, ] correspond to if there is a success
grasp_poses, successes = load_grasps('datasets/acronym/{0}/{1}'.format(category, grasp_file))

data = h5py.File('datasets/acronym/{0}/{1}'.format(category, grasp_file), "r")
print(np.array(data['object/scale']))

# sRT['RT'] @ t brings the gripper from its system to the desired grasping posotion
# scale is the corresponding scale to bring gripper to the canonical scale of the object
# So this scale might be not so similar to the object world scale
sRT = np.load('datasets/acronym/{0}/transform_matrix.npz'.format(category))

# Acquire the center points of the gripper
gripper_line_start = np.array([[-4.10000000e-02, -7.27595772e-12, 6.59999996e-02], [4.100000e-02, -7.27595772e-12, 6.59999996e-02]],dtype=np.float32)
gripper_line_end = np.array([[-4.10000000e-02, -7.27595772e-12, 1.12169998e-01], [4.100000e-02, -7.27595772e-12, 1.12169998e-01]],dtype=np.float32)
gripper_line_start_midpoint = gripper_line_end.mean(0) # [1, 3]
gripper_line_end_midpoint = gripper_line_end.mean(0) # [1, 3]
vis_rotation = rotation_matrix(np.pi/2, [0, 1, 0])

# visualize the object and the mesh
obj_anno = trimesh.load('datasets/obj/mug/train/62634df2ad8f19b87d1b7935311a2ed0/0.ply')
scene, contact_points, grasps = [], [], []
gripper = create_gripper_marker(color=[0, 255, 0])
for i, t in enumerate(grasp_poses[np.where(successes==1)[0]]):
     # In canonical space
    point_loc = gripper_line_end_midpoint @ (sRT['RT'] @ t)[:3, :3].T + (sRT['RT'] @ t)[:3, 3]
    point_loc *= sRT['scale']
    gripper_t = gripper.copy().apply_transform(sRT['RT'] @ t).apply_scale(sRT['scale'])
    gripper_t.apply_transform(vis_rotation)
    # points = trimesh.points.PointCloud(point_loc, colors=[255, 0, 0, 255])
    contact_points += [point_loc]
    grasps += [gripper_t]
    # if i > 5:
    #     break

prim_colors = random_colors(255)

# Load semantic primitives
prim_anno = np.load('to_use_grasp_transfer/template/primitive.npz')['primitives'] # [M, 4]
contact_points = np.stack(contact_points, axis=0) # [N, 3]
points_label = compute_state(prim_anno, contact_points) # [N, ]
prim_graspbooks = {k: [] for k in range(len(prim_anno))}
for p, k, g in zip(contact_points, points_label, grasps):
    prim_graspbooks[k].append(g)

for k, v in prim_graspbooks.items():
    if len(v) > 10:
        print(k)


# draw the primitive-corresponding grasping
obj_anno = trimesh_to_o3d(obj_anno.apply_transform(vis_rotation)).paint_uniform_color([0, 0.8, 0])
scene_o3d = [obj_anno]
query_prim_ids = [17, 136, 183, ]
for query_prim_id in query_prim_ids: 
    prim_loc = prim_anno[query_prim_id, 1:] @ rotation_matrix(np.pi/2, [0, 1, 0])[:3, :3].T
    prim_o3d = o3d.geometry.TriangleMesh.create_sphere()
    prim_o3d.compute_vertex_normals()
    prim_o3d.scale(0.03, [0,0,0])
    prim_o3d.translate(prim_loc)
    prim_o3d.paint_uniform_color(prim_colors[query_prim_id])
    scene_o3d.append(prim_o3d)
    for g in prim_graspbooks[query_prim_id]:
        scene_o3d.append(trimesh_to_o3d(g).paint_uniform_color(prim_colors[query_prim_id]))


# Transfer grasping w/ local translation
shape_ids = ['fad118b32085f3f2c2c72e575af174cd', '3c0467f96e26b8c6a93445a1757adf6', ]
            #  '8570d9a8d24cb0acbebd3c0c0c70fb03', 'b88bcf33f25c6cb15b4f129f868dedb']
for i, shape_id in enumerate(shape_ids):
    prim_trans =  np.load('to_use_grasp_transfer/trans/primitive_{}.npz'.format(shape_id))['primitives']
    obj_trans = trimesh_to_o3d(trimesh.load('to_use_grasp_transfer/trans/shape_{}.obj'.format(shape_id)).apply_transform(vis_rotation))
    vis_dist = (i+1)*2
    obj_trans.translate([0, 0, vis_dist])
    scene_o3d.append(obj_trans)
    for query_prim_id in query_prim_ids:
        prim_loc_trans = prim_trans[query_prim_id, 1:] @ rotation_matrix(np.pi/2, [0, 1, 0])[:3, :3].T
        prim_loc_anno = prim_anno[query_prim_id, 1:] @ rotation_matrix(np.pi/2, [0, 1, 0])[:3, :3].T
        prim_offset = prim_loc_trans - prim_loc_anno

        prim_o3d = o3d.geometry.TriangleMesh.create_sphere()
        prim_o3d.compute_vertex_normals()
        prim_o3d.scale(0.03, [0,0,0])
        prim_o3d.translate(prim_loc_trans)
        prim_o3d.translate([0, 0, vis_dist])
        prim_o3d.paint_uniform_color(prim_colors[query_prim_id])
        scene_o3d.append(prim_o3d)
        for g in prim_graspbooks[query_prim_id]:
            gripper_o3d = trimesh_to_o3d(g).paint_uniform_color(prim_colors[query_prim_id])
            gripper_o3d.translate(prim_offset)
            gripper_o3d.translate([0, 0, vis_dist])
            scene_o3d.append(gripper_o3d)


o3d.visualization.draw(scene_o3d)

