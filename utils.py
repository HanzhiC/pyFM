import os, sys, h5py
import torch
import trimesh
import plyfile
import numpy as np
import math
from math import cos, sin
import cv2
import numpy as np
import torch
import open3d as o3d
from pytorch3d.structures import Meshes
import trimesh
import json
from glob import glob
import mcubes
import colorsys
import random
def rot_X(theta):
    return np.array([[1,0,0],[0,cos(theta),-sin(theta)],[0,sin(theta),cos(theta)]],dtype=np.float32)
def rot_Y(theta):
    return np.array([[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]],dtype=np.float32)
def rot_Z(theta):
    return np.array([[cos(theta),-sin(theta),0],[sin(theta),cos(theta),0],[0,0,1]],dtype=np.float32)

# from https://github.com/NVlabs/acronym
def load_grasps(filename):
    """Load transformations and qualities of grasps from a JSON file from the dataset.
    Args:
        filename (str): HDF5 file name.

    Returns:
        np.ndarray: Homogenous matrices describing the grasp poses. 2000 x 4 x 4.
        np.ndarray: List of binary values indicating grasp success in simulation.
    """
    assert filename.endswith(".h5"), "Unknown file ending: {}".format(filename)
    data = h5py.File(filename, "r")
    T = np.array(data["grasps/transforms"])
    success = np.array(data["grasps/qualities/flex/object_in_gripper"])
    return T, success

def create_gripper_marker(color=[0, 0, 255], tube_radius=0.001, sections=6):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [4.10000000e-02, -7.27595772e-12, 6.59999996e-02],
            [4.10000000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [-4.100000e-02, -7.27595772e-12, 6.59999996e-02],
            [-4.100000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=0.002, sections=sections, segment=[[0, 0, 0], [0, 0, 6.59999996e-02]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[[-4.100000e-02, 0, 6.59999996e-02], [4.100000e-02, 0, 6.59999996e-02]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    return tmp

def save_ply(verts, faces, path_ply):
    '''
    Write a ply file to specified path.
    Args:
        verts: [V,3]
        faces: [F,3]
        path_ply: path where ply file be writed.
    '''
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]
    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    for i in range(0, num_verts):
        verts_tuple[i] = tuple(verts[i, :])
    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])
    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces],text=True)
    ply_data.write(path_ply)

def trimesh_scene_to_mesh(obj_mesh):
    if isinstance(obj_mesh, trimesh.Scene):
        geo_list = []
        for k, v in obj_mesh.geometry.items():
            geo_list.append(v)
        obj_mesh = trimesh.util.concatenate(geo_list)
    return obj_mesh

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_code(model, meta, file_params):
    if file_params['mode']=='train':
        '''get train code'''
        subject_idx_cuda = torch.Tensor([file_params['subject_idx']]).squeeze().long().cuda()[None,...]
        shape_code = model.get_latent_code(subject_idx_cuda)
    elif file_params['mode']=='eval':
        '''get eval code'''
        shape_code = np.loadtxt(os.path.join('DIF_decoder/eval/{0}'.format(meta['experiment_name']), file_params['filename'], 'checkpoints/embedding_epoch_0049.txt'), dtype=np.float32)
        shape_code = torch.from_numpy(shape_code).cuda()        
    # print("latent code shape: ", shape_code.shape)
    return shape_code

def load_recon_ply(meta, file_params):
    if file_params['mode']=='train':
        if file_params['filename'] == 'template':
            obj_recon = trimesh.load('DIF_decoder/recon/{0}/train/template.ply'.format(meta['experiment_name']))
        else:
            obj_recon = trimesh.load('DIF_decoder/recon/{0}/train/test{1}.ply'.format(meta['experiment_name'], str(file_params['subject_idx']).zfill(4)))
    elif file_params['mode']=='eval':
        obj_recon = trimesh.load('DIF_decoder/eval/{0}/{1}/checkpoints/test.ply'.format(meta['experiment_name'], file_params['filename']))
    return obj_recon

def get_model(meta):
    # define DIF-Net
    model = DeformedImplicitField(**meta)
    model.load_state_dict(torch.load(meta['checkpoint_path']))
    return model.cuda()

def R2q(R:np.array):
    
    w = R[0,0]+R[1,1]+R[2,2]+1
    x = R[0,0]-R[1,1]-R[2,2]+1
    y = -R[0,0]+R[1,1]-R[2,2]+1
    z = -R[0,0]-R[1,1]+R[2,2]+1
    
    q = np.array([w,x,y,z])

    index = np.argmax(q)

    # assert q[index]>0, "max(q) > 0"
    if q[index]<=0:
        return np.array([0,0,0,1],dtype=np.float32)

    q[index] = math.sqrt(q[index]) / 2

    if index==0:
        q0 = q[index]
        q1 = (R[2,1]-R[1,2]) / (4*q0)
        q2 = (R[0,2]-R[2,0]) / (4*q0)
        q3 = (R[1,0]-R[0,1]) / (4*q0)
    elif index==1:
        q1 = q[index]
        q0 = (R[2,1]-R[1,2]) / (4*q1)
        q2 = (R[0,1]+R[1,0]) / (4*q1)
        q3 = (R[2,0]+R[0,2]) / (4*q1)
    elif index==2:
        q2 = q[index]
        q0 = (R[0,2]-R[2,0]) / (4*q2)
        q1 = (R[0,1]+R[1,0]) / (4*q2)
        q3 = (R[1,2]+R[2,1]) / (4*q2)
    elif index==3:
        q3 = q[index]
        q0 = (R[1,0]-R[0,1]) / (4*q3)
        q1 = (R[2,0]+R[0,2]) / (4*q3)
        q2 = (R[1,2]+R[2,1]) / (4*q3)
    else:
        raise ValueError('index error:' +  str(index))
    # print(np.linalg.norm(np.array([q1,q2,q3,q0],dtype=np.float32)))
    return np.array([q1,q2,q3,q0],dtype=np.float32)

# Most important thing to remember: grasping pose is always discussed w.r.t real-world scale
# So always to first bring object system to the real-world one.
def get_RT_from_grasp_params(panda_grasp_info, obj_scale):
    left_points = panda_grasp_info['left_points']
    right_points = panda_grasp_info['right_points']
    appro = panda_grasp_info['approach_vector']
    depth = panda_grasp_info['depth']

    hori = right_points - left_points
    hori = hori / np.linalg.norm(hori)
    normal = np.cross(hori, appro)
    normal = normal / np.linalg.norm(normal)

    appro_correct = np.cross(normal, hori)

    RT = np.eye(4, dtype=np.float32)
    RT[:3,:3] = np.array([normal, hori, appro_correct], dtype=np.float32).T
    # as stored left_points and right_points live in the[-1, 1]^3 (SDF space)
    # After acquiring the center_points of (left_points + right_points)/ 2
    # We need to transform them into [-0.5, 0.5]^3 by multiplying 1/2
    # Finally, transform the center_points to real-world scale
    RT[:3,3] = ((left_points + right_points)/ 2)  * (1/2) * obj_scale
    # This can be understood easily with some drawings
    # Note that as depth is already in real-world scale (acquired from the gripper)
    # So there is no need to do scaling for vector of [-1 * appro_correct * depth]
    RT[:3,3] = (RT[:3,3]) + (-1 * appro_correct * depth)

    return RT

def get_width_from_grasp_params(panda_grasp_info, obj_scale):
    left_points = panda_grasp_info['left_points']
    right_points = panda_grasp_info['right_points']

    left_points = left_points * (obj_scale / 2)
    right_points = right_points * (obj_scale / 2)
    hori = right_points - left_points
    width = np.linalg.norm(hori)

    return width
    
############# For refine ##############
def get_points_sdf_normal(model, shape_code, points):
    sdf = model.inference_with_grad(points, shape_code)
    grad_sdf = torch.autograd.grad(sdf, points, grad_outputs=torch.ones_like(sdf), create_graph=True)[0]
    return sdf.squeeze(0), grad_sdf

def get_template_points_sdf_normal(model, points):
    sdf = model.get_template_field_with_grad(points)
    grad_sdf = torch.autograd.grad(sdf, points, grad_outputs=torch.ones_like(sdf), create_graph=True)[0]
    return sdf.squeeze(0), grad_sdf
########################################    

def o3d_to_pytorch(mesh_o3d, scale=1.):
    mesh_torch = Meshes(verts=torch.from_numpy(np.asarray(mesh_o3d.vertices)[None] * scale).float(), 
                        faces=torch.from_numpy(np.asarray(mesh_o3d.triangles)[None]).float())
    return mesh_torch

def trimesh_to_o3d(mesh, scale=1.):
    mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices * scale), 
                                     o3d.utility.Vector3iVector(mesh.faces))
    # mesh_o3d.paint_uniform_color([255 / 255, 229 / 255, 180 / 255])
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector( mesh.visual.vertex_colors[:, :3]/255)
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d

def o3d_to_trimesh(mesh_o3d, scale=1.):
    verts = np.asarray(mesh_o3d.vertices) * scale
    triangles = np.asarray(mesh_o3d.triangles)
    mesh = trimesh.Trimesh(verts, triangles)
    if mesh_o3d.has_vertex_colors():
        textures = np.array(mesh_o3d.vertex_colors)
        mesh.visual.vertex_colors = textures
    return mesh

def export_tsdf_mesh(tsdf, res, origin, out_file=None):
    _vertices, triangles = mcubes.marching_cubes(tsdf, 0.0)
    vertices = _vertices / res + origin
    mesh =  trimesh.Trimesh(vertices, triangles)
    colors = np.array([[255, 229, 180]]).repeat(len(vertices), 0) / 255
    mesh.visual.vertex_colors = colors
    if out_file is not None:
        mesh.export(out_file)
    return mesh

def compute_state(attrs, points):
    attrs = torch.Tensor(attrs)
    points = torch.Tensor(points)
    attrs = attrs.unsqueeze(1)
    points = points.unsqueeze(0)
    dists_all = torch.sum((points - attrs)**2, dim=-1)
    dist_min, points_label = torch.min(dists_all, dim=-2)
    return points_label.cpu().numpy().astype(np.int32)

def random_colors(N, bright=True, shuffle=False):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    if shuffle:
        random.shuffle(colors)
    return colors

def pc_normalize(pc):
    centroid = (np.amax(pc, axis=-2, keepdims=True) + np.amin(pc, axis=-2, keepdims=True))/2
    pc = pc - centroid
    l = np.sqrt(np.sum(pc**2, axis=-1, keepdims=True))
    m = np.amax(l, axis=-2,keepdims=True)
    pc = pc / m
    return pc, m, centroid