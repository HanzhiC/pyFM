import numpy as np

from pyFM.mesh import TriMesh
from pyFM.functional import FunctionalMapping

import trimesh
import open3d as o3d

def trimesh_to_o3d(mesh, color):
    mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), 
                                    o3d.utility.Vector3iVector(mesh.faces))
    mesh_o3d.paint_uniform_color(color)
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d

def visu(vertices):
    min_coord,max_coord = np.min(vertices,axis=0,keepdims=True),np.max(vertices,axis=0,keepdims=True)
    cmap = (vertices-min_coord)/(max_coord-min_coord)
    return (cmap * 255).astype(np.uint8)

def get_submesh(mesh, correspondence, query=100):

    # Acquiring submesh 0
    triangle_ids = np.arange(len(mesh.faces))
    valid_face_ids = []
    for i_vert, vert in enumerate(np.array(mesh.vertices)):
        i_prim = correspondence[i_vert]
        if i_prim == query:
            connected_triangles = np.logical_or(np.logical_or(
                                                mesh.faces[:, 0] == i_vert, 
                                                mesh.faces[:, 1] == i_vert),
                                                mesh.faces[:, 2] == i_vert)
            valid_face_id = triangle_ids[connected_triangles]
            valid_face_ids += valid_face_id.tolist()
            
    sub_mesh_prim = mesh.submesh([valid_face_ids], append=True)
    return sub_mesh_prim

def normal_to_rot(normal):
    nx, ny, nz = normal[0], normal[1], normal[2]
    dist = (nx**2+ny**2)**0.5
    rot = np.array([[ny/dist, -nx/dist, 0], [nx*nz/dist, ny*nz/dist, -dist], [nx, ny, nz]])
    return rot

def vis_plane_o3d(plane_points, plane_normals ):
    size = 0.7
    plane_o3d = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=0.001)
    rotation = normal_to_rot(plane_normals)
    plane_o3d.rotate(rotation.T)
    plane_o3d.translate([-size/2, -size/2, 0]).translate(plane_points)
    return plane_o3d

mesh_anno = trimesh.load('to_use_grasp_transfer/pred/mesh_0.obj')
vert2prim_anno = np.load('to_use_grasp_transfer/correspondence_0.npy')
mesh_trans = trimesh.load('to_use_grasp_transfer/pred/mesh_2.obj')
vert2prim_trans = np.load('to_use_grasp_transfer/correspondence_2.npy')

for q_id in range(250):
    # anno
    sub_mesh_prim_anno = get_submesh(mesh_anno, vert2prim_anno, q_id)
    sub_mesh_prim_anno.visual.vertex_colors = np.array([255, 0, 0])[None].repeat(len(sub_mesh_prim_anno.vertices), axis=0)
    plane_points_anno, plane_normals_anno = trimesh.points.plane_fit(trimesh.sample.sample_surface(sub_mesh_prim_anno, 100)[0])

    # trans
    sub_mesh_prim_trans = get_submesh(mesh_trans, vert2prim_trans, q_id)
    sub_mesh_prim_trans.visual.vertex_colors = np.array([0, 255, 0])[None].repeat(len(sub_mesh_prim_trans.vertices), axis=0)
    plane_points_trans, plane_normals_trans = trimesh.points.plane_fit(trimesh.sample.sample_surface(sub_mesh_prim_trans, 100)[0])

    # Acquire relative transformation, move anno to trans
    T_anno_trans = np.eye(4)
    t_anno_trans = plane_points_trans - plane_points_anno
    R_anno_trans = normal_to_rot(plane_normals_trans).T @ normal_to_rot(plane_normals_anno)
    T_anno_trans[:3, :3] = R_anno_trans 
    T_anno_trans[:3,  3] = R_anno_trans @ t_anno_trans

    # Visualization
    # mesh_o3d_prim = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(sub_mesh_prim_anno.vertices*1.02),  o3d.utility.Vector3iVector(sub_mesh_prim_anno.faces))
    mesh_anno_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh_anno.vertices),  o3d.utility.Vector3iVector(mesh_anno.faces))
    mesh_trans_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh_trans.vertices),  o3d.utility.Vector3iVector(mesh_trans.faces))

    coord = mesh_anno_o3d.create_coordinate_frame()
    # mesh_anno_o3d.transform(T_anno_trans)
    # plane_anno_o3d = vis_plane_o3d(plane_points_anno, plane_normals_anno).transform(T_anno_trans)
    plane_anno_o3d = vis_plane_o3d(plane_points_anno, plane_normals_anno).rotate(R_anno_trans).translate(t_anno_trans)
    mesh_anno_o3d = mesh_anno_o3d.rotate(R_anno_trans).translate(t_anno_trans)
    plane_trans_o3d = vis_plane_o3d(plane_points_trans, plane_normals_trans )

    mesh_anno_o3d.paint_uniform_color([0,1,0])
    mesh_trans_o3d.paint_uniform_color([1,0,0])
    plane_anno_o3d.paint_uniform_color([0,0.6,0.5])
    plane_trans_o3d.paint_uniform_color([0.6,0,0.2])

    mesh_trans_o3d.compute_vertex_normals()
    mesh_anno_o3d.compute_vertex_normals()
    plane_anno_o3d.compute_vertex_normals()
    plane_trans_o3d.compute_vertex_normals()
    o3d.visualization.draw([coord, mesh_anno_o3d, mesh_trans_o3d,
                            plane_trans_o3d, plane_anno_o3d, ])

# o3d.visualization.draw_geometries([plane_o3d])
# for q_id in range(255):

#     # trans
#     sub_mesh_prim_trans = get_submesh(mesh_trans, vert2prim_trans, q_id)
#     sub_mesh_prim_trans.visual.vertex_colors = np.array([0, 255, 0])[None].repeat(len(sub_mesh_prim_trans.vertices), axis=0)
#     sub_mesh_prim_anno_o3d = trimesh_to_o3d(sub_mesh_prim_anno, [1, 0, 0])
#     sub_mesh_prim_trans_o3d = trimesh_to_o3d(sub_mesh_prim_trans, [0, 1, 0])
#     o3d.visualization.draw([sub_mesh_prim_anno_o3d, sub_mesh_prim_trans_o3d])

    # Mesh registration --> ICP-based
    # mesh_to_other, _ = trimesh.registration.mesh_other(sub_mesh_prim_anno, sub_mesh_prim_trans, 
    #                                                 samples=100, icp_first=1, reflection=False, scale=False)
    # sub_mesh_prim_anno.apply_transform(mesh_to_other)
    # sub_mesh_prim_anno_o3d = trimesh_to_o3d(sub_mesh_prim_anno, [1, 0, 0])
    # sub_mesh_prim_trans_o3d = trimesh_to_o3d(sub_mesh_prim_trans, [0, 1, 0])
    
    # # Show mesh alignment
    # mesh_anno_o3d = trimesh_to_o3d(mesh_anno.copy().apply_transform(mesh_to_other), [1, 0, 1])
    # mesh_trans_o3d = trimesh_to_o3d(mesh_trans.copy(), [0, 1, 1])
    # o3d.visualization.draw([mesh_anno_o3d, mesh_trans_o3d, 
    #                         sub_mesh_prim_trans_o3d.scale(1.05, sub_mesh_prim_trans_o3d.get_center()), 
    #                         sub_mesh_prim_anno_o3d.scale(1.05, sub_mesh_prim_anno_o3d.get_center())])

    # Mesh registration --> procrustes-based
    # mesh_to_other, _, _ = trimesh.registration.procrustes(trimesh.sample.sample_surface(sub_mesh_prim_anno, 100)[0], 
    #                                                     trimesh.sample.sample_surface(sub_mesh_prim_trans, 100)[0],
    #                                                     reflection=False, 
    #                                                     scale=False)
    # sub_mesh_prim_anno.apply_transform(mesh_to_other)
    # sub_mesh_prim_anno_o3d = trimesh_to_o3d(sub_mesh_prim_anno, [1, 0, 0])
    # sub_mesh_prim_trans_o3d = trimesh_to_o3d(sub_mesh_prim_trans, [0, 1, 0])
    # o3d.visualization.draw([sub_mesh_prim_trans_o3d, sub_mesh_prim_anno_o3d])


    # sub_mesh_prim_anno.apply_transform(mesh_to_other)
    # sub_mesh_prim_anno_o3d = trimesh_to_o3d(mesh_anno, [1, 0, 0])
    # sub_mesh_prim_trans_o3d = trimesh_to_o3d(mesh_trans, [0, 1, 0])
    # o3d.visualization.draw([sub_mesh_prim_trans_o3d, sub_mesh_prim_anno_o3d])



