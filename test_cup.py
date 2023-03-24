import numpy as np

from pyFM.mesh import TriMesh
from pyFM.functional import FunctionalMapping

import trimesh
import open3d as o3d

def get_submesh(mesh, correspondence, query=50):

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
            
    sub_mesh_prim = mesh_anno.submesh([valid_face_ids], append=True)
    return sub_mesh_prim

# anno
mesh_anno = trimesh.load('../gcasp_more/to_use_grasp_transfer/pred/mesh_0.obj')
vert2prim_anno = np.load('../gcasp_more/to_use_grasp_transfer/correspondence_0.npy')
sub_mesh_prim_anno = get_submesh(mesh_anno, vert2prim_anno)

# trans
mesh_trans = trimesh.load('../gcasp_more/to_use_grasp_transfer/pred/mesh_2.obj')
vert2prim_trans = np.load('../gcasp_more/to_use_grasp_transfer/correspondence_2.npy')
sub_mesh_prim_trans = get_submesh(mesh_trans, vert2prim_trans)


# Visualization
mesh_o3d_prim = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(sub_mesh_prim_anno.vertices*1.1), 
                                     o3d.utility.Vector3iVector(sub_mesh_prim_anno.faces))
mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh_anno.vertices), 
                                     o3d.utility.Vector3iVector(mesh_anno.faces))

mesh_o3d.paint_uniform_color([0,1,0])
mesh_o3d_prim.paint_uniform_color([1,0,0])
mesh_o3d_prim.compute_vertex_normals()
mesh_o3d.compute_vertex_normals()

o3d.visualization.draw([mesh_o3d_prim, mesh_o3d])