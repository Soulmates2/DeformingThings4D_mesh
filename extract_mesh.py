import os
import sys
import argparse
import numpy as np
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))
ROOT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))
sys.path.append(ROOT_DIR)

# from file_io import *


parser = argparse.ArgumentParser()
parser.add_argument('--input_anime_dir', type=str, default='/home/kaist984/dataset/DeformingThings4D/DeformingThings4D/', help='dataset path')
parser.add_argument('--output_mesh_dir', type=str, default='/home/kaist984/dataset/DeformingThings4D/DeformingThings4D/', help='dataset path')

args = parser.parse_args()


def tri_mesh_to_obj(out_file_path, vertices, faces):
    mesh_f = open(out_file_path, 'w')
    mesh_f.write(f'# number of vertices: {vertices.shape[0]}\n')
    mesh_f.write(f'# number of vertices: {faces.shape[0]}\n')
    
    for i in range(vertices.shape[0]):
        mesh_f.write(f'v {vertices[i, 0]} {vertices[i, 1]} {vertices[i, 2]}\n')
    
    for i in range(faces.shape[0]):
        mesh_f.write(f'f {faces[i, 0]+1} {faces[i, 1]+1} {faces[i, 2]+1}\n')
    mesh_f.close()


def tri_mesh_to_ply(out_file_path, vertices, faces):
    mesh_f = open(out_file_path, 'w')
    mesh_f.write('ply\n')
    mesh_f.write('format ascii 1.0\n')
    mesh_f.write(f'element vertex {vertices.shape[0]}\n')
    mesh_f.write('property double x\n')
    mesh_f.write('property double y\n')
    mesh_f.write('property double z\n')
    mesh_f.write(f'element face {faces.shape[0]}\n')
    mesh_f.write('property list uchar int vertex_indices\n')
    mesh_f.write('end_header\n')
    
    for i in range(vertices.shape[0]):
        mesh_f.write(f'{vertices[i, 0]} {vertices[i, 1]} {vertices[i, 2]}\n')

    for i in range(faces.shape[0]):
        mesh_f.write(f'3 {faces[i, 0]} {faces[i, 1]} {faces[i, 2]}\n')
    mesh_f.close()


def anime_read(filename):
    """
    Code adapted from https://github.com/rabbityl/DeformingThings4D/blob/main/code/anime_renderer.py

    filename: path of .anime file
    return:
        nf: number of frames in the animation
        nv: number of vertices in the mesh (mesh topology fixed through frames)
        nt: number of triangle face in the mesh
        vert_data: vertice data of the 1st frame (3D positions in x-y-z-order)
        face_data: triangle face data of the 1st frame
        offset_data: 3D offset data from the 2nd to the last frame
    """
    f = open(filename, 'rb')
    nf = np.fromfile(f, dtype=np.int32, count=1)[0]
    nv = np.fromfile(f, dtype=np.int32, count=1)[0]
    nt = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    '''check data consistency'''
    if len(offset_data) != (nf - 1) * nv * 3:
        raise ("data inconsistent error!", filename)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))
    return nf, nv, nt, vert_data, face_data, offset_data


def extract_mesh(nf, vert_data, face_data, offset_data, output_dir):
    # Make sure that the number of provided offsets is equal to the animation length
    assert offset_data.shape[0] == nf - 1  # nf also contains the 1st frame
    
    # Create a Mesh file for the first frame
    first_frame_mesh_file_path = os.path.join(output_dir, "000000.obj")
    tri_mesh_to_obj(first_frame_mesh_file_path, vert_data, face_data)
    
    # mesh_initial = trimesh.Trimesh(vertices=vert_data, faces=face_data)
    # obj_out = trimesh.exchange.obj.export_obj(mesh_initial)
    # with open(os.path.join(path_to_meshes, "000000.obj"), "w") as f:
    #     f.write(obj_out)
    
    for f in range(nf-1):
        # Create and store one Mesh file for every frame
        vertices = vert_data + offset_data[f]
        frame_mesh_file_path = os.path.join(output_dir, f"{f:06}.obj")
        tri_mesh_to_obj(frame_mesh_file_path, vertices, face_data)
    print(f"Frames are saved at {output_dir}")


if __name__ == "__main__":
    path = args.input_anime_dir
    out_path = args.output_mesh_dir
    categories = ['animals', 'humanoids']
    for category in categories:
        characters = sorted(os.listdir(os.path.join(path, category)))
        for character in characters:
            input_anime_file_name = os.path.join(path, category, character, f'{character}.anime')
            assert(os.path.exists(input_anime_file_name))
            
            
            output_dir = os.path.join(out_path, category, character, "mesh_seq")
            if not os.path.exists(output_dir): os.makedirs(output_dir)

            nf, nv, nt, vert_data, face_data, offset_data = anime_read(input_anime_file_name)
            extract_mesh(nf, vert_data, face_data, offset_data, output_dir)
