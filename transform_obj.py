import argparse
import glob
import os

from autolab_core import RigidTransform
import numpy as np
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=str, help='foo help')
args = parser.parse_args()

files = glob.glob(os.path.join(args.directory, '*.obj'))
filename = files[0]


def apply_transform(points, transform):
    print(points.shape)
    x_homog = np.r_[points.transpose(), np.ones([1, points.shape[0]])]
    print(x_homog.shape)
    x_homog_tf = transform.matrix.dot(x_homog)
    x_tf = x_homog_tf[0:3, :]
    print(transform.matrix)
    return x_tf.transpose()

original_mesh = trimesh.load(filename, process=False)
temp_mesh = original_mesh.copy()

transform_matrix = np.loadtxt(os.path.join(args.directory, 'preprocessed', 'transform.txt'), delimiter=' ')
rotation, translation = RigidTransform.rotation_and_translation_from_matrix(transform_matrix)
transform = RigidTransform(rotation=rotation, translation=translation)
temp_mesh.vertices = apply_transform(original_mesh.vertices, transform)

temp_mesh.export(os.path.join(args.directory, 'preprocessed', 'transformed.obj'))
np.save(os.path.join(args.directory, 'preprocessed', 'transform.npy'), transform_matrix)