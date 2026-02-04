import kagglehub
import open3d as o3d
import matplotlib.pyplot as plt
import os
import numpy as np
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['XDG_SESSION_TYPE'] = 'x11'


# 1. Descargar el dataset
path = kagglehub.dataset_download("priteshraj10/point-cloud-lidar-toronto-3d")

# 2. Localizar el primer archivo .ply disponible
files = [f for f in os.listdir(path) if f.endswith('.ply')]
if not files:
    # Si los archivos están en subcarpetas, buscamos una más profunda
    for root, dirs, filenames in os.walk(path):
        for f in filenames:
            if f.endswith('.ply'):
                sample_file = os.path.join(root, f)
                break
else:
    sample_file = os.path.join(path, files[0])

# 3. Cargar la nube de puntos
pcd = o3d.io.read_point_cloud(sample_file)
downpcd = pcd.voxel_down_sample(voxel_size=0.8)

distances=pcd.compute_nearest_neighbor_distance()
avg_dist=np.mean(distances)
radius=3*avg_dist

bpa_mesh=o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius,radius*2]))

o3d.visualization.draw_geometries([bpa_mesh],
                                  zoom=0.3812,)