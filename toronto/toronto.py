import kagglehub
import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configuration for Linux (prevents window crashes in Wayland/X11)
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['XDG_SESSION_TYPE'] = 'x11'

# Create output folder if it doesn't exist
os.makedirs("toronto/output", exist_ok=True)

# DATA DOWNLOAD AND LOADING
print("\n--- STARTING ADAS PIPELINE (TORONTO 3D) ---")
path = kagglehub.dataset_download("priteshraj10/point-cloud-lidar-toronto-3d")

# Locate the first .ply file
sample_file = None
for root, dirs, files in os.walk(path):
    for f in files:
        if f.endswith('.ply'):
            sample_file = os.path.join(root, f)
            break
    if sample_file: break

if not sample_file:
    print("ERROR: .ply file not found")
    exit()

pcd = o3d.io.read_point_cloud(sample_file)

# Center the cloud at (0,0,0)
pcd.translate(-pcd.get_center())
print(f"Original cloud loaded: {len(pcd.points)} points.")


# VOXEL GRID
pcd_reduced = pcd.voxel_down_sample(voxel_size=0.3) 

print(f"Points after Voxel Grid: {len(pcd_reduced.points)}")
reduction = 100 - (len(pcd_reduced.points) / len(pcd.points) * 100)
print(f"   -> We have reduced the information by {reduction:.1f}%")


# RANSAC
plane_model, inliers = pcd_reduced.segment_plane(
    distance_threshold=0.3,
    ransac_n=3,              
    num_iterations=1000
)

cloud_ground = pcd_reduced.select_by_index(inliers)
cloud_obstacles = pcd_reduced.select_by_index(inliers, invert=True)

print(f"RANSAC applied. Ground points removed: {len(cloud_ground.points)}")
print(f"   -> Remaining floating points (obstacles): {len(cloud_obstacles.points)}")


# DBSCAN 
labels = np.array(cloud_obstacles.cluster_dbscan(eps=0.5, min_points=15))
max_label = labels.max()
print(f"DBSCAN applied. {max_label + 1} objects/clusters have been detected.")


# AXIS-ALIGNED BOUNDING BOXES (AABB)

print("\nGenerating image Version 1 (AABB - Straight boxes)...")
points_obs = np.asarray(cloud_obstacles.points)

plt.figure(figsize=(14, 10))
mask = (points_obs[:, 0] > -50) & (points_obs[:, 0] < 50) & (points_obs[:, 1] > -50) & (points_obs[:, 1] < 50)

plt.scatter(points_obs[mask, 0], points_obs[mask, 1], c='lightgray', s=1) 
ax = plt.gca()

counts = {"Cars": 0, "Poles/Pedestrians": 0, "Facades/Walls": 0, "Others": 0}

for i in range(max_label + 1):
    indices = np.where(labels[mask] == i)[0]
    if len(indices) == 0: continue

    points_object = points_obs[mask][indices]

    min_x, max_x = np.min(points_object[:, 0]), np.max(points_object[:, 0])
    min_y, max_y = np.min(points_object[:, 1]), np.max(points_object[:, 1])
    min_z, max_z = np.min(points_object[:, 2]), np.max(points_object[:, 2])

    dim_x = max_x - min_x
    dim_y = max_y - min_y
    length = max(dim_x, dim_y)
    width = min(dim_x, dim_y)
    height = max_z - min_z
    
    if length < 0.5 or width < 0.5 or height < 0.5: continue

    box_color = 'orange' 
    if (2.0 < length < 6.5) and (1.5 < width < 3.0) and (1.0 < height < 3.0):
        box_color = 'red'
        counts["Cars"] += 1
    elif (length < 1.5) and (width < 1.5) and (height > 1.5):
        box_color = 'blue'
        counts["Poles/Pedestrians"] += 1
    elif (length > 6.5):
        box_color = 'gray'
        counts["Facades/Walls"] += 1
    else:
        counts["Others"] += 1

    rect = patches.Rectangle((min_x, min_y), dim_x, dim_y,
                             linewidth=1.5, edgecolor=box_color, facecolor='none')
    ax.add_patch(rect)

legend = [
    patches.Patch(color='red', fill=False, linewidth=2, label=f'Cars ({counts["Cars"]})'),
    patches.Patch(color='blue', fill=False, linewidth=2, label=f'Poles/Pedestrians ({counts["Poles/Pedestrians"]})'),
    patches.Patch(color='gray', fill=False, linewidth=2, label=f'Facades/Walls ({counts["Facades/Walls"]})'),
    patches.Patch(color='orange', fill=False, linewidth=2, label=f'Others ({counts["Others"]})')
]
plt.legend(handles=legend, loc='upper right', fontsize=10)
plt.title("Version 1: AABB (Straight Boxes)")
plt.xlabel("X Distance (meters)")
plt.ylabel("Y Distance (meters)")
plt.axis('equal') 
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("toronto/output/01_urban_classification_aabb.png", dpi=200, bbox_inches='tight')
plt.close()


# ORIENTED BOUNDING BOXES (OBB)

print("Generating image Version 2 (OBB - Oriented Boxes using PCA)...")

plt.figure(figsize=(14, 10))
plt.scatter(points_obs[mask, 0], points_obs[mask, 1], c='lightgray', s=1) 
ax_obb = plt.gca()

counts_obb = {"Cars": 0, "Poles/Pedestrians": 0, "Facades/Walls": 0, "Others": 0}

for i in range(max_label + 1):
    indices = np.where(labels[mask] == i)[0]
    if len(indices) == 0: continue

    points_object = points_obs[mask][indices]
    if len(points_object) < 3: 
        continue

    min_z, max_z = np.min(points_object[:, 2]), np.max(points_object[:, 2])
    height_3d = max_z - min_z

    # PCA for orientation
    pts_2d = points_object[:, :2]
    center_2d = np.mean(pts_2d, axis=0)
    
    # Covariance matrix and eigenvectors
    cov = np.cov(pts_2d, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    
    # The eigenvector with the highest value indicates the longest direction of the object
    v1 = evecs[:, 1] 
    angle_rad = np.arctan2(v1[1], v1[0])
    angle_deg = np.degrees(angle_rad)
    
    # Inverse rotation matrix (to temporarily straighten the points)
    c, s = np.cos(-angle_rad), np.sin(-angle_rad)
    R = np.array([[c, -s], [s, c]])
    
    # We rotate the points around the center to make them straight
    pts_rot = np.dot(pts_2d - center_2d, R.T)
    
    # Now we can calculate the actual minimum width and length
    min_rot = np.min(pts_rot, axis=0)
    max_rot = np.max(pts_rot, axis=0)
    
    length_obb = max_rot[0] - min_rot[0]
    width_obb = max_rot[1] - min_rot[1]
    
    # Homogenize what is length and what is width for the classifier
    actual_length = max(length_obb, width_obb)
    actual_width = min(length_obb, width_obb)

   
    # Noise
    if actual_length < 0.5 or actual_width < 0.5 or height_3d < 0.5: continue

    # CLASSIFIER 
    box_color = 'orange' 
    if (2.0 < actual_length < 6.5) and (1.5 < actual_width < 3.0) and (1.0 < height_3d < 3.0):
        box_color = 'red'
        counts_obb["Cars"] += 1
    elif (actual_length < 1.5) and (actual_width < 1.5) and (height_3d > 1.5):
        box_color = 'blue'
        counts_obb["Poles/Pedestrians"] += 1
    elif (actual_length > 6.5):
        box_color = 'gray'
        counts_obb["Facades/Walls"] += 1
    else:
        counts_obb["Others"] += 1

    # DRAW THE ROTATED BOX
    # The starting corner must return to the original rotated space
    R_inv = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], 
                      [np.sin(angle_rad), np.cos(angle_rad)]])
    orig_corner = np.dot(min_rot, R_inv.T) + center_2d

    # Matplotlib will draw the box from that corner and rotate it with "angle"
    rect = patches.Rectangle(orig_corner, length_obb, width_obb, angle=angle_deg,
                             linewidth=1.5, edgecolor=box_color, facecolor='none')
    ax_obb.add_patch(rect)


legend_obb = [
    patches.Patch(color='red', fill=False, linewidth=2, label=f'Cars ({counts_obb["Cars"]})'),
    patches.Patch(color='blue', fill=False, linewidth=2, label=f'Poles/Pedestrians ({counts_obb["Poles/Pedestrians"]})'),
    patches.Patch(color='gray', fill=False, linewidth=2, label=f'Facades/Walls ({counts_obb["Facades/Walls"]})'),
    patches.Patch(color='orange', fill=False, linewidth=2, label=f'Others ({counts_obb["Others"]})')
]
plt.legend(handles=legend_obb, loc='upper right', fontsize=10)
plt.title("Version 2: OBB (Oriented Boxes with PCA)")
plt.xlabel("X Distance (meters)")
plt.ylabel("Y Distance (meters)")
plt.axis('equal') 
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig("toronto/output/02_urban_classification_obb.png", dpi=200, bbox_inches='tight')
print("Images successfully generated in the 'toronto/output' folder!")