import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from load_data import Data
import os
import random
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pickle
from models import Analytics

def bootstrap(iterations, pt_data):
    """Optimized bootstrap without expensive concatenations."""
    keys = list(pt_data.keys())
    n_keys = len(keys)
    
    # Pre-compute block statistics
    block_sums = np.array([np.sum(pt_data[k]) for k in keys])
    block_sizes = np.array([len(pt_data[k]) for k in keys])
    total_size = np.sum(block_sizes)
    
    # Vectorized sampling
    sample_indices = np.random.randint(0, n_keys, size=(iterations, n_keys))
    sampled_sums = np.sum(block_sums[sample_indices], axis=1)
    
    return (sampled_sums / total_size).tolist()

def groups_to_dict(grouped_data):
    """
    Convert grouped data to dictionary format.
    Input: List of groups, where each group is a list of (coord, value) pairs
    Output: Dictionary {group_num: [list of delta values]}
    """
    result_dict = {}
    
    for group_num, group in enumerate(grouped_data):
        # Extract just the delta values (second element of each tuple)
        delta_values = [val for coord, val in group]
        result_dict[group_num] = delta_values
    
    return result_dict


def extract_surface(mask):
    """Extract surface voxels from a binary mask using morphological operations."""
    return mask ^ binary_erosion(mask)


def voxel_coords(binary_mask, spacing=(1.0, 1.0, 1.0)):
    """Convert binary mask indices to physical coordinates using spacing information."""
    indices = np.argwhere(binary_mask)
    return indices * np.array(spacing), indices


def separation_distances(gt, pred):
    """Calculate distances from ground truth surface points to predicted surface."""
    pred_surface = extract_surface(pred > 0)
    gt_surface = extract_surface(gt > 0)

    pred_pts_phys, _ = voxel_coords(pred_surface)
    gt_pts_phys, _ = voxel_coords(gt_surface)

    pred_tree = cKDTree(pred_pts_phys)
    d_gt_to_pred, _ = pred_tree.query(gt_pts_phys)

    return list(zip(gt_pts_phys, d_gt_to_pred))


def separation_deltas(gt, seg1, seg2):
    """Calculate separation distance differences between two segmentations relative to ground truth."""
    sep1 = separation_distances(gt, seg1)
    sep2 = separation_distances(gt, seg2)
    
    # Extract just the distances (second element of each tuple)
    coords1 = np.array([coord for coord, _ in sep1])
    distances1 = np.array([dist for _, dist in sep1])
    
    coords2 = np.array([coord for coord, _ in sep2])
    distances2 = np.array([dist for _, dist in sep2])
    
    # Calculate deltas
    deltas = distances1 - distances2
    
    # Return coordinates with their delta values
    return list(zip(coords1, deltas))


def greedy_iteration(data: List[Tuple[Tuple[float, float, float], float]], k: int) -> List[List[Tuple[Tuple[float, float, float], float]]]:
    """
    Groups using a KMeans-seeded Greedy algorithm.
    Input is a list of tuples (coord, val) where coord is (x, y, z) and val is the relevant entry in D_delta.
    Returns a list of groups, where each group is a list of (coord, value) pairs.
    """
    coords = np.array([pt for pt, val in data])
    n = len(coords)
    group_size = n // k
    remainder = n % k
    group_capacities = [group_size + 1 if i < remainder else group_size for i in range(k)]

    # First KMeans is used to find cluster centres
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=random.randint(1, 10000)).fit(coords)
    centers = kmeans.cluster_centers_

    # Create the matrix of the distances of every point to every cluster centre
    D = cdist(coords, centers, metric='sqeuclidean')

    # Use a greedy algorithm, assign each point to the closest cluster centre while ensuring size constraints
    assignments = [-1] * n
    group_counts = [0] * k

    for i in np.argsort(D.min(axis=1)):  # prioritize points close to some center
        sorted_group_indices = np.argsort(D[i])
        for j in sorted_group_indices:
            if group_counts[j] < group_capacities[j]:
                assignments[i] = j
                group_counts[j] += 1
                break

    # Collect the points to return
    grouped_data = [[] for _ in range(k)]
    for i, group_idx in enumerate(assignments):
        grouped_data[group_idx].append(data[i])

    return grouped_data

def determine_pvalue(distance_delta, num_groups, num_iters_bootstrap, num_iters_groups):
    total_p = 0
    for _ in range(num_iters_groups):
        grouped_data = groups_to_dict(greedy_iteration(distance_delta, num_groups))
        bootstrap_results = np.array(bootstrap(num_iters_bootstrap, grouped_data))
        pval = np.sum(bootstrap_results > 0) / num_iters_bootstrap
        total_p += pval
    
    return total_p / num_iters_groups

def bootstrap_segmentation(gt, seg1, seg2, min_groups, max_groups, num_iters_bootstrap, num_iters_groups, identifier, interval = 100, display=False):
    distance_delta = separation_deltas(gt, seg1, seg2)
    num_pts = len(distance_delta)
    min_group_size = int(num_pts/max_groups)
    max_group_size = int(num_pts/min_groups)
    p_vals = {}
    seen_counts = set()
    for grp_size in range(min_group_size, max_group_size, interval):
        grp_count = num_pts//grp_size

        if grp_count in seen_counts:
            continue

        seen_counts.add(grp_count)
        p_vals[grp_size] = determine_pvalue(distance_delta, grp_count, num_iters_bootstrap, num_iters_groups)

    with open(os.path.join("significance_pickles", identifier + ".pkl"), "wb") as f:
        pickle.dump(p_vals, f)
    
    return p_vals






if __name__ == "__main__":
    data = Data()

    pc_path = r"/media/joshua/Expansion1/tst/BaseLearnerInference"

    training_images = [f"{a:03d}" for a in range(1, 33)]
    all_images = [f"{a:03d}" for a in range(1, 41)]
    testing_images = [f"{a:03d}" for a in range(33, 41)]

    models = ["BasicPlans", "DA5_Segmentations", "LargeEncoder"] 

    seg_path = os.path.join(pc_path, "Probabilities")
    gt_path = os.path.join(pc_path, "GroundTruths")

    data.get_simple_segmentations(seg_path, all_images, ".nrrd")
    data.get_groundtruths(gt_path)

    gt = data.gts["035"]
    seg1 = data.simple_data["035"]["LargeEncoder"]
    seg2 = data.simple_data["035"]["DA5_Segmentations"]
    


    
    a = bootstrap_segmentation(gt, seg1, seg2, 10, 12, 1000, 100, "test")
    print(a)