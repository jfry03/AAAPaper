import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from load_data import Data
import os
import random
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


class BlockBootstrap:
    def __init__(self):
        pass

    def bootstrap(self, iterations, pt_data):
        """
        pt_data is a dictionary with values different groups
        """
    
        keys = list(pt_data.keys())
        blocks = [np.asarray(pt_data[k]) for k in keys]
        n_keys = len(keys)

        sample_idx = np.random.randint(0, n_keys, size = (iterations, n_keys))

        results = []

        for row in sample_idx:
            sample = np.concatenate([blocks[i] for i in row])
            results.append(np.mean(sample))
        
        return results
    
    def groups_to_dict(self, grouped_data):
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
    
    @staticmethod
    def extract_surface(mask):
        return mask ^ binary_erosion(mask)
    @staticmethod
    def voxel_coords(binary_mask, spacing = (1.0, 1.0, 1.0)):
        indices = np.argwhere(binary_mask)
        return indices * np.array(spacing), indices

    
    def separation_distances(self, gt, pred):

        pred_surface = self.extract_surface(pred > 0)
        gt_surface = self.extract_surface(gt > 0)

        pred_pts_phys, _ = self.voxel_coords(pred_surface)
        gt_pts_phys, _ = self.voxel_coords(gt_surface)

        pred_tree = cKDTree(pred_pts_phys)
        d_gt_to_pred, _ = pred_tree.query(gt_pts_phys)

        return list(zip(gt_pts_phys, d_gt_to_pred))

    def seperation_deltas(self, gt, seg1, seg2):
        sep1 = self.separation_distances(gt, seg1)  # You'll need to pass gt
        sep2 = self.separation_distances(gt, seg2)
        
        # Extract just the distances (second element of each tuple)
        coords1 = np.array([coord for coord, _ in sep1])
        distances1 = np.array([dist for _, dist in sep1])
        
        coords2 = np.array([coord for coord, _ in sep2])
        distances2 = np.array([dist for _, dist in sep2])
        
        # Calculate deltas"""  """
        deltas = distances1 - distances2
        
        # Return coordinates with their delta values
        return list(zip(coords1, deltas))
    
    def greedy_iteration(self, 
        data: List[Tuple[Tuple[float, float, float], float]],
        k: int
    ) -> List[List[Tuple[Tuple[float, float, float], float]]]:
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
    

if  __name__ == "__main__":
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
    seg1 = data.simple_data["035"]["BasicPlans"]
    seg2 = data.simple_data["035"]["DA5_Segmentations"]
    b = BlockBootstrap()
    b2 = b.separation_distances(gt, seg1)
    b3 = b.separation_distances(gt, seg2)

    print(np.mean([k[1] for k in b2]))

    print(np.mean([k[1] for k in b3]))

    b4 = b.seperation_deltas(gt, seg1, seg2)

    print(np.mean([k[1] for k in b4]))


    #print(b.seperation_deltas(seg1, seg2))

    b5 = b.greedy_iteration(b4, 10)
    grp_means = []

    # Calculate mean for each group
    for i, group in enumerate(b5):
        # Extract the delta values (second element of each tuple)
        group_deltas = [val for coord, val in group]
        group_mean = np.mean(group_deltas)
        grp_means.append(group_mean)

    print(f"\nOverall group means: {grp_means}")
    print(f"Mean of group means: {np.mean(grp_means):.4f}")

    b6 = b.groups_to_dict(b5)
    bootstrap_results = b.bootstrap(100, b6)
    bootstrap_results = np.array(bootstrap_results)  # Convert to numpy array
    proportion_positive = np.sum(bootstrap_results > 0) / len(bootstrap_results)
    print(f"Proportion of samples > 0: {proportion_positive:.3f}")


        



