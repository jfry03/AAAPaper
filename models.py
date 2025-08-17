from load_data import Data
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import methods
import os
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from scipy.ndimage import label, generate_binary_structure
from sklearn.svm import SVC



class Model:
    def __init__(self, segmentation_data, groundtruths, models):
        self.segmentation_data = segmentation_data
        self.groundtruths = groundtruths
        self.models = models
        self.model = None

    def build_feature_vector(self, images):
        X = []
        for model in self.models:
            feature = []
            for image in images:
                feature += self.segmentation_data[image][model].flatten().tolist()
            X.append(feature)
        
        y = []
        for image in images:
            y += self.groundtruths[image].flatten().tolist()
        
        return np.array(X).T, np.array(y)
    
    def predict(self, test_images, remove_islands=True):
        X, _ = self.build_feature_vector(test_images)
        predictions = self.model.predict(X)

        # Reshape predictions back to image format
        result_dict = {}
        pixel_idx = 0

        for image in test_images:
            if image in self.segmentation_data:
                original_shape = self.segmentation_data[image][self.models[0]].shape
                num_pixels = np.prod(original_shape)
                image_predictions = predictions[pixel_idx:pixel_idx + num_pixels]
                new_img = image_predictions.reshape(original_shape)
                if remove_islands:
                    new_img = self.keep_largest_component(new_img)
                result_dict[image] = new_img
                pixel_idx += num_pixels
            

        return result_dict
    
        
    @staticmethod
    def keep_largest_component(arr, full_connectivity=False, return_mask=False):
        """
        Keep only the largest connected component of a binary array.
        
        Parameters
        ----------
        arr : np.ndarray
            Input array. Nonzero values are treated as foreground.
        full_connectivity : bool
            If False: 4-connectivity (2D) / 6-connectivity (3D).
            If True : 8-connectivity (2D) / 26-connectivity (3D).
        return_mask : bool
            If True, return a boolean mask. If False, return array with only largest
            component kept (same dtype as input, others set to 0).
        """
        # Ensure boolean foreground
        fg = (arr != 0)

        if arr.ndim < 1:
            raise ValueError("Array must be at least 1D.")
        if arr.ndim > 3:
            # Works for ND, but typical use is 2D/3D
            pass

        # Connectivity structure (order=1 = minimal, order=arr.ndim = full)
        order = arr.ndim if full_connectivity else 1
        structure = generate_binary_structure(arr.ndim, order)

        labeled, n = label(fg, structure=structure)
        if n == 0:
            # No foreground
            return np.zeros_like(fg if return_mask else arr)

        # Count pixels in each component; ignore background (label 0)
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        largest_label = counts.argmax()

        mask = (labeled == largest_label)
        if return_mask:
            return mask
        out = np.zeros_like(arr)
        out[mask] = arr[mask]
        return out


class LogisticRegressor(Model):
    def __init__(self,segmentation_data, groundtruths, models):
        super().__init__(segmentation_data, groundtruths, models)

    def train_model(self, training_images, model_params={}):
        X, y = self.build_feature_vector(training_images)


        self.model = LogisticRegression(**model_params)
        self.model.fit(X, y)
        return self.model

class SVMClassifier(Model):
    def __init__(self,segmentation_data, groundtruths, models):
        super().__init__(segmentation_data, groundtruths, models)
    
    def train_model(self, training_images, model_params = {}):
        X, y = self.build_feature_vector(training_images)

        self.model = SVC(**model_params)
        self.model.fit(X, y)

        return self.model

class LinearCombo(Model):
    def __init__(self,segmentation_data, groundtruths, models):
        super().__init__(segmentation_data, groundtruths, models)
    
    def train_model(self, training_images, weights = None):

        if weights is None:
            weights = [1/(len(models)) for _ in range(len(models))]
        
        base_copy = np.zeros_like()
        pass
    
    def predict_linear(self, training_images, weights = None):
        results = {}
        if weights is None:
            weights = {model: 1/len(self.models) for model in self.models}
        for image in self.segmentation_data:
            label = np.zeros_like(self.segmentation_data[image]["DA5_Segmentations"])
            for model in self.segmentation_data[image]:
                label = label + weights[model] * self.segmentation_data[image][model]
            results[image] = label
        return results

class Analytics:
    def __init__(self, predictions, gts):
        self.predictions = predictions
        self.gts = gts

    @staticmethod
    def _dc(prediction, gt):
        intersection = np.sum(prediction * gt)
        union = np.sum(prediction) + np.sum(gt)
        if union == 0:
            return 1.0
        return 2.0 * intersection / union
    
    def threshold(self, threshold_value=0.5):
        thresholded = {}
        for img in self.predictions.keys():
            thresholded[img] = (self.predictions[img] > threshold_value).astype(int)
        return thresholded

    def dice_coefficients(self):
        dices = {}
        for image in self.predictions.keys():
            dices[image] = self._dc(self.predictions[image], self.gts[image])

        return dices
    @staticmethod
    def extract_surface(mask):
        return mask ^ binary_erosion(mask)
    @staticmethod
    def voxel_coords(binary_mask, spacing = (1.0, 1.0, 1.0)):
        indices = np.argwhere(binary_mask)
        return indices * np.array(spacing), indices
    
    def maximum_hausdorff_distance(self):
        maximums = {}
        for img, pred in self.predictions.items():
            gt = self.gts[img]
            pred_surface = self.extract_surface(pred>0)
            gt_surface = self.extract_surface(gt > 0)
            pred_pts, _ = self.voxel_coords(pred_surface)
            gt_pts, _ = self.voxel_coords(gt_surface)

            pred_tree = cKDTree(pred_pts)
            gt_tree = cKDTree(gt_pts)
            
            d_gt_pred = pred_tree.query(gt_pts, k=1)[0]
            d_pred_gt = gt_tree.query(pred_pts, k=1)[0]

            maximums[img] = max(list(d_pred_gt) + list(d_gt_pred))
        
        return maximums
    
    def separation_distances(self, return_pts = False):
        separation_distances = {}
        for img, pred in self.predictions.items():
            gt = self.gts[img]

            pred_surface = self.extract_surface(pred > 0)
            gt_surface = self.extract_surface(gt > 0)

            pred_pts_phys, _ = self.voxel_coords(pred_surface)
            gt_pts_phys, _ = self.voxel_coords(gt_surface)

            pred_tree = cKDTree(pred_pts_phys)
            d_gt_to_pred, _ = pred_tree.query(gt_pts_phys)

            if return_pts:
                separation_distances[img] = list(zip(gt_pts_phys, d_gt_to_pred))
            else:
                separation_distances[img] = np.mean(d_gt_to_pred)

        return separation_distances



if __name__ == "__main__":
    
    print("Testing load_data.py")
    data = Data()
    pc_path = r"E:\AAAPaperData\nnUNetFrame\BaseLearnerInference"

    models = ["BasicPlans", "DA5_Segmentations", "LargeEncoder"]

    seg_path = os.path.join(pc_path, "mySegmentations")

    gt_path = os.path.join(pc_path, "GroundTruths")

    images = [f"{a:03d}" for a in range(1, 33)]
    data.get_simple_segmentations(seg_path, images, models)
    data.get_groundtruths(gt_path)

    logreg = LogisticRegressor(data.simple_data, data.gts, models)
    print(data.gts.keys())
    logreg.train_model(images)

