import numpy as np
import nrrd
import pandas as pd
import os
import methods

pc_path = r"E:\AAAPaperData\nnUNetFrame\BaseLearnerInference"


class Data:
    def __init__(self):
        pass

    def get_detailed_segmentations(self, path, folds = [], models = []):
        data = {fold: {} for fold in folds}
        for model in models:
            model_path = os.path.join(path, model)

            for fold in folds:
                model_fold_path = os.path.join(model_path, fold)

                for image in os.listdir(model_fold_path):
                    if not image.endswith("nrrd"):
                        continue
                    image_num = image[5:8]
                    if image_num not in data[fold]:
                        data[fold][image_num] = {}
                    data[fold][image_num][model] = nrrd.read(os.path.join(model_fold_path, image))[0]
        self.detailed_data = data
    
    def get_simple_segmentations(self, path, images, ending = ".npz"):
        data = {image: {} for image in images}


        for model in os.listdir(path):
            model_path = os.path.join(path, model)

            for fold in os.listdir(model_path):
                fold_path = os.path.join(model_path, fold)

                for image in os.listdir(fold_path):
                    image_num = image[5:8]
                    if image_num not in images or not image.endswith(ending):
                        continue
                    if ending == '.npz':
                        d = np.load(os.path.join(fold_path, image))['probabilities'][1, :, :, :]
                        data[image_num][model] = np.transpose(d, (2, 1, 0))
                    elif ending == '.nrrd':
                        d = nrrd.read(os.path.join(fold_path, image))[0]
                        data[image_num][model] = d
            self.simple_data = data       


    def get_groundtruths(self, path):
        gts = {}
        for fold in os.listdir(path):
            fold_path = os.path.join(path, fold)

            for image in os.listdir(fold_path):
                if not image.endswith(".nrrd"):
                    continue
                image_num = image[5:8]
                gts[image_num] = nrrd.read(os.path.join(fold_path, image))[0]
        self.gts = gts

if __name__ == "__main__":
    print("Testing load_data.py")
    data = Data()

    training_folds = [f"Fold{i}" for i in range(4)] + ["Fold_All"]
    models = ["BasicPlans", "DA5_Segmentations", "LargeEncoder"]

    seg_path = os.path.join(pc_path, "mySegmentations")
    data.get_detailed_segmentations(seg_path, training_folds, models)

    gt_path = os.path.join(pc_path, "GroundTruths")

    data.get_groundtruths(gt_path)

    print(data.gts.keys())