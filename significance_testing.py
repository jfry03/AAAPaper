import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from load_data import Data
from models import LogisticRegressor, Model, Analytics, LinearCombo
from sklearn.linear_model import LogisticRegression
import os
import pickle
from Bootstrap import determine_pvalue, bootstrap_segmentation

# ===============Load Data===================
prob_data = Data()
binary_data = Data()

pc_path = r"/media/joshua/Expansion1/tst/BaseLearnerInference"
models = ["BasicPlans", "DA5_Segmentations", "LargeEncoder"]

training_images = [f"{a:03d}" for a in range(1, 33)]
all_images = [f"{a:03d}" for a in range(1, 41)]
testing_images = [f"{a:03d}" for a in range(33, 41)]


seg_path = os.path.join(pc_path, "Probabilities")
gt_path = os.path.join(pc_path, "GroundTruths")

prob_data.get_simple_segmentations(seg_path, all_images, ".npz")
prob_data.get_groundtruths(gt_path)

binary_data.get_simple_segmentations(seg_path, all_images, ".nrrd")
binary_data.get_groundtruths(gt_path)

print("Data Loaded!")

# =====================Train Models/Get Segmentations=====================


logreg = LogisticRegressor(prob_data.simple_data, prob_data.gts, models)

logreg.train_model(training_images, model_params = {"C": 3.1622776601683795e-5})

logreg_predictions = logreg.predict(testing_images)

model_segs = {model: {} for model in models}

model_segs["Ensemble"] = logreg_predictions

for model in models:
    for image in testing_images:
        model_segs[model][image] = binary_data.simple_data[image][model]

print("Models Trained!")

# ==================Compare on 1st Order Metrics=================

model_hd = {}
model_dc = {}
model_sd = {}

for model in model_segs.keys():
    model_analytics = Analytics(model_segs[model], binary_data.gts)
    mean_dice = np.mean(list(model_analytics.dice_coefficients().values()))
    mean_hd = np.mean(list(model_analytics.maximum_hausdorff_distance().values()))
    model_sd[model] = np.mean(list(model_analytics.separation_distances().values()))
    model_hd[model] = mean_hd
    model_dc[model] = mean_dice

results_df = pd.DataFrame({
    "Model": list(model_dc.keys()),
    "Mean Dice": [model_dc[m] for m in model_dc],
    "Mean Hausdorff": [model_hd[m] for m in model_hd],
    "Mean Separation": [model_sd[m] for m in model_sd]
})

p_values = {}
for image in testing_images:
    image_vals = {}
    for model in models:
        ensemble_seg = logreg_predictions[image]
        gt = binary_data.gts[image]
        established_seg = model_segs[model][image]
        p_val = bootstrap_segmentation(gt, established_seg, ensemble_seg, 7,150, 10000, 50, f"{image}{model}", interval = 20)
        image_vals[model] = p_val
    p_values[image] = image_vals
