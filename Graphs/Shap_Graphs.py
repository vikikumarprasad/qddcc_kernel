import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qddcc-mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "qddcc-cache"))

import joblib
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import matplotlib.pyplot as plt

from sklearn.kernel_ridge import KernelRidge
#from Circuit_Classes import ZZ_featuremap
#from QiskitRegressor import *
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
#from squlearn.kernel import QKRR
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
import itertools
#from FunctionEncoder import *
import csv
from sklearn.ensemble import RandomForestRegressor
import shap

molecule_structure = "Water"
scale = "Water-Scaled"
basis = "STO-3G"
num_features = 5

molecules = molecule_structure.split("-")

total_X_test = np.array([])
total_y_test = np.array([])
total_X_train = np.array([])
total_y_train = np.array([])
for molecule_struct in molecules:
    print(molecule_struct)
    top=os.getcwd()
    if scale != "Water-Scaled":
        with open(rf'data/training/{basis}/{molecule_struct}/scaled/{num_features}_feature_train.bin','rb') as f:
            ddcc_train = joblib.load(f)

        with open(rf'data/training/{basis}/{molecule_struct}/scaled/{num_features}_feature_test.bin','rb') as f:
            ddcc_test = joblib.load(f)

        with open(rf'data/training/{basis}/{molecule_struct}/scaled/t2_scaler.bin','rb') as f:
            ddcc_scaler = joblib.load(f)
    else:
        with open(rf'data/training/{basis}/{molecule_struct}/scaled/full_feature_train.bin','rb') as f:
            ddcc_train = joblib.load(f)

        with open(rf'data/training/{basis}/{molecule_struct}/scaled/full_feature_test.bin','rb') as f:
            ddcc_test = joblib.load(f)

        with open(rf'data/training/{basis}/{molecule_struct}/scaled/t2_scaler.bin','rb') as f:
            ddcc_scaler = joblib.load(f)

    # print(ddcc_train.keys())
    X_ddcc_train, y_ddcc_train = ddcc_train['X'],ddcc_train['y']
    X_ddcc_test, y_ddcc_test = ddcc_test['X'],ddcc_test['y']

    X_train, y_train = X_ddcc_train, y_ddcc_train
    X_test, y_test = X_ddcc_test, y_ddcc_test

    if scale=="Water-Scaled":
        print("Scaling data")
        with open(rf"data/training/{basis}/Water/scaled/feature_scaler.bin", 'rb') as f:
            water_feature_scaler = joblib.load(f)
        with open(rf"data/training/{basis}/Water/scaled/t2_scaler.bin", 'rb') as f:
            water_t2_scaler = joblib.load(f)
        with open(rf"data/training/{basis}/{molecule_struct}/scaled/feature_scaler.bin", 'rb') as f:
            molecule_feature_scaler = joblib.load(f)
        with open(rf"data/training/{basis}/{molecule_struct}/scaled/t2_scaler.bin", 'rb') as f:
            molecule_t2_scaler = joblib.load(f)
        print(water_feature_scaler, water_t2_scaler)
        print(molecule_feature_scaler, molecule_t2_scaler)
        X_train = molecule_feature_scaler.inverse_transform(X_train)
        y_train = molecule_t2_scaler.inverse_transform(y_train.reshape(-1, 1))

        X_test = molecule_feature_scaler.inverse_transform(X_test)
        y_test = molecule_t2_scaler.inverse_transform(y_test.reshape(-1, 1))

        top5=np.array([20, 21, 24, 23, 22][::-1])
        top10=np.array([28, 8, 13, 26, 12, 20, 21, 24, 23, 22][::-1])
        top15=np.array([18, 5, 9, 29, 27, 28, 8, 13, 26, 12, 20, 21, 24, 23, 22][::-1])
        print(molecule_structure, X_train.shape, X_test.shape)

        if len(total_X_train) == 0:
            total_y_train = y_train
            total_y_test = y_test
            total_X_train = X_train
            total_X_test = X_test
        else:
            total_y_train = np.append(total_y_train, y_train)
            total_y_test = np.append(total_y_test, y_test)
            total_X_train = np.vstack((total_X_train, X_train))
            total_X_test = np.vstack((total_X_test, X_test))
X_train = total_X_train
y_train = total_y_train
X_test = total_X_test
y_test = total_y_test

print(len(X_train), X_train[0].shape, X_train[-1].shape)
print(len(y_train), y_train[0].shape, y_train[-1].shape)

print(X_test.shape)

properties=['Evir1', 'Hvir1', 'Jvir1', 'Kvir1', 'Evir2', 'Hvir2', 'Jvir2', 'Kvir2', 'Eocc1', 'Jocc1', 'Kocc1', 'Hocc1','Eocc2', 'Jocc2', 'Kocc2', 'Hocc2', 'Jia1', 'Jia2', 'Kia1', 'Kia2','diag', 'orbdiff', 'doublecheck', 't2start', 't2mag', 't2sign', 'Jia1mag', 'Jia2mag','Kia1mag', 'Kia2mag','t2']

rfr=RandomForestRegressor()

model = RandomForestRegressor(**{'bootstrap': True,
'ccp_alpha': 0.0,
'criterion': 'squared_error',
'max_depth': None,
'max_features': 1.0,
'max_leaf_nodes': None,
'max_samples': None,
'min_impurity_decrease': 0.0,
'min_samples_leaf': 1,
'min_samples_split': 2,
'min_weight_fraction_leaf': 0.0,
'monotonic_cst': None,
'n_estimators': 300,
'n_jobs': -1,
'oob_score': False,
'random_state': 42,
'verbose': 0,
'warm_start': False}
)

models={}
model.fit(X_train, np.ravel(y_train))
print(model.score(X_train, np.ravel(y_train)))

# Use an interventional background sample to retain the original script's
# background-data semantics while using the efficient tree implementation.
background = shap.sample(np.vstack([X_train, X_test]), 100, random_state=42)
explainer = shap.TreeExplainer(
    model,
    data=background,
    feature_perturbation="interventional",
    feature_names=properties[:-1],
)
shap_values = explainer(X_test)

selected_features = ["t2start", "t2mag", "doublecheck", "orbdiff", "diag"]

indices = [shap_values.feature_names.index(name) for name in selected_features]
shap_subset = shap_values[:, indices]


y_pred = model.predict(X_test)
print(f"R² score: {r2_score(y_test, y_pred):.6f}")

# SHAP
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{braket}",
    "font.weight": "bold",
    "axes.labelweight": "bold"
})

output_dir = os.getenv("FIGURE_OUTPUT_DIR", "paper_results/regenerated_figures")
os.makedirs(output_dir, exist_ok=True)
# -----------------------------
# 6. Visualizations
# -----------------------------
# Summary plot (global feature importance)
feature_label_map = {
    "diag": r"$\braket{ij \Vert ab}$",
    "orbdiff": r"$\varepsilon_{i} + \varepsilon_{j} - \varepsilon_{a} - \varepsilon_{b}$",
    "doublecheck": r"$\mathrm{Boolean}(a=b)$",
    "t2start": r"$t^{ab}_{ij}(\mathrm{MP2})$",
    "t2mag": r"$\log_{10}|t^{ab}_{ij}(\mathrm{MP2})|$",
}
shap_subset.feature_names = [feature_label_map[name] for name in selected_features]

plt.figure()
#shap.plots.bar(shap_subset, show=False, max_display=10)

vals = shap_subset.values
features = shap_subset.feature_names

if len(vals.shape) > 1:
    vals = np.abs(vals).mean(0)

idx = np.argsort(vals)
plt.barh([features[i] for i in idx], [vals[i] for i in idx], color="#1e88e5")

plt.xlabel(r"Mean $\vert\mathrm{SHAP}\vert$")

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "ccsd_shap_summary_bar_recomputed.png"),
    dpi=300,
)
plt.close()
