import psutil
import sys
import joblib
import time
from shutil import copy
import numpy as np
import pandas as pd
import os
from pathlib import Path
import traceback
from matplotlib import cm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from tqdm.notebook import tqdm
import seaborn as sns
from collections import Counter

from glob import glob
import psi4
from helper_CC_ML_spacial import *

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def require_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


file_num = require_env("FILE_NUM")
structure = require_env("STRUCTURE")
basis_set = require_env("BASIS")


availablememory = 6
threads_count = int(psutil.cpu_count() * 0.75)

print(f"Running on {threads_count} threads using {availablememory} Gb")
properties=['Evir1', 'Hvir1', 'Jvir1', 'Kvir1', 'Evir2', 'Hvir2', 'Jvir2', 'Kvir2', 'Eocc1', 'Jocc1', 'Kocc1', 'Hocc1','Eocc2', 'Jocc2', 'Kocc2', 'Hocc2', 'Jia1', 'Jia2', 'Kia1', 'Kia2','diag', 'orbdiff', 'doublecheck', 't2start', 't2mag', 't2sign', 'Jia1mag', 'Jia2mag','Kia1mag', 'Kia2mag','t2']
# properties = ['t2']


geometry_root = Path(os.getenv("GEOMETRY_ROOT", "data/geometries/training"))
training_root = Path(os.getenv("TRAINING_DATA_ROOT", "data/training"))
structure_lower = structure.lower()
geometry_stem = f"{structure_lower}{file_num}"
molecule_file = f"{structure}"
data_path = geometry_root / structure_lower / f"{geometry_stem}.xyz"
out_path = training_root / basis_set / structure

print(data_path)

for subdir in ("", "output", "pickle", "scaled"):
    (out_path / subdir).mkdir(parents=True, exist_ok=True)
(out_path / "failures").mkdir(parents=True, exist_ok=True)

with open(data_path, 'r') as f:
    text=f.read()

xyz=False
if xyz==True: 
    qmol = psi4.qcdb.Molecule.from_string(text, dtype='xyz')
    mol = psi4.geometry(qmol.create_psi4_string_from_molecule()+ 'symmetry c1')                
else:
    # text = "symmetry c1\n\n" + text                       
    mol = psi4.geometry(text)  

psi4.core.clean()
psi4.core.be_quiet()
psi4.set_memory(f'{int(np.floor(availablememory) * 0.8)} GB')
psi4.core.set_output_file(str(out_path / "output" / "output.dat"), True)
psi4.core.set_num_threads(threads_count)
psi4.set_options({'basis': basis_set,
                    'scf_type':     'pk',
                    'reference':    'rhf',
                    'mp2_type':     'conv',
                    'e_convergence': 1e-8,
                    'd_convergence': 1e-8})
mol.reset_point_group('C1')
try:
    rhf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
    scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
    print(rhf_e, scf_e)
    A=HelperCCEnergy(mol, rhf_e, scf_wfn, freeze_core=True)
    
    A.compute_energy()
    print(A.FinalEnergy)

    data = pd.DataFrame(np.array([getattr(A,attr).flatten() for attr in properties]).T,columns=properties)

    pickle_path = out_path / "pickle" / f"{geometry_stem}_alldata.pkl.gz"
    data.to_pickle(pickle_path, compression='gzip')
    t1_pickle_path = out_path / "pickle" / f"t1_{geometry_stem}_alldata.pkl.gz"
    t1_data = pd.DataFrame(np.array([A.t1.flatten()]).T, columns=["t1"])
    t1_data.to_pickle(t1_pickle_path, compression="gzip")

    psi4.core.clean()
except Exception:
    failure_path = out_path / "failures" / f"{geometry_stem}.log"
    failure_path.write_text(traceback.format_exc())
    psi4.core.clean()
    raise
