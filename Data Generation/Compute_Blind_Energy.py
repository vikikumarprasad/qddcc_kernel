#!/usr/bin/env python
# coding: utf-8
import psutil
import sys
# !{sys.executable} --version
# !{sys.executable} -m pip install shap --upgrade 
import joblib
import time
from shutil import copy
import numpy as np

# Compatibility for supplied pandas pickles created with NumPy 2.
if int(np.__version__.split(".", 1)[0]) < 2:
    import numpy.core.multiarray as _numpy_multiarray
    import numpy.core.numeric as _numpy_numeric

    sys.modules.setdefault("numpy._core.multiarray", _numpy_multiarray)
    sys.modules.setdefault("numpy._core.numeric", _numpy_numeric)

import pandas as pd
#import tensorflow as tf
import os
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
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
from scipy.special import rel_entr
from scipy.stats import entropy

from tqdm.notebook import tqdm
import seaborn as sns
from collections import Counter

from glob import glob
import psi4
from helper_CC_ML_spacial import *
import csv

# # SHAP
# import shap
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

file_num = os.getenv("FILE_NUM")
structure = os.getenv("STRUCTURE")
basis_set = os.getenv("BASIS")
# file_num = 2
# structure = "Benzene"
# basis_set = "STO-3G"

availablememory = 6 # psutil.virtual_memory().available / (1024.0 ** 3)
threads_count = int(psutil.cpu_count() * 0.75)

print(f"Running on {threads_count} threads using {availablememory} Gb")
properties=['Evir1', 'Hvir1', 'Jvir1', 'Kvir1', 'Evir2', 'Hvir2', 'Jvir2', 'Kvir2', 'Eocc1', 'Jocc1', 'Kocc1', 'Hocc1','Eocc2', 'Jocc2', 'Kocc2', 'Hocc2', 'Jia1', 'Jia2', 'Kia1', 'Kia2','diag', 'orbdiff', 'doublecheck', 't2start', 't2mag', 't2sign', 'Jia1mag', 'Jia2mag','Kia1mag', 'Kia2mag','t2']
# properties = ['t2']
# data_dir = 'PrasadData'


molecule_file = f"{structure}"
data_path = rf"data/geometries/blind/{molecule_file}.xyz"
out_path = rf"./data/blind/{structure}/"

print(data_path)

try:
    os.makedirs(out_path)
    os.makedirs(out_path+"output/")
    os.makedirs(out_path+"pickle/")
    os.makedirs(out_path+"scaled/")
except:
    pass

with open(data_path,'r') as f:
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
# psi4.core.set_output_file(out_path+'output/output.dat', True)
psi4.core.set_num_threads(threads_count)
psi4.set_options({'basis': basis_set,
                    'scf_type':     'pk',
                    'reference':    'rhf',
                    'mp2_type':     'conv',
                    'e_convergence': 1e-8,
                    'd_convergence': 1e-8})
mol.reset_point_group('C1')
written_mp2 = False
try:
    rhf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
    scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
    print(rhf_e, scf_e)

    t1_path = os.path.join(out_path + "pickle/t1_" + molecule_file + "_alldata" + ".pkl.gz")
    t2_path = os.path.join(out_path + "pickle/" + molecule_file + "_alldata" + ".pkl.gz")

    ml_data = pd.DataFrame(pd.read_csv("paper_results/blind/Expected_Predicted.csv"))
    ml_data = ml_data[ml_data["Molecule"] == molecule_file]
    
    ml_data['Circuit_Type'] = ml_data['Circuit'].str.split('_').str[0]

    # Create a positional index within each Circuit
    ml_data['Positional_Index'] = ml_data.groupby(['Circuit']).cumcount()


    # Group by Circuit_Type and Positional_Index and calculate the mean for both Expected and Predicted values
    ml_data = ml_data.groupby(['Circuit_Type', 'Positional_Index']).mean(numeric_only=True).reset_index()
    
    # Unstack the data to create columns for each circuit
    ml_data_predicted = ml_data.pivot(index='Positional_Index', columns='Circuit_Type', values='Predicted')
    
    print(ml_data_predicted.head(20))
    # Get a list of unique circuits to run
    circuits_to_run = ml_data_predicted.columns.unique()
    
    for circuit in circuits_to_run:
        mp2_data = pd.read_pickle(t2_path)["t2start"].to_numpy()
        t2_data = pd.read_pickle(t2_path)["t2"].to_numpy()

        print("-" * 60)
        print(f"Running on {circuit}")

        # Get the averaged predicted data for the current circuit
        ml_model_data = ml_data_predicted[circuit].to_numpy()
        molecule_t1 = pd.read_pickle(t1_path)["t1"].to_numpy()

        mae = mean_absolute_error(t2_data.flatten(), ml_model_data.flatten())
        print(f"{molecule_file} MAE: {mae:.3e}")
        print(f"${molecule_file} R^2: {r2_score(t2_data.flatten(), ml_model_data.flatten())}")
        
        # Instantiate HelperCCEnergy for each circuit
        A = HelperCCEnergy(mol, rhf_e, scf_wfn, freeze_core=True)
        B = HelperCCEnergy(mol, rhf_e, scf_wfn, freeze_core=True)
        C = HelperCCEnergy(mol, rhf_e, scf_wfn, freeze_core=True)

        A.t2 = mp2_data.reshape(A.t2.shape)
        A.t1 = molecule_t1.reshape(A.t1.shape)
        A.compute_energy(iterate=True)

        B.t2 = t2_data.reshape(B.t2.shape)
        B.t1 = molecule_t1.reshape(B.t1.shape)
        B.compute_energy(iterate=True)

        C.t2 = ml_model_data.reshape(C.t2.shape)
        C.t1 = molecule_t1.reshape(C.t1.shape)
        C.compute_energy(iterate=True)

        mp2_eng = A.FinalEnergy
        ml_eng = C.FinalEnergy
        t2_eng = B.FinalEnergy

        print(rf"{molecule_file} E_MP2: {mp2_eng}")
        print(rf"{molecule_file} {circuit}: {ml_eng}")
        print(rf"{molecule_file} E_CCSD: {t2_eng}")
        print(rf"MP2 Delta E = {t2_eng - mp2_eng:.3e}")
        print(rf"DDCC {circuit} Delta E = {t2_eng - ml_eng:.3e}")
        
        out_path = "paper_results/blind/Energy_MAE_Data.csv"
        if not written_mp2:
            written_mp2 = True
            with open(out_path, mode='a', newline='') as file:
                fieldnames = ['Molecule', 'Circuit', 't2 MAE', 'Energy Devience']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                # Append the label and value as a new row
                # Check if the file is empty to write the header
                if file.tell() == 0:
                    writer.writeheader()
                # Create a dictionary with the data
                
                data_row = {
                    'Molecule': molecule_file,
                    'Circuit': "MP2",
                    't2 MAE': mean_absolute_error(t2_data.flatten(), mp2_data.flatten()),
                    'Energy Devience': np.abs(t2_eng - mp2_eng)
                }
                
                # Write the dictionary to the file
                writer.writerow(data_row)

        with open(out_path, mode='a', newline='') as file:
            fieldnames = ['Molecule', 'Circuit', 't2 MAE', 'Energy Devience']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            # Append the label and value as a new row
            # Check if the file is empty to write the header
            if file.tell() == 0:
                writer.writeheader()
            # Create a dictionary with the data
            
            data_row = {
                'Molecule': molecule_file,
                'Circuit': f"{circuit}",
                't2 MAE': mae,
                'Energy Devience': np.abs(t2_eng - ml_eng)
            }
            
            # Write the dictionary to the file
            writer.writerow(data_row)
            
        print("-" * 60)
        psi4.core.clean()

except Exception as e:
    print(e)
