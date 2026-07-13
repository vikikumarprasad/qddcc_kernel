#Molecule,Circuit,Expected,Predicted,t2_num
# Water0,Classical_0_1_0,-0.032258810320823464,-0.035648776611737855,0
# Water0,Classical_0_1_0,7.343018212450086e-06,-0.00038598220468590206,1
# Water0,Classical_0_1_0,7.343018212488334e-06,-0.0003859822046873009,2
# Water0,Classical_0_1_0,-0.016898595153658564,-0.018623180877013833,3
# Water0,Classical_0_1_0,-2.029232148243412e-05,0.0018392283314581069,4
# Water0,Classical_0_1_0,-0.03180956195801589,-0.034833146990988884,5

basis = "STO-3G" #"cc-pVDZ" #
percents = [0.01, 0.02, 0.03, 0.04, 0.05]
feature_distr = [5]
iterate = False

chunk_size = 2000 * 64  # Number of rows per chunk

import psutil
import sys
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
import os
import io
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

import re

def get_numbers_at_end(string):
    match = re.search(r'\d+$', string)
    return int(match.group()) if match else None

# Define the expected number of elements per row
expected_columns = 5  # Adjust this to your specific case

def read_valid_csv(file_path):
    # """
    # Reads a CSV file, skipping any rows that do not have exactly 5 columns.

    # Args:
    #     file_path (str): The path to the CSV file.

    # Returns:
    #     pd.DataFrame: A DataFrame containing only the rows with 5 columns.
    # """

    try:
        # This reads and returns the TextFileReader iterable directly
        df_iterator = pd.read_csv(file_path, 
                                  chunksize=chunk_size, 
                                  header=0) # Assuming the first row is a header
        return df_iterator
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return iter([]) # Return an empty iterator on failure

if basis == "STO-3G":
    test_molecules = ['Water66', 'Water168', 'Water33', 'Water96', 'Water185', 'Water37', 'Water147', 'Water188', 'Water98', 'Water81', 'Water70', 'Water14', 'Water184', 'Water164', 'Water27', 'Water51', 'Water23', 'Water191', 'Water20', 'Water99', 'Water0', 'Water145', 'Water79', 'Water196', 'Water197', 'Water141', 'Water28', 'Water2', 'Water115', 'Water130', 'Water178', 'Water47', 'Water63', 'Water111', 'Water190', 'Water148', 'Water150', 'Water18', 'Water94', 'Water60']
else:
    test_molecules = ['Water156', 'Water195', 'Water73', 'Water120', 'Water40', 'Water138', 'Water0', 'Water133', 'Water146', 'Water169', 'Water80', 'Water107', 'Water197', 'Water83', 'Water130', 'Water181', 'Water36', 'Water87', 'Water164', 'Water43', 'Water145', 'Water110', 'Water153', 'Water118', 'Water190', 'Water117', 'Water132', 'Water15', 'Water174', 'Water180', 'Water187', 'Water159', 'Water177', 'Water39', 'Water22', 'Water193', 'Water54', 'Water113', 'Water8', 'Water75']

conv_distance = 1e-7

for num_features in feature_distr:
    for percent in percents:
        written_mp2 = False
        written_mp2_conv = False
        # print(percent)
        try:
            # large_data_partial = read_valid_csv(f"paper_results/water_energy/{percent}-{num_features}-{basis}-Water-R2.csv")
            large_data_partial = pd.read_csv(
                f"paper_results/water_energy/{percent}-{num_features}-{basis}-Water-R2.csv",
                chunksize=chunk_size, 
                header=0, # Assuming first row is header
                on_bad_lines='skip', # This is the key fix for the ParserError
                sep=',',
                names=['Molecule','Circuit','Expected','Predicted','t2_num'] # Explicitly name columns
            )
        except Exception as e:
            print(e)
            print(f"{percent}-{num_features}-{basis}")
            with open("file_errors.csv", mode='a', newline='') as file:
                    writer = csv.writer(file)

                    writer.writerow([f"{percent}-{num_features}-{basis}, {e}"])
            continue
        for chunk in large_data_partial:
            data_partial = chunk
            print(data_partial)
            circuits = data_partial["Circuit"].unique()
            print(circuits)
            for circuit in circuits:
                if not isinstance(circuit, str):
                    continue
                if not "_0_" in circuit:
                    circuits = circuits[~np.isin(circuits, [circuit])]
                    continue
                
                temp_df = data_partial[data_partial["Circuit"] == circuit].copy()
                if basis == "STO-3G":
                    num_t2s = 64
                elif basis == "cc-pVDZ":
                    num_t2s = 5776
                else:
                    continue

                print(circuit)
                print(temp_df.head(10))

                try:
                    r2_value = r2_score(temp_df["Expected"], temp_df["Predicted"])
                except Exception as e:
                    print(e)
                    continue
                print(f"{percent}{circuit} - R2: {r2_value}")
                molecules = data_partial["Molecule"].unique()

                energy_deviences = []
                for molecule in molecules:
                    if "Classical" in circuit:
                        written_mp2_conv = False
                    else:
                        written_mp2_conv = True
                    if not isinstance(molecule, str):
                        print(molecule)
                        time.sleep(1)
                        continue
                    if not "Water" in molecule:
                        continue
                    
                    index = int(get_numbers_at_end(molecule))
                    molecule_number = get_numbers_at_end(test_molecules[index]) + 2
                    
                    molecule_file = f"Water{molecule_number}"

                    availablememory = 6 # psutil.virtual_memory().available / (1024.0 ** 3)
                    threads_count = int(psutil.cpu_count() * 0.75)
                    data_path = f"data/geometries/training/water/{molecule_file.lower()}.xyz"
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

                    psi4.core.set_num_threads(threads_count)
                    psi4.set_options({'basis': basis,
                                        'scf_type':     'pk',
                                        'reference':    'rhf',
                                        'mp2_type':     'conv',
                                        'e_convergence': 1e-8,
                                        'd_convergence': 1e-8})
                    mol.reset_point_group('C1')

                    rhf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
                    scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
                    print(rhf_e, scf_e)

                    t1_path = os.path.join(f"data/training/{basis}/Water/pickle/t1_" + str(molecule_file) + "_alldata" + ".pkl.gz")
                    t2_path = os.path.join(f"data/training/{basis}/Water/pickle/" + str(molecule_file.lower()) + "_alldata" + ".pkl.gz")
                    # t2_path = os.path.join(out_path + "pickle/" + molecule_file + "_alldata" + ".pkl.gz")

                    mp2_data = pd.read_pickle(t2_path)["t2start"].to_numpy()
                    # t2_data = pd.read_pickle(t2_path)["t2"].to_numpy()

                    print("-" * 60)
                    print(f"Running on {molecule_file}")

                    # Get the averaged predicted data for the current circuit
                    # ml_model_data = ml_data_predicted[circuit].to_numpy()
                    molecule_df = temp_df[temp_df["Molecule"] == molecule]
                    molecule_t1 = pd.read_pickle(t1_path)["t1"].to_numpy()

                    ml_data_predicted = molecule_df["Predicted"].to_numpy().astype(float)
                    ml_data_expected = molecule_df["Expected"].to_numpy().astype(float)
                    if (ml_data_predicted.shape[0] != num_t2s):
                        continue
                    # Instantiate HelperCCEnergy for each circuit
                    A = HelperCCEnergy(mol, rhf_e, scf_wfn, freeze_core=True)
                    B = HelperCCEnergy(mol, rhf_e, scf_wfn, freeze_core=True)
                    C = HelperCCEnergy(mol, rhf_e, scf_wfn, freeze_core=True)

                    A.t2 = mp2_data.reshape(A.t2.shape)
                    # A.t1 = molecule_t1.reshape(A.t1.shape)
                    if not written_mp2_conv:
                        mp2_hist = A.compute_energy(e_conv = conv_distance, r_conv = conv_distance, iterate=iterate)
                    else:
                        A.compute_energy(iterate=False)

                    B.t2 = ml_data_expected.reshape(B.t2.shape)
                    B.t1 = molecule_t1.reshape(B.t1.shape)
                    B.compute_energy(e_conv = conv_distance, r_conv = conv_distance, iterate=False)

                    C.t2 = ml_data_predicted.reshape(C.t2.shape)
                    # C.t1 = molecule_t1.reshape(C.t1.shape)
                    ml_hist = C.compute_energy(e_conv = conv_distance, r_conv = conv_distance, iterate=iterate)

                    mp2_eng = A.FinalEnergy
                    ml_eng = C.FinalEnergy
                    t2_eng = B.FinalEnergy

                    print(rf"{molecule_file} E_MP2: {mp2_eng}")
                    print(rf"{molecule_file} {circuit}: {ml_eng}")
                    print(rf"{molecule_file} E_CCSD: {t2_eng}")
                    print(rf"MP2 Delta E = {t2_eng - mp2_eng:.3e}")
                    print(rf"DDCC {circuit} Delta E = {t2_eng - ml_eng:.3e}")

                    if not iterate:
                        out_path = f"paper_results/energy/{basis}_{num_features}_Energy_Convergence.csv"
                        if not written_mp2_conv:
                            written_mp2_conv = True
                            with open(out_path, mode='a', newline='') as file:
                                fieldnames = ['Percent of Train', 'Circuit', 'Energy_Hist', 'Iteration']
                                writer = csv.DictWriter(file, fieldnames=fieldnames)
                                # Append the label and value as a new row
                                # Check if the file is empty to write the header
                                if file.tell() == 0:
                                    writer.writeheader()
                                # Create a dictionary with the data
                                
                                for iteration in range(len(mp2_hist)):
                                    data_row = {
                                        'Percent of Train': percent,
                                        'Circuit': f"MP2",
                                        'Energy_Hist': mp2_hist[iteration],
                                        'Iteration': iteration
                                    }
                                    
                                    # Write the dictionary to the file
                                    writer.writerow(data_row)
                                data_row = {
                                        'Percent of Train': percent,
                                        'Circuit': f"CCSD",
                                        'Energy_Hist': t2_eng,
                                        'Iteration': 0
                                    }
                                    
                                # Write the dictionary to the file
                                writer.writerow(data_row)


                        with open(out_path, mode='a', newline='') as file:
                            fieldnames = ['Percent of Train', 'Circuit', 'Energy_Hist', 'Iteration']
                            writer = csv.DictWriter(file, fieldnames=fieldnames)
                            # Append the label and value as a new row
                            # Check if the file is empty to write the header
                            if file.tell() == 0:
                                writer.writeheader()
                            # Create a dictionary with the data
                            
                            for iteration in range(len(ml_hist)):
                                data_row = {
                                    'Percent of Train': percent,
                                    'Circuit': f"{circuit}",
                                    'Energy_Hist': ml_hist[iteration],
                                    'Iteration': iteration
                                }
                                
                                # Write the dictionary to the file
                                writer.writerow(data_row)
                    energy_deviences.append([t2_eng, ml_eng, mp2_eng])
                    psi4.core.clean()
                    # except Exception as e:
                    #     print(e)
                print(energy_deviences)
                energy_deviences = np.array(energy_deviences)

                if energy_deviences.shape[0] == 0:
                    continue
                
                ml_mean_abs_error = mean_absolute_error(energy_deviences[:, 0], energy_deviences[:, 1])
                print(f"{percent}-{circuit} MAE: {ml_mean_abs_error:.3e}")

                out_path = f"paper_results/energy/{basis}_{num_features}_Water_Energy_and_MAE_Data.csv"
                if not written_mp2:
                    written_mp2 = True
                    mp2_mean_abs_error =  mean_absolute_error(energy_deviences[:, 0], energy_deviences[:, 2])
                    with open(out_path, mode='a', newline='') as file:
                        fieldnames = ['Percent of Train', 'Circuit', 'R2', 'MAE Energy Devience']
                        writer = csv.DictWriter(file, fieldnames=fieldnames)
                        # Append the label and value as a new row
                        # Check if the file is empty to write the header
                        if file.tell() == 0:
                            writer.writeheader()
                        # Create a dictionary with the data
                        
                        data_row = {
                            'Percent of Train': percent,
                            'Circuit': "MP2",
                            'R2': -50,
                            'MAE Energy Devience': mp2_mean_abs_error,
                        }
                        
                        # Write the dictionary to the file
                        writer.writerow(data_row)

                with open(out_path, mode='a', newline='') as file:
                    fieldnames = ['Percent of Train', 'Circuit', 'R2', 'MAE Energy Devience']
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    # Append the label and value as a new row
                    # Check if the file is empty to write the header
                    if file.tell() == 0:
                        writer.writeheader()
                    # Create a dictionary with the data
                    
                    data_row = {
                        'Percent of Train': percent,
                        'Circuit': circuit,
                        'R2': r2_value,
                        'MAE Energy Devience': ml_mean_abs_error
                    }
                    
                    # Write the dictionary to the file
                    writer.writerow(data_row)
            print(circuits)
psi4.core.clean()
