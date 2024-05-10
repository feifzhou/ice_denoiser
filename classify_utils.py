import numpy as np
import ase.io
from dscribe.descriptors import SOAP
import glob
from collections import defaultdict
from sklearn.decomposition import PCA
import argparse
from sklearn.metrics.pairwise import cosine_distances,euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# Function to calculate SOAP features
def soap_feats(structures, r_cut=12, n_max=10, l_max=8):
    species = set(structures.get_chemical_symbols())
    soap = SOAP(
        species=species,
        periodic=True,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        average="off"
    )
    feature_vectors = soap.create(structures, n_jobs=8)
    feature_vectors = np.array(feature_vectors)
    return feature_vectors

# Function to classify configurations
def classify(feats, feats_classes,pca_transform=False, classify_type='cos',pca=None):
    """
    Classify a given set of features (feats) into one of the known classes (feats_classes) based on similarity scores.

    Args:
        feats (numpy.ndarray): Features of the test atom.
        feats_classes (dict): Dictionary containing features of all Wyckoff atoms for each known class (i.e. ideal bulk structure:'2','1c','1h','3','6','7','sI') .
        pca_transform (bool): Whether to apply PCA transformation on the features before calculating similarity scores.
        classify_type (str): Type of similarity measure to use, either 'cos' (cosine similarity) or 'dist' (Euclidean distance).


    Returns:
        str: Predicted class name based on the highest similarity score.
    """
    dist_classes = []
    class_map = {}
    count = 0
    if pca_transform:
        feats = pca.transform(feats.reshape(1, -1))
    else:
        feats = feats.reshape(1, -1)
  

    # Calculate similarity scores between the test atom and each ideal bulk structure
    for key, feats_class in feats_classes.items():
        class_map[count] = key

        if classify_type == 'cos':
            distances = cosine_distances(feats, feats_class)
        else:
            distances = euclidean_distances(feats, feats_class)
        
        dist_classes.append(np.min(distances))
        count += 1

    sim_score= np.min(dist_classes) 

    class_index = np.argmin(dist_classes)
    class_name = class_map[class_index]

    return class_name,sim_score

def find_bulkliquid_threshold(classify_type):
    """
    Determine the threshold value for distinguishing bulk and liquid phases based on the classification type.

    Args:
        classify_type (str): The type of classification method used ('cos' for cosine distance or 'other' for another method).

    Returns:
        float: The threshold value for distinguishing bulk and liquid phases.
    """
    if classify_type == 'cos':
        # Recommended values for cosine distance are between 0.15 and 0.2
        # Check the distribution plots of bulk and liquid phases in 'plot_hist_interfaces.ipynb'
        bulkliquid_threshold = 0.17
    else:
        # Recommended values for Euclidean distance are between 10 and 15
        # Check the distribution plots of bulk and liquid phases in 'plot_hist_interfaces.ipynb'
        bulkliquid_threshold = 12.72

    # Uncomment the following lines to use k-means clustering to determine the bulk_liquid_threshold
    # data = np.array(sim_score_list)
    # data = data.reshape(-1, 1)  # Reshape data if needed
    #
    # k = 2  # Choose the number of clusters (k)
    # print(data)
    #
    # kmeans = KMeans(n_clusters=k)
    # labels = kmeans.fit_predict(data)
    #
    # # Threshold is the midpoint between the highest value of the first cluster and lowest value of the second cluster
    # threshold = (np.max(data[labels == 0]) + np.min(data[labels == 1])) / 2
    # bulkliquid_threshold = threshold

    return bulkliquid_threshold

def classify_update(min_score_list,class_name_list,threshold=None):
    
    revised_classname_list=class_name_list

    for count,min_score in enumerate(min_score_list):
        if min_score >= threshold:
            revised_classname_list[count]='w'
    return revised_classname_list


def load_initial_structures(data_dir, ideal_struct_list,pca,scaler_transform,scaler):
    """
    Load initial structures from CIF files and calculate SOAP features.

    Args:
        data_dir (str): Path to the directory containing CIF files.
        label_list (list): List of class labels for different ice phases.

    Returns:
        feats_classes (dict): Dictionary containing SOAP features for each class label (i.e. ideal bulk structure:'2','1c','1h','3','6','7','sI' ).
        feats_classes_pca (dict): Dictionary containing PCA-transformed SOAP features for each class label.
    """
    feats_classes = {}
    feats_classes_pca = {}
    atom_size = []
    start = 0
    feats_all = []

    # Calculate SOAP features and PCA-transformed features for each label
    for label in ideal_struct_list:

        files = glob.glob(f'{data_dir}/{label}*.cif')
        atoms = ase.io.read(files[0])
        # only O atoms used for generating the SOAP features
        atoms = atoms[[atom.index for atom in atoms if atom.symbol != 'H']]
        end = start + len(atoms)
        atom_size.append([label, start, end])
        start = end
        feats_atom = soap_feats(atoms)
        feats_classes[label] = feats_atom
        feats_all.extend(list(feats_atom))
    if scaler_transform:
        feats_all=scaler.fit_transform(feats_all)
    feats_pca = pca.fit_transform(feats_all)
    for label, start, end in atom_size:
        feats_classes_pca[label] = feats_pca[start:end, :]
        start = end

    return feats_classes, feats_classes_pca,pca,scaler



def classify_configurations(feats_classes, feats_classes_pca, traj_dir, label_list, classify_type, pca_transform,scaler_transform, test_liquid,before_denoiser,pca,scaler):
    pred_class_names, label_class_names,feats_all_struct = [], [],[]
    sim_score_list=[]
    tot_atom, num_mislabeled = 0, 0
    feats_all_struct=np.empty((0,feats_classes[next(iter(feats_classes))].shape[1]))
    for label in label_list:
        test_dump = glob.glob(f'{traj_dir}/ice_phase_{label}_*.extxyz')
        count = 0
        for traj in test_dump:
            config = f'{label}_{count}'
            if before_denoiser:
                atoms = ase.io.read(traj, ':')[0]
            else:
                atoms = ase.io.read(traj, ':')[-1]
            atoms = atoms[[atom.index for atom in atoms if atom.symbol != 'H']]
            tot_atom += len(atoms)
            feats_structure = soap_feats(atoms)
            feats_all_struct = np.concatenate([feats_all_struct, feats_structure], axis=0)
            for feats_atom in feats_structure:
                if scaler_transform:
                    feats_atom=scaler.transform(feats_atom.reshape(1,-1))
                if pca_transform:
                    class_name, sim_score = classify(feats_atom, feats_classes_pca, pca_transform, classify_type,pca=pca)
                else:
                    class_name, sim_score = classify(feats_atom, feats_classes, pca_transform, classify_type)
                pred_class_names.append(class_name)
                label_class_names.append(label)
                sim_score_list.append(sim_score)

                if class_name != label:
                    num_mislabeled += 1
                    print(f"predicted class name : {class_name}, actual class name: {label}")
            count += 1
    if test_liquid:
        #correct the classification for the liquid water phase based on the threshold score value 
        #of bulk and liquid phases determined from their distribution plots.
        bulkliquid_threshold=find_bulkliquid_threshold(classify_type)
        pred_class_names=classify_update(sim_score_list,pred_class_names,bulkliquid_threshold)
    print(f"Total number of atoms for testing: {tot_atom}, Total number of mislabeled atoms: {num_mislabeled}")
    return feats_all_struct, pred_class_names,label_class_names,sim_score_list



def save_results(feats_all_struct, pred_class_names,label_class_names,sim_score_list,out_dir,file_name):
    np.save(f'{out_dir}/feats_all_{file_name}.npy', feats_all_struct)
    np.save(f'{out_dir}/pred_all_{file_name}.npy', pred_class_names)
    np.save(f'{out_dir}/labels_all_{file_name}.npy', label_class_names)
    np.save(f'{out_dir}/sim_score_list_{file_name}.npy', sim_score_list)
