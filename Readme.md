# Graph Network Model Denoiser and Water Classification with SOAP Descriptors

This sub-package was authored by Hong Sun (sun36@llnl.gov). It is part of the [graphite](https://github.com/LLNL/graphite) package.

It contains code for two modules:

1. Module I: Training and applying a score-based denoiser model to denoise thermal perturbation in crystal structures.
2. Module II: Classifying ice configurations based on similarity measures using SOAP descriptors.

## Module I: Denoiser Model

### Requirements

- Python 3.x
- PyTorch
- PyTorch Geometric (PyG)
- YACS
- ASE
- NumPy
- Matplotlib
- tqdm

Install the required packages using pip:

```bash
pip install torch torch_geometric yacs ase numpy matplotlib tqdm
```

### Configuration

Hyperparameters and configurations for the model and training process are managed using the YACS configuration system. Default configurations are defined in `denoiser_config.py`. Override the defaults by creating a separate YAML file and passing it as an argument to the training and inference scripts.

Example user-defined configuration file (`user_configs/user_config.yaml`):

```yaml
CUTOFF: 4.0

TRAIN:
  BATCH_SIZE: 16
  LEARN_RATE: 1e-4

MODEL:
  NUM_SPECIES: 3
```

### Training

Train the graph network model by running `denoiser_training.py`. Optionally provide a user-defined configuration file using the `--cfg` argument.

```bash
python denoiser_training.py --cfg user_configs/user_config.yaml
```

Trained models are saved in the `./saved_models` directory.

### Inference

Apply the trained model for denoising by running `denoiser_inference.py`. Optionally provide a user-defined configuration file using the `--cfg` argument.

```bash
python denoiser_inference.py --cfg user_configs/user_config.yaml
```

Denoised structures are saved in the directory specified by `DATA.OUTPUT_FILE` in the configuration file.

## Module II: Water Classification with SOAP Descriptors

### Usage

Run the script using the following command:

```
python classify_main.py [--data_dir DATA_DIR] [--traj_dir TRAJ_DIR] [--classify_type {cos,dist}][--pca_transform {True,False}] [--scaler_transform {True,False}][--before_denoiser] [--test_interface][--test_water_bulk] [--out_dir OUT_DIR] [--file_name FILE_name]
```
For examples,

Testing for the "before-denoising" structures including seven ice bulk phases and liquid water using "Euclidiean distance"
```bash
python classify_main.py --classify_type 'dist' --before_denoiser --test_water_bulk --file_name 'before_denoiser_water_pca_dist' > log_dist_water_before
```

Testing for the "after-denoising" structures including three liquid-ice interface trajs using "Cosine distance"
```bash
python classify_main.py --classify_type 'cos' --test_interface --file_name 'cos_after_interface' > log_cos_after_interface 
```   

### Functionality

1. Load ideal bulk structures specified in `label_list` from `data_dir`.
2. Extract features from the loaded structures and optionally apply PCA transformation.
3. Classify each atom in the configurations of MD trajectories stored in `traj_dir` based on the specified similarity measure (`classify_type`).
4. Save classification results (features, predicted class names, label class names, and similarity scores) in `out_dir`.

### Output

The script generates the following output files:

- `feats_all_struct.npy`: NumPy array containing features for all structures.
- `pred_class_names.npy`: NumPy array containing predicted class names for each atom.
- `label_class_names.npy`: NumPy array containing label class names for each atom.
- `sim_score_list.npy`: NumPy array containing similarity scores for each atom.

## Overall File Structure

- `denoiser_config.py`: Default configurations for the model and training process.
- `denoiser_training.py`: Script for training the denoiser model.
- `denoiser_inference.py`: Script for applying the trained model to denoise thermal perturbation.
- `classify_main.py`,`classify_utils.py` : Script for ice classification and its utils files.
- `update_src/`: Utility functions and modules used for training the denoiser model. Copied from the score-based denoiser model
- `data/`: Directory containing ideal structure files for training and test structures for inference.
- `saved_models/`: Directory where trained denoiser models are saved.
- `user_configs/`:Directory where user defined configs yaml files of denoiser model are stored.
- `plots_notebooks/`:Directory containing the notebooks to reproduce the PCA and histogram distribution plots in the paper.


## Release
LLNL-CODE-836648

