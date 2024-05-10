import argparse
from sklearn.decomposition import PCA
from classify_utils import load_initial_structures, classify_configurations, save_results
from sklearn.preprocessing import StandardScaler

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Classify ice configurations.')
    parser.add_argument('--data_dir', type=str, default='./data/ice-unitcell', help='Path to the directory containing CIF files.')
    parser.add_argument('--traj_dir', type=str, default='./data/saved_denoised_trajs', help='Path to the directory containing trajectory files.')
    parser.add_argument('--classify_type', type=str, default='dist', choices=['cos', 'dist'], help='Type of similarity measure to use.')
    parser.add_argument('--pca_transform', type=bool, default=True, help='Apply PCA transformation on features.')
    parser.add_argument('--scaler_transform', type=bool, default=True, help='Apply standardscaler transformation on features.')
    parser.add_argument('--before_denoiser', action='store_true', help='Use the first frame of the trajectory (before denoising).')
    parser.add_argument('--test_interface', action='store_true', help='Classification for the liquid-bulk interface.')
    parser.add_argument('--test_water_bulk', action='store_true', help='Classification for the all bulk and water liquid phases.')
    parser.add_argument('--out_dir', type=str, default='./results_classify', help='Folder name of output files.')
    parser.add_argument('--file_name', type=str, default='after_denoise', help='Output file name prefix.')
    return parser.parse_args()

def main():
    args = parse_arguments()

    #check if any liquid phase was tested
    test_liquid=False
    if args.test_interface:
        label_list=['interface_1c','interface_1h','interface_sI']
        test_liquid=True
    elif args.test_water_bulk:
        label_list = ['2', '1c', '1h', '3', '6', '7', 'sI','w']
        test_liquid=True
    else: label_list = ['2', '1c', '1h', '3', '6', '7', 'sI']

    pca = PCA(n_components=30)
    scaler=StandardScaler()
    # Load data: ideal bulk structures
    ideal_struct_list= ['2', '1c', '1h', '3', '6', '7', 'sI']

    feats_classes, feats_classes_pca,pca,scaler = load_initial_structures(args.data_dir, ideal_struct_list,pca,args.scaler_transform,scaler)

    
    
    # Classify each atom in configurations of MD trajetories
    feats_all_struct, pred_class_names, label_class_names, sim_score_list = classify_configurations(
        feats_classes, feats_classes_pca, args.traj_dir, label_list,
        args.classify_type, args.pca_transform,args.scaler_transform, test_liquid,args.before_denoiser,pca,scaler
    )

    # Save results
    save_results(feats_all_struct, pred_class_names,label_class_names,sim_score_list,args.out_dir,args.file_name)

if __name__ == '__main__':
    main()