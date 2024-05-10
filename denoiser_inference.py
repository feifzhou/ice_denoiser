import argparse
import ase.io
import torch
from update_src.transforms.radius_graph_ase import RadiusGraph_ase
from tqdm.notebook import trange
from denoiser_config import get_cfg_defaults
from denoiser_training import atoms2pygdata,InitialEmbedding
import glob

@torch.no_grad()
def denoise_snapshot(atoms, model,cutoff,scale=1, steps=8):
    pos2graph = RadiusGraph_ase(cutoff=cutoff)
    pos_traj = [atoms.positions]
    
    data = atoms2pygdata(atoms, ase=True)
    data.pos  *= scale
    data.cell *= scale
    
    for _ in trange(steps):
        data = pos2graph(data)
        disp = model(data)
        data.pos -= disp
        pos_traj.append(data.pos.clone().numpy() / scale)
    
    return pos_traj

def main():
    parser = argparse.ArgumentParser(description="Inference denoiser model")
    parser.add_argument("--cfg", default="", help="Path to the user-defined config file")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    cfg.freeze()

    # Load the trained model
    model = torch.load(cfg.MODEL.SAVE_FILE)
    model = model.to('cpu')

    label_list=cfg.INFERENCE.LABEL_LIST
    for label in label_list:

        test_dump = glob.glob(cfg.DATA.TEST_STRUCTURES.format(label=label))
        count = 0
        for path in test_dump:
          # uncomment if analyzing the last frame of trajetory file
          #  noisy_atoms = ase.io.read(path, ':')[-1]
            noisy_atoms = ase.io.read(path)
            pos_traj = denoise_snapshot(noisy_atoms, model,cutoff=cfg.CUTOFF,scale=cfg.INFERENCE.SCALE, steps=cfg.INFERENCE.STEPS)
            denoising_traj = [
                ase.Atoms(
                    symbols=noisy_atoms.get_chemical_symbols(),
                    positions=pos,
                    cell=noisy_atoms.cell,
                    pbc=True
                )
                for pos in pos_traj
            ]
            for atoms in denoising_traj:
                atoms.wrap()
            ase.io.write(cfg.DATA.OUTPUT_FILE.format(label=label,count=count), denoising_traj)
            count += 1

if __name__ == '__main__':
    main()