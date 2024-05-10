from yacs.config import CfgNode

_C = CfgNode()

# Cutoff distance for graph construction
_C.CUTOFF = 5.0

# Training configurations
_C.TRAIN = CfgNode()
_C.TRAIN.PIN_MEMORY = True
_C.TRAIN.NUM_WORKERS = 4
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.LEARN_RATE = 2e-4
_C.TRAIN.NUM_UPDATES = 300_000
_C.TRAIN.SIGMA_MAX = 0.15

# Model configurations
_C.MODEL = CfgNode()
_C.MODEL.NUM_SPECIES = 2
_C.MODEL.IRREPS_HIDDEN='8x0e +8x1o + 4x2e'
_C.MODEL.NUM_NEIGH=16
_C.MODEL.CKT_FILE='./saved_models/ice-denoiser-ckt.pt'
_C.MODEL.SAVE_FILE='./saved_models/ice-denoiser.pt'

# Data configurations
_C.DATA = CfgNode()
_C.DATA.IDEAL_STRUCTURES = './data/ice-unitcell/*.cif'
#_C.DATA.TEST_STRUCTURES = './traj_liquid/h2o.nph.ice-{label}.lammpstrj'
_C.DATA.TEST_STRUCTURES = './data/ice-traj/{label}/traj*'
_C.DATA.OUTPUT_FILE = './data/denoised/ice_phase_{label}_{count}_denoised.extxyz'

# Inference configurations
_C.INFERENCE = CfgNode()
_C.INFERENCE.SCALE = 1
_C.INFERENCE.STEPS = 20
_C.INFERENCE.LABEL_LIST = ['2', '1c', '1h', '3', '6', '7', 'sI', 'w']

def get_cfg_defaults():
    return _C.clone()