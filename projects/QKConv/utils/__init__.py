from .smd_dataset import SMDDataset
from .qrecc_dataset import QReCCDataset
from .wow_dataset import KILTWoWDataset

DATASET_ZOO = {
    "smd": SMDDataset,
    "qrecc": QReCCDataset,
    "wow": KILTWoWDataset
}