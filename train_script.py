import argparse

from torch.utils.data import DataLoader
from cnd.ocr.dataset import OcrDataset
from cnd.ocr.model import CRNN
from cnd.config import OCR_EXPERIMENTS_DIR, CONFIG_PATH, Config
from cnd.ocr.transforms import ResizeToTensor, get_transforms
from cnd.ocr.metrics import WrapCTCLoss
from catalyst.dl import SupervisedRunner, CheckpointCallback
import string
from pathlib import Path
import torch
import os

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
parser.add_argument("-en", "--experiment_name", help="Save folder name", required=True)
parser.add_argument("-gpu_i", "--gpu_index", type=str, default="0", help="gpu index")
args = parser.parse_args()

# IF YOU USE GPU UNCOMMENT NEXT LINES:
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index

# define experiment path
EXPERIMENT_NAME = args.experiment_name
EXPERIMENT_DIR = OCR_EXPERIMENTS_DIR / EXPERIMENT_NAME

CV_CONFIG = Config(CONFIG_PATH)

DATASET_PATH = CV_CONFIG.data['data_path']

# CHANGE YOUR BATCH SIZE
BATCH_SIZE = 16
# 400 EPOCH SHOULD BE ENOUGH
NUM_EPOCHS = 100

alphabet = " "
alphabet += string.ascii_uppercase
alphabet += "".join([str(i) for i in range(10)])

MODEL_PARAMS = {
    "image_height" : 32, 
    "number_input_channels" : 3, 
    "number_class_symbols" : len(alphabet) + 1, 
    "rnn_size" : 64
}

if __name__ == "__main__":
    if EXPERIMENT_DIR.exists():
        print(f"Folder 'EXPERIMENT_DIR' already exists")
    else:
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    transforms = get_transforms(CV_CONFIG.data['ocr_image_size'])

    train_dataset = OcrDataset(DATASET_PATH + 'train/', DATASET_PATH + 'train.csv', transforms= transforms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=6,
    )
    
    val_dataset = OcrDataset(DATASET_PATH + 'val/', DATASET_PATH + 'val.csv',
                         transforms = ResizeToTensor(CV_CONFIG.data['ocr_image_size']))

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = CRNN(**MODEL_PARAMS)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    callbacks = [CheckpointCallback(save_n_best=10)]
    runner = SupervisedRunner(input_key="image", input_target_key="targets")

    runner.train(
    model=model,
    criterion=WrapCTCLoss(alphabet),
    optimizer=optimizer,
    scheduler=scheduler,
    loaders={'train': train_loader, "valid": val_loader},
    logdir="./logs/ocr",
    num_epochs=NUM_EPOCHS,
    verbose=True,
    callbacks=callbacks
    )
