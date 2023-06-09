import torch

# Set device cuda for GPU if it's available, otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extra
SEED = 1
SEED_2 = 2

# Model
IN_CHANNELS = 3
OUT_CHANNELS = 64


# Training hyperparameters
INPUT_SHAPE = 3 * 224 * 224
NUM_CLASSES = 2
LEARNING_RATE = 0.001  # (=1e-3)
BATCH_SIZE = 32

# Dataset
BASE_DIR = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST"
DATA_DIR = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/all"

ALL_CSV = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/all.csv"
TRAIN_CSV = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/train_set.csv"
TRAIN_CSV_1 = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/train_set_1.csv"
TRAIN_CSV_2 = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/train_set_2.csv"

VAL_CSV_1 = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/val_set_1.csv"
VAL_CSV_2 = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/val_set_2.csv"
TEST_CSV = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/test_set.csv"

MEAN = [0.48643434047698975, 0.5090485215187073, 0.40648937225341797]
STD = [0.1766146421432495, 0.17801520228385925, 0.21651023626327515]

LABEL_COLUMN = "file_name"
TEST_SIZE = 0.2
VAL_SIZE = 0.2
NUM_WORKERS = 4

# Compute related
ACCELERATOR = "gpu"  # or "auto"
DEVICES = "auto"
MIN_EPOCHS = 1
MAX_EPOCHS = 50
DETERMINISTIC = True
