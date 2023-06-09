import torch

# Set device cuda for GPU if it's available, otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extra
SEED = 56
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
BASE_DIR = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/train_balanced"
DATA_DIR = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/train_balanced/all"

ALL_CSV = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/train_balanced/all.csv"
TRAIN_CSV = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/train_balanced/train_set.csv"
TRAIN_CSV_1 = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/train_balanced/train_set_1.csv"
TRAIN_CSV_2 = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/train_balanced/train_set_2.csv"
VAL_CSV = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/train_balanced/val_set.csv"
VAL_CSV_1 = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/train_balanced/val_set_1.csv"
VAL_CSV_2 = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/train_balanced/val_set_2.csv"
TEST_CSV = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/train_balanced/test_set.csv"

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
