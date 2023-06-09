import torch

# Set device cuda for GPU if it's available, otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extra
SEED = 1

# Model
IN_CHANNELS = 3
OUT_CHANNELS = 64


# Training hyperparameters
INPUT_SIZE = 3 * 224 * 224
NUM_CLASSES = 2
LEARNING_RATE = 0.001  # (=1e-3)
BATCH_SIZE = 16
#NUM_EPOCHS = 10

# Dataset
BASE_DIR = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/Danish Challenge/2023 J#"
DATA_DIR = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/Danish Challenge/2023 J#/All"
ALL_CSV = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/Danish Challenge/2023 J#/xlbst.csv"
TRAIN_CSV = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/Danish Challenge/2023 J#/train_set.csv"
VAL_CSV = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/Danish Challenge/2023 J#/val_set.csv"
TEST_CSV = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/Danish Challenge/2023 J#/test_set.csv"
LABEL_COLUMN = "file_name"
TEST_SIZE = 0.2
NUM_WORKERS = 4

# Compute related
ACCELERATOR = "gpu"  # or "auto"
DEVICES = "auto"
MIN_EPOCHS = 1
MAX_EPOCHS = 5
DETERMINISTIC = True
