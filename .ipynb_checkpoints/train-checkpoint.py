import os
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from lightning.pytorch import seed_everything
import random

import config

# Create .csv file


def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


## Find folder paths

base_path = r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/Danish Challenge/2023 J#/"
target_dirs = os.listdir(base_path)
print(target_dirs)

## Create 2 separate df's: one for Approved images, one for NonApproved, containing file_name and label

### Assign label = 0 to Approved images
approved = pd.DataFrame(
    data=os.listdir(os.path.join(base_path, target_dirs[1])), columns=["file_name"]
)
approved = approved.assign(label=0)

### Assign label = 1 to NonApproved images
nonapproved = pd.DataFrame(
    data=os.listdir(os.path.join(base_path, target_dirs[2])), columns=["file_name"]
)
nonapproved = nonapproved.assign(label=1)

## Merge into 1 df
df = pd.concat([approved, nonapproved])

## Add parcel_id column containing character 3-10 from file_name column
df["parcel_id"] = df["file_name"].str[3:10]

## Write .csv
df.to_csv(
    r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/Danish Challenge/2023 J#/xlbst.csv",
    sep=",",
    encoding="utf-8",
    index=False,
)


## Split group by

set_all_seeds(config.SEED)

from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(test_size=0.20, n_splits=2, random_state=config.SEED)
split = splitter.split(df, groups=df.parcel_id)
train_inds, test_inds = next(split)

train_set = df.iloc[train_inds]
test_set = df.iloc[test_inds]


print(test_set.to_string())
print("Test set length:", len(test_set), "Train set length:", len(train_set), sep="\n")


## Split train into 80/20 train/val

train_set2 = train_set
splitter2 = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=config.SEED)
split2 = splitter2.split(train_set2, groups=train_set2.parcel_id)
train_inds2, val_inds = next(split2)

train_set2 = train_set.iloc[train_inds2]
val_set = train_set.iloc[val_inds]


## Save to csv

train_set2.to_csv(
    r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/Danish Challenge/2023 J#/train_set.csv",
    sep=",",
    encoding="utf-8",
    index=False,
)
val_set.to_csv(
    r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/Danish Challenge/2023 J#/val_set.csv",
    sep=",",
    encoding="utf-8",
    index=False,
)
test_set.to_csv(
    r"C:/Users/kaspe/OneDrive - Aarhus Universitet/Skrivebord/BI/4. semester/Data/LBST/Danish Challenge/2023 J#/test_set.csv",
    sep=",",
    encoding="utf-8",
    index=False,
)


# Train model

from model import NN
from dataset import CustomDataModule
import config

seed_everything(
    42, workers=True
)  # By setting workers=True in seed_everything(), Lightning derives unique seeds across all dataloader workers and processes for torch, numpy and stdlib random number generators. When turned on, it ensures that e.g. data augmentations are not repeated across workers.

if __name__ == "__main__":
    model = NN(
        input_size=config.INPUT_SIZE,
        num_classes=config.NUM_CLASSES,
        learning_rate=config.LEARNING_RATE,
    )  # .to(device)
    dm = CustomDataModule(
        data_dir=config.DATA_DIR,
        train_csv=config.TRAIN_CSV,
        val_csv=config.VAL_CSV,
        test_csv=config.VAL_CSV,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    trainer = pl.Trainer(
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=config.MIN_EPOCHS,
        max_epochs=config.MAX_EPOCHS,
        deterministic=config.DETERMINISTIC,
    )  # deterministic ensures random seed reproducibility
    trainer.fit(model, dm)  # it will automatically know which dataloader to use
    trainer.validate(model, dm)
    trainer.test(model, dm)

# A general place to start is to set num_workers equal to the number of CPU cores on that machine. You can get the number of CPU cores in python using os.cpu_count(), but note that depending on your batch size, you may overflow RAM memory.
