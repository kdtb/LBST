import os
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import random
from sklearn.model_selection import GroupShuffleSplit

# Create .csv files

class TrainTest:
    def __init__(self, base_dir, all_csv, train_csv, test_csv, label_column, test_size, seed):
        super().__init__()
        self.base_dir = base_dir
        self.all_csv = all_csv
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.label_column = label_column
        self.test_size = test_size
        self.seed = seed
    
    # Set seed function    
    def set_all_seeds(self):
        os.environ["PL_GLOBAL_SEED"] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    # Split function    
    def df(self):
        
        # Set seed
        TrainTest.set_all_seeds(self)
        base_path = self.base_dir
        target_dirs = os.listdir(base_path)

        ## Make csv with all images

        ### Assign label = 0 to Approved images
        approved = pd.DataFrame(
            data=os.listdir(os.path.join(base_path, target_dirs[1])), # 1 represents the second folder in the base path
            columns=[self.label_column]
        )
        approved = approved.assign(label=0)

        ### Assign label = 1 to NonApproved images
        nonapproved = pd.DataFrame(
            data=os.listdir(os.path.join(base_path, target_dirs[2])),
            columns=[self.label_column]
        )
        nonapproved = nonapproved.assign(label=1)

        ## Merge into 1 df
        df = pd.concat([approved, nonapproved])

        ## Add parcel_id column containing character 0-10 from file_name column
        df["parcel_id"] = df[self.label_column].str[0:10]
        df = df[df.file_name != 'desktop.ini']
        ## Write .csv
        df.to_csv(
            self.all_csv,
            sep=",",
            encoding="utf-8",
            index=False,
            mode='w' # overwriting existing file
        )


        ## Split into train and test while taking care of parcel_ID

        splitter = GroupShuffleSplit(test_size=self.test_size, n_splits=1, random_state=self.seed)
        split = splitter.split(df, groups=df.parcel_id)
        train_inds, test_inds = next(split)

        train_set = df.iloc[train_inds]
        test_set = df.iloc[test_inds]


        # Print all content
        # print(test_set.to_string())
        print("Train set length:", len(train_set), "Test set length:", len(test_set), "Train set label distribution:", train_set['label'].value_counts(), "Test set label distribution:", test_set['label'].value_counts(), sep="\n")

        ## Save to csv

        train_set.to_csv(
            self.train_csv,
            sep=",",
            encoding="utf-8",
            index=False,
            mode='w'
        )
        test_set.to_csv(
            self.test_csv,
            sep=",",
            encoding="utf-8",
            index=False,
            mode='w'
        )


    
class TrainVal:
    def __init__(self, orig_train_csv, train_csv, val_csv, label_column, val_size, seed):
        super().__init__()
        self.orig_train_csv = orig_train_csv
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.label_column = label_column
        self.val_size = val_size
        self.seed = seed
    
    # Set seed function    
    def set_all_seeds(self):
        os.environ["PL_GLOBAL_SEED"] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    # Split function    
    def df(self):
        
        TrainVal.set_all_seeds(self)
        train_set = pd.read_csv(
            self.orig_train_csv,
            sep=",",
            encoding="utf-8"
        )
        print("Original train set length:", len(train_set), sep="\n")
        
        ## Split into train and test while taking care of parcel_ID
        
        
        train_set2 = train_set
        splitter2 = GroupShuffleSplit(test_size=self.val_size, n_splits=1, random_state=self.seed)
        split2 = splitter2.split(train_set2, groups=train_set2.parcel_id)
        train_inds2, val_inds = next(split2)

        train_set2 = train_set.iloc[train_inds2]
        val_set = train_set.iloc[val_inds]

        print("New train set length and first n rows:", len(train_set2), train_set2.head(), "Val set length and first n rows:", len(val_set), val_set.head(), "Train set label distribution:", train_set2['label'].value_counts(), "Val set label distribution:", val_set['label'].value_counts(), sep="\n")

        ## Save to csv

        train_set2.to_csv(
            self.train_csv,
            sep=",",
            encoding="utf-8",
            index=False,
            mode='w' # overwrite
        )
        val_set.to_csv(
            self.val_csv,
            sep=",",
            encoding="utf-8",
            index=False,
        )
        
class TrainValMini:
    def __init__(self, train_csv_1, train_csv_2, val_csv_2, label_column, val_size, seed):
        super().__init__()
        self.train_csv_1 = train_csv_1
        self.train_csv_2 = train_csv_2
        self.val_csv_2 = val_csv_2
        self.label_column = label_column
        self.val_size = val_size
        self.seed = seed
    
    # Set seed function    
    def set_all_seeds(self):
        os.environ["PL_GLOBAL_SEED"] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    # Split function    
    def df(self):
        
        TrainValMini.set_all_seeds(self)
        train_set = pd.read_csv(
            self.train_csv_1,
            sep=",",
            encoding="utf-8"
        )
        print("Original train set length:", len(train_set), sep="\n")
        
        train_set = train_set.sample(frac = 0.5)
        
        ## Split into train and test while taking care of parcel_ID
        
        
        train_set2 = train_set
        splitter2 = GroupShuffleSplit(test_size=self.val_size, n_splits=1, random_state=self.seed)
        split2 = splitter2.split(train_set2, groups=train_set2.parcel_id)
        train_inds2, val_inds = next(split2)

        train_set2 = train_set.iloc[train_inds2]
        val_set = train_set.iloc[val_inds]

        print("New train set length and first n rows:", len(train_set2), train_set2.head(), "Val set length and first n rows:", len(val_set), val_set.head(), "Train set label distribution:", train_set2['label'].value_counts(), "Val set label distribution:", val_set['label'].value_counts(), sep="\n")

        ## Save to csv

        train_set2.to_csv(
            self.train_csv_2,
            sep=",",
            encoding="utf-8",
            index=False,
            mode='w' # overwrite
        )
        val_set.to_csv(
            self.val_csv_2,
            sep=",",
            encoding="utf-8",
            index=False,
        )