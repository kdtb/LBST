import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class LBSTDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(
            self.root_dir, self.annotations.iloc[index, 0]
        )  # 0 indicates 1st column of the csv file. The first column in my .csv files are the file path
        
        image = Image.open(img_path)
        
        y_label = self.annotations.iloc[index, 1]#torch.tensor(
            #self.annotations.iloc[index, 1], dtype = float
        #)  # y_label is placed in the second column of the csv file


        if self.transform:
            image = self.transform(
                image
            )

        return (image, y_label)


# This class loads 1 image and the corresponding class to that image.
# https://www.youtube.com/watch?v=ZoZHd0Zm3RY&ab_channel=AladdinPersson