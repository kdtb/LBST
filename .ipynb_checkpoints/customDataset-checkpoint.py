# https://www.youtube.com/watch?v=ZoZHd0Zm3RY&ab_channel=AladdinPersson

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io
from PIL import Image

class LBSTDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations) # total sample size (80 images)
    
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0]) # 0 indicates 1st column of the csv file. os.path.join() method in Python join one or more path components intelligently. This method concatenates various path components with exactly one directory separator (‘/’) following each non-empty part except the last path component. If the last path component to be joined is empty then a directory separator (‘/’) is put at the end. 
        image = Image.open(img_path)
#        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1])) # y_label is placed in the second column of the csv file
        
 #       parcel_id = np.array((self.annotations.iloc[index, 2]))
        #parcel_id = torch.tensor(int(self.annotations.iloc[index, 2]))
        
        if self.transform:
            image = self.transform(image) # optional: if we send in trasnforms, it will transform images
            
        return (image, y_label)
    
# This class loads 1 image and the corresponding class to that image.