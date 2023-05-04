import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from customDataset import LBSTDataset


class MeanStdModule(pl.LightningDataModule):
    def __init__(self, data_dir, train_csv, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.train_csv = train_csv
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        # multiple gpu
        self.train_set = LBSTDataset(
            csv_file=self.train_csv,
            root_dir=self.data_dir,
            transform=transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ]
            ),
        )
        

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )