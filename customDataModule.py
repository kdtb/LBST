import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl


from customDataset import LBSTDataset


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, train_csv, val_csv, test_csv, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.num_workers = num_workers

    #    def prepare_data(self): # downloading the data here so we have it to disc
    #        LBSTDataset(train_csv_file, data_dir, transform=None)
    #        LBSTDataset(val_csv_file, data_dir, transform=None)
    #        LBSTDataset(test_csv_file, data_dir, transform=None)
    # single gpu

    def setup(self, stage):
        # multiple gpu
        self.train_set = LBSTDataset(
            csv_file=self.train_csv,
            root_dir=self.data_dir,
            transform=transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    #transforms.RandomResizedCrop(224),
                    transforms.ToTensor()  # ,
                    # transforms.Normalize()
                ]
            ),
        )
        self.val_set = LBSTDataset(
            csv_file=self.val_csv,
            root_dir=self.data_dir,
            transform=transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomResizedCrop(224),
                    transforms.ToTensor()  # ,
                    # transforms.Normalize()
                ]
            ),
        )
        self.test_set = LBSTDataset(
            csv_file=self.test_csv,
            root_dir=self.data_dir,
            transform=transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomResizedCrop(224),
                    transforms.ToTensor()  # ,
                    # transforms.Normalize()
                ]
            ),
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
