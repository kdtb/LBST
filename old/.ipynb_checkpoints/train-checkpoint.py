from createCSV import createCSV
import config
csv = createCSV(
        base_dir = config.BASE_DIR,
        all_csv = config.ALL_CSV,
        train_csv = config.TRAIN_CSV,
        val_csv = config.VAL_CSV,
        test_csv = config.TEST_CSV,
        label_column = config.LABEL_COLUMN,
        test_size = config.TEST_SIZE,
        seed = config.SEED)
csv.set_all_seeds()
csv.df()

# Train model

from model import NN
from dataset import CustomDataModule
import config
from callbacks import MyPrintingCallback, EarlyStopping
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything

seed_everything(
    42, workers=True
)  # By setting workers=True in seed_everything(), Lightning derives unique seeds across all dataloader workers and processes for torch, numpy and stdlib random number generators. When turned on, it ensures that e.g. data augmentations are not repeated across workers.

if __name__ == "__main__":
    logger = TensorBoardLogger("tb_logs", name="my_model") # tb_logs is the folder, name is the name of the experiment/model
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
        logger=logger, # PyTorch lightning will automatically know what we are logging by looking at our model.py logs
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=config.MIN_EPOCHS,
        max_epochs=config.MAX_EPOCHS,
        deterministic=config.DETERMINISTIC,
        callbacks=[MyPrintingCallback(), EarlyStopping(monitor="val_loss")],
    )  # deterministic ensures random seed reproducibility
    trainer.fit(model, dm)  # it will automatically know which dataloader to use
    trainer.validate(model, dm)
    trainer.test(model, dm)

# A general place to start is to set num_workers equal to the number of CPU cores on that machine. You can get the number of CPU cores in python using os.cpu_count(), but note that depending on your batch size, you may overflow RAM memory.
