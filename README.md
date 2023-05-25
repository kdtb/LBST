# Pioneering Automation in Agricultural Subsidy Processing through Deep Learning for Computer Vision

## Description

The README file serves as a valuable resource that provides detailed information about each script included in the repository. All code in this project is written in Python.

## Data Preparation (data_prep)

- **data_prep.ipynb**: The main script in this folder is responsible for dataset splitting into 80/20% train/test sets. It also organizes the data into appropriate folders on your local computer. Additionally, it verifies the reliability of the holdout validation protocol.

## Model Building (experimental_models & initial_models)

All the model building components are stored in the following folders:

### Standardization

- **calculateMeanStd.py**: This script calculates the mean and standard deviation of the dataset. These values are used for dataset standardization before model building.

### Configurations

- **config.py**: A script containing all the configurations for your project, such as the number of epochs, optimizer settings, root paths, etc.

### Creating .csv files

- **createCSV.py**: This script creates the necessary .csv files that contain information on image file names and target labels required for training the model.

### Loading photographs

- **customDataset.py**: A script responsible for reading and loading photograph files from the .csv files.

### Dataloaders

- **customDataModule.py**: This script creates the train/val/test dataloaders used in model training. It also handles dataset shuffling, data augmentation, and other related tasks.

### Model architecture

- **model.py**: A script that defines the architecture of the model.

### Plotting

- **plot_loss_and_acc.py**: This script provides visualization of the model's performance through learning curves and confusion matrices.

### Training

- **train.ipynb**: A JupyterLab notebook where you can train your model. This script loads all the above scripts and performs the model training.

## Requirements

- **requirements.txt**: A file containing all libraries needed to execute my code.

## License

Copyright © Kasper Dupont Toft Braun (kasperdtb@gmail.com), June
2023. This work is licensed under the Creative Commons attribution Non
Commercial No Derivations license (CC BY-NC-ND 2.0).
