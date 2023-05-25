# Project Name

## Description

Provide a brief description of your project.

## Data Preparation

- **data_prep.ipynb**: The main script in this folder is responsible for dataset splitting into 80/20% train/test sets. It also organizes the data into appropriate folders on your local computer. Additionally, it verifies the reliability of the holdout validation protocol.

## Model Building

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

## Usage

Explain how to use your project and provide any necessary instructions.

## Credits

Mention any credits or acknowledgments you would like to give.

## License

Specify the license under which your project is released.

All code in this project is written in Python.

Feel free to customize this template according to your requirements.


	- data_prep folder : the main script in this folder is data_prep.ipynb. I split my dataset into 80/20% train/test, i place my data into proper folders on my local computer, i verify the reliability of the holdout validation protocol.
	- All other folders : all my model building consists of the following scripts:
		- calculateMeanStd.py : a script calculating the dataset mean and standard deviation, which is used to standardize the dataset before model building.
		- config.py : A script containing all my configs, such as epochs, optimizer, root paths etc.
		- createCSV.py : A script creating the necessary .csv files with information on image file name and target label needed to train the model
		- customDataset.py : A script reading and loading the photograph files from .csv files
		- customDataModule.py : A script that creates the train/val/test dataloaders used in model training. Also shuffles the dataset, enables data augmentation etc.
		- model.py : A script for defining the model architecture
		- plot_loss_and_acc.py : A script for visualizing model performance through learning curves and confusion matrices.
		- train.ipynb : A JupyterLab notebook where i train my model. This script is used to load all the above scripts.
