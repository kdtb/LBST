A brief summary:

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
