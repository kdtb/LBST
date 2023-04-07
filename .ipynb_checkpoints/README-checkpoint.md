A brief summary:

	- config.py : all my configs
	- customDataset.py : my custom dataset is created in this script. I have 65 samples in train set, 13 in val set, 15 in test set (its small because im testing the code)
	- customDataModule.py : contains a setup module where i call the customDataset class from customDataset.py. And then my dataloaders are also created in this script
	- model.py : The LightningModule is used in this script
	- train.ipynb : is the JupyterLab notebook where i train my model