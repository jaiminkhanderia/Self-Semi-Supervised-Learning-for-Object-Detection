
# Final Competition of Deep Learning Spring 2022

## Self/Semi Supervised Learning for Object Detection
The repository contains the work carried out by **Team 9** whose team members are mentioned below:

- Arnab Arup Gupta (ag7654)

- Jaimin Dineshbhai Khanderia (jk7178)

- Jaya Mundra (jm8834)

## Approach
We have used the [VICReg: Variance-Invariance-Covariance Regularization For Self-Supervised Learning](https://arxiv.org/abs/2105.04906)[^1] model as the SSL model to train using the unlabeled train dataset. We then use the trained backbone Resnet-50 in the VICReg architecture and fit that as pretrained backbone in the [Faster R-CNN](https://arxiv.org/abs/1506.01497)[^2] architecture to carry out the training (finetune) for object detection class using the labeled train dataset. We use the labeled validation dataset during the finetune training to evaluate the training process of the Faster R-CNN model. The following section contains all the instructions needed to replicate the final evaluation result for our trained model.


## Instructions to run the training code
### Instructions to train the VICReg model
We followed the instructions mentioned in the official implementation[^3] of VICReg to carry out the single node local training and the following are the steps to run our VICReg training model:
1. Navigate to `vicreg` directory which contains the files needed to run the training of the VICReg model.
	```
	\> cd vicreg
	```
2. We have written and submitted a `.slurm` file (`multiple_vicreg.slurm`) which contains the sbatch commands to configure and select the partition on GCP as well the python command to run `main_vicreg.py` script which is the entry point to train the VICReg model.
	-  Please be sure to modify the account id, partition for GPUs and time to run the batch job.
	- The final command contains command line arguments such as experiment directory which will contain checkpoint of the model (of which you have to pass the full path of the directory), data directory, epochs, learning rate and the batch size. Please be sure to modify those as per your requirement. The following values (one line per command line argument) were used by the team to train the model (and are the same as present in the file):
		```
		--data-dir /unlabeled
		--exp-dir /home/jk7178/multiple_run_04_23_2022_17_30/
		--arch resnet50
		--epochs 230
		--batch-size 128
		--base-lr 0.15
		```

3. Please submit the batch job using the following command:
	```
	\> sbatch multiple_vicreg.slurm
	```

### Instructions to train the Faster R-CNN model
Next, we move to finetune the Faster R-CNN model through a script `finetune.py` written by us using the trained Resnet-50 backbone from the previous SSL training of VICReg model. 

1. Navigate to `code` directory which contains the files needed to run the training (finetuning) of the Faster R-CNN model.
	```
	\> cd code
	```
2. We have written and submitted a `.slurm` file (`finetune.slurm`) which contains the sbatch commands to configure and select the partition on GCP as well the python command to run `finetune.py` script which is the entry point to train the Faster R-CNN model.
	-  Please be sure to modify the account id, partition for GPUs and time to run the batch job.
	- The final command contains command line arguments such as checkpoint directory that contains the checkpoint of the backbone model (of which you have to pass the full path of the directory and make sure that is same as the path used in the VICReg training), epochs, learning rate and the batch size. Please be sure to modify those as per your requirement. The following values (one line per command line argument) were used by the team to train the model (and are the same as present in the file):
		```
		--exp-dir /home/jk7178/finetune_run_04_24_2022_20_00
		--epochs 20
		--batch-size 8
		--lr 0.005
		--wd 0.0005
		--momentum 0.9
		```

		Note: We finetuned in chunks of 20 epochs (total epochs = 70) and we changed the logic to use a constant learning rate (passed as command line argument) after 40 epochs when we were resuming from previous checkpoint instead of using the learning rate saved in the previous checkpoint. We then used StepLR as it is to make the model learn for a longer period of time instead of decaying learning rate which helped us achieve better results.
		
3. Please submit the batch job using the following command:
	```
	\> sbatch finetune.slurm
	```

## Instructions to run the evaluation code
The model saved from the finetuning of Faster R-CNN was used to evaluate the model on validation dataset and the model was uploaded to Google Drive for easier sharing.

1. Please download the model using the following Google Drive link. Place it in the repository: https://drive.google.com/file/d/1uHO8PIVZbrKXAGJiqXjLTzgURwo1aRaH/view?usp=sharing

2. The following is the directory structure:
```
code
|-- evaluate.py
|-- evaluate.slurm
|-- ...
vicreg
|-- main_vicreg.py
|-- ...
README.md
validation_results.txt
model.pth
```

3. Navigate to `code` dir which contains the files needed to run the evaluation of the model built by our team for object detection task.
	```
	\> cd code
	```  

4. We have written and submitted a `.slurm` file (`evaluate.slurm`) which contains the sbatch commands to configure and select the partition on GCP as well the python command to run our `evaluate.py` script.
	- Please be sure to modify the account id.
	- This contains arguments such as the checkpoint of the model, and the batch size, of which you have to pass the full path of the directory that contains the downloaded model weights (checkpoint). 

6. Please submit the batch job using the following command:
	```
	\> sbatch evaluate.slurm
	```  

6. The results of our `evaulate.py` script on the validation labeled dataset are in the file `validation_results.txt`.
  

## Citations

[^1]: [VICReg: Variance-Invariance-Covariance Regularization For Self-Supervised Learning](https://arxiv.org/abs/2105.04906) (https://arxiv.org/abs/2105.04906)
	```
	@misc{https://doi.org/10.48550/arxiv.2105.04906,
	doi = {10.48550/ARXIV.2105.04906},
	url = {https://arxiv.org/abs/2105.04906},
	author = {Bardes, Adrien and Ponce, Jean and LeCun, Yann},
	keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
	title = {VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning},
	publisher = {arXiv},
	year = {2021},
	copyright = {arXiv.org perpetual, non-exclusive license}
	}
	```
	
[^2]:  [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) (https://arxiv.org/abs/1506.01497) 
	``` 
	@misc{https://doi.org/10.48550/arxiv.1506.01497,
	doi = {10.48550/ARXIV.1506.01497},
	url = {https://arxiv.org/abs/1506.01497},
	author = {Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
	keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
	title = {Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks},
	publisher = {arXiv},
	year = {2015},
	copyright = {arXiv.org perpetual, non-exclusive license}
	}
	```
[^3]: [Official repository that provides a PyTorch implementation and pretrained models for VICReg](https://github.com/facebookresearch/vicreg) (https://github.com/facebookresearch/vicreg)
	```
	@inproceedings{bardes2022vicreg,
	author  = {Adrien Bardes and Jean Ponce and Yann LeCun},
	title   = {VICReg: Variance-Invariance-Covariance Regularization For Self-Supervised Learning},
	booktitle = {ICLR},
	year    = {2022},
	}
	```
