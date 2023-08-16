VAE Implementation in pytorch with visualizations
========

This repository implements a simpleVAE for training on CPU on the MNIST dataset and provides ability
to visualize the latent space, entire manifold as well as visualize how numbers interpolate between each other.

The purpose of this project is to get a better understanding of VAE by playing with the different parameters
and visualizations.

# Quickstart
* Create a new conda environment with python 3.8 then run below commands
* ```cd Pytorch-VAE```
* ``` pip install -r requirements.txt```
* ```python -m tools.train_vae.py```
* ```python -m tools.inference.py ```


## Data preparation
We don't use the torchvision mnist dataset to allow replacement with any other image dataset. 

For setting up the dataset:
* Download the csv files for mnist(https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
and save them under ```data```directory.
* Run ```python utils/extract_mnist_images.py``` 

Verify the data directory has the following structure:
```
Pytorch-VAE/data/train/images/{0/1/.../9}
	*.png
Pytorch-VAE/data/test/images/{0/1/.../9}
	*.png
```

## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of ```task_name``` key in config will be created and ```output_train_dir``` will be created inside it.

During training the following output will be saved 
* Best Model checkpoints in ```task_name``` directory
* PCA information in pickle file in ```task_name``` directory
* 2D Latent space plotting the images of test set for each epoch in ```task_name/output_train_dir``` directory

During inference the following output will be saved
* Reconstructions for sample of test set in ```task_name/output_train_dir/reconstruction.png``` 
* Decoder output for sample of points evenly spaced across the projection of latent space on 2D in ```task_name/output_train_dir/manifold.png```
* Interpolation between two randomly sampled points in ```task_name/output_train_dir/interp``` directory


## Sample Output
Latent Visualization

![latent_epoch_1](https://github.com/tusharkumar91/Pytorch-VAE/assets/462771/e0b3f001-26d8-42bb-8b4b-15606c90fc42)

Manifold

![manifold](https://github.com/tusharkumar91/Pytorch-VAE/assets/462771/99322dd6-3775-4d7a-9d98-c23ec922921b)

Reconstruction Images(in white, original in black)

![reconstruction](https://github.com/tusharkumar91/Pytorch-VAE/assets/462771/472433c0-aeab-4ace-aca0-753d9a5d8b70)
