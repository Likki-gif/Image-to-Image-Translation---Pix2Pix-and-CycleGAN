# Image-to-Image-Translation---Pix2Pix-and-CycleGAN
pix2pix: Image-to-Image Translation with Conditional Adversarial Networks
This project implements a image-to-image translation method as described in the paper - Image-to-Image Translation with Conditional Adversarial Networks by Phillip Isola et al. (arXiv:1611.07004)

It was made as the final project for CS 763 - Computer Vision course in Spring 2019 at Indian Institute of Technology (IIT) Bombay, India.

Abstract
pix2pix uses a conditional generative adversarial network to efficiently design a general-purpose image-to- image translation system. Image-to-image translation involves learning a mapping from images from one distribution to corresponding images in another distribution. Many kinds of problems can be viewed as an image-to-image translation problem, including image colorization, edges to object visualization, style transfer etc.

For example, an output for Satellite-to-Maps view would be

1.png

Note
All the image output files in this project will be of the above format i.e.

[Source - Target_Ground_Truth - Target_Generated]

Datasets
I had tested this project with the following datasets released public by the authors (link in Acknowledgements section)

Facades
Maps (satellite-to-map)
Maps (map-to-satellite)
Getting Started
Follow the instructions below to get our project running on your local machine.

Clone the repository and make sure you have prerequisites below to run the code.
Run python src/main.py --help to see the various options available to specify.
To train the model, run the command python src/main.py ... along with the flags. For example, to run on the maps (map-to-satellite) dataset, you may run
python src/main.py --mode train --data_root '../datasets/maps' --num_epochs 100 --data_invert
All the outputs will be saved to src/output/[timestamp] where [timestamp] is the time of start of training.
Prerequisites
Python 3.7.1 or above

PyTorch 1.0.0 or above

CUDA 9.1 (or other version corresponding to PyTorch) to utilize any compatible GPU present for faster training

[The code is tested to be working with the above versions on a Windows 10 machine with GTX 1070. It may also work for other lower versions.]

Architecture
Code of the various modules can be found in the modules.py file.

Generator
I had used a U-Net (arXiv:1505.04597) like architecture for the generator, which is simply an encoder-decoder architecture with skip connections in between them.
U-Net

[Image Courtesy: Author's paper]

Precisely, the encoder channels vary as in_channels -> 64 -> 128 -> 256 -> 512 -> 512 -> 512 -> 512 and the decoder's channel sizes vary accordingly.

Discriminator

For the discriminator, a PatchGAN is used. A PatchGAN is similar to a common discriminator, except that it tries to classify each patch of N × N size whether it is real or fake.
In our case, we take N = 70​. This is in our code achieved by using a Convolutional network whose receptive field is 70 on the input image to the discriminator. Mathematically, this can be checked to be equivalent to what has been described in the paper.
The channel sizes in our PatchGAN  vary as in_channels -> 64 -> 128 -> 256 -> 512 -> out_channels.
Hyperparameters

I had used the default parameters mentioned in the code of main.py. You may easily test on other values by suitably changing the flags.
Results
All the results shown here are on test data.

Map-to-Satellite
1.png	2.png
3.png	4.png
5.png	6.png
Satellite-to-Map
1.png	2.png
3.png	4.png
5.png	6.png
Facades
1.png	2.png
3.png	4.png
5.png	6.png
As a sanity check, I would like to point out that on the training set, the model was able to give good outputs as shown below, indicating that it's capacity was quite sufficient.

train_1.png	train_2.png
Plots
For the Facades dataset,

Generator Loss [Training]	Discriminator Loss [Training]
g_loss.png	d_loss.png
Authors
Vamsi Krishna Reddy Satti - vamsi3
Acknowledgements
I would like to thank the authors of the paper for the amazing public dataset found here.
License
This project is licensed under MIT License - please see the LICENSE file for details.
