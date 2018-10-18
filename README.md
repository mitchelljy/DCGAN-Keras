# DCGAN-Keras

## Introduction

This is a relatively simple Deep Convolutional Generative Adversarial Network built in Keras. Given a dataset of images it will be able 
to generate new images similar to those in the dataset. It was originally built to generate landscape paintings such 
as the ones shown below. As a result, also contained are some scripts for collecting artwork from ArtUK and resizing 
images to make them work with the network.

## Example Landscape Outputs

The following images were generated at 256x192, then upscaled using [the bigjpg tool](https://bigjpg.com/) which is a GAN based upscaling tool.

|Mountain Lake|Peninsula|
|:-----------:|:-----------:|
| <img src="https://raw.githubusercontent.com/DataSnaek/DCGAN-Keras/master/data/output/upscaled/Mountain_Lake.png"> | <img src="https://raw.githubusercontent.com/DataSnaek/DCGAN-Keras/master/data/output/upscaled/Peninsula.png"> |

|Hill Bushes|Grassy Mountain|
|:-----------:|:-----------:|
| <img src="https://raw.githubusercontent.com/DataSnaek/DCGAN-Keras/master/data/output/upscaled/Hill_Bushes.png"> | <img src="https://raw.githubusercontent.com/DataSnaek/DCGAN-Keras/master/data/output/upscaled/Grassy_Mountain.png"> |

Here is a selection of images generated at 128x128

|128x Selection|
|:-----------:|
|<img src="https://raw.githubusercontent.com/DataSnaek/DCGAN-Keras/master/data/output/grid/out-64.png" width="80%"> |

# Getting Started

This section talks about how to use this model, its prerequisites and its paramaters.

## Prerequisites
This model was built using the following packages and versions (earlier versions may still work):

### DCGAN.py
```
- Python 3.6
- tensorflow/tensorflow_gpu 1.11
- Keras 2.2.4
- Pillow 5.1.0
- numpy 1.14.5
- scipy 1.1.0
- Ideally GPU/CUDA support setup with tensorflow_gpu, otherwise training will take a very long time
```
### scrape_imgs.py
```
- Python 3.6
- requests 2.18.4
- bs4 0.0.1
```
### resize_imgs.py
```
- Pillow 5.1.0
```
## Parameters for DCGAN.py

List of paramaters for the DCGAN.py file:

* ```--load_generator```: Path to existing generator weights file
  * e.g. ```../data/models/generat.h5```
* ```--load_discriminator```: Path to existing discriminator weights file
  * e.g. ```../data/models/discrim.h5```
* ```--data```: Path to directory of images of correct dimensions, using *.[filetype] (e.g. *.png) to reference images
  * e.g. ```../data/resized/paintings_256x/*.png```
* ```--sample```: If given, will generate that many samples from existing model instead of training
  * e.g. ```20```
* ```--sample_thresholds```: The values between which a generated image must score from the discriminator
  * e.g. ```(0.0,0.1)```
* ```--batch_size```: Number of images to train on at once
  * e.g. ```24```
* ```--image_size```: Size of images as tuple (height,width). Height and width must both be divisible by (2^5)
  * e.g. ```(192,256)```
* ```--epochs```: Number of epochs to train for
  * e.g. ```500000```
* ```--save_interval```: How many epochs to go between saves/outputs
  * e.g. ```100```
* ```--output_directory```: Directoy to save weights and images to
  * e.g. ```../data/output```
  
 ## Example Usage
 
 ### DCGAN.py
 
 To train a fresh model on some data, the following command template is ideal:
 
 ```python DCGAN.py --data /data/images/*.png --epochs 100000 --output /data/output```
 
 Modifications can be made to image size, batch size etc. using the parameters. If your GPU doesn't have enough memory, you can change the size of the filters within the file, the image size and the batch size to better suit your GPU capability.
 
 ### scrape_imgs.py
 
 In it's current state it will download all of the images on [this ArtUK page]("https://artuk.org/discover/artworks/search/class_title:landscape--category:countryside/page/0"). You can modify the URL given in the file with any page or set of categories on ArtUK and it will download those instead.
 
 ### resize_imgs.py
 
 Will resize any directory of imgs to the specified size and store the new imgs in a different directory
 
 # Licensing
 
 This project is released under the MIT license, see LICENSE.md for more details.
