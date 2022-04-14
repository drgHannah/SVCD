# Spatially Varying Color Distributions for  Interactive Multi-Label Segmentation

An unofficial **Python** implementation of the paper [Spatially Varying Color Distributions for  Interactive Multi-Label Segmentation](https://vision.in.tum.de/_media/spezial/bib/nieuwenhuis-cremers-pami12.pdf) by Claudia Nieuwenhuis and Daniel Cremers.

<img src="https://github.com/drgHannah/SVCD/blob/main/img/kea.jpg" width = 250px> <img src="https://github.com/drgHannah/SVCD/blob/main/img/kea_scribble.png" width = 250px> <img src="https://github.com/drgHannah/SVCD/blob/main/img/kea_seg.png" width = 250px>

## Get Started
- **Install**  
Please install this repository by:

	    pip install ./SVCD/

- **Data**
The required data consists of an RGB image and a suitable image with user-drawn scribbles. In the image with scribbles, black values and pixels with alpha<1 are assumed to be unknown, besides each individual color is treated as a class.

## Usage
Please also have a look into *samples*.
```
from svcd.svcd import SVCDutils, SVCDSegmentation

image_path = ...
scribble_path = ...
save_path = ...

# load image and scribbles
image = SVCDutils.load_image(image_path)
scribble, colors = SVCDutils.load_segmentation_from_image(scribble_path)

# run algorithm
svcd = SVCDSegmentation(max_iter=1000)
segmentation, energies = svcd(image, scribble)  

# save segmentation
SVCDutils.save_segmentation_as_image(save_path, segmentation, colors)
```
