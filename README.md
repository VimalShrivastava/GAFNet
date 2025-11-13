# GAFNet
Global-Attention Fusion Network for Land Cover Segmentation

This repository contains the implementation of a custom DeepLabV3+ model integrating ResNet50, VGG19, Dense Prediction Cell (DPC), Attention Blocks, and Squeeze-and-Excitation (SE) modules for multiclass semantic segmentation of high-resolution remote sensing imagery.

**Key Features**
Combines ResNet50 and VGG19 backbones for multi-scale feature extraction
Employs Dense Prediction Cell (DPC) module for rich contextual learning
Integrates Attention Mechanisms and Squeeze-and-Excite (SE) modules to enhance spatial-channel awareness
Designed for 5-class land cover segmentation using 512×512 satellite images

**Requirements**
Install dependencies before running the model:
pip install tensorflow keras scikit-learn numpy pandas matplotlib opencv-python tifffile pillow

**Usage**
Clone the repository
git clone https://github.com/VimalShrivastava/GAFNet.git

**Run the model script**
python main.py
Modify the input_shape and num_classes as per your dataset.

**Model Overview**
Input: 512×512×3 (RGB Image)
Architecture: ResNet50 + VGG19 encoder, DPC, Attention, SE modules
Output: Softmax layer for multiclass prediction

**Example Application**
Used for land cover classification from multispectral remote sensing data.


Would you like me to tailor this README to explicitly match your ISRO RESPOND project (LandCover.ISRO / MARS-UNet framework) so it fits your other repositories too?
