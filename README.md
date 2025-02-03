# Forest Images Segmentation
This project implements U-Net model to identify the forest area of an image. Adam is chosen as the optimizer and Tversky loss as the loss function. Intersection over Union (IoU) is used as an indicator for validation.


## Data sources
The dataset comes from Quadeer's [Forest Aerial Images for Segmentation](https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation) on Kaggle. The 5108 images and masks from the dataset are split into *4137 (training data)*, *460 (validation data)* and *511 (testing data)*.


## Methodology
Masks are converted into binary single-channel representations. Low-contrast images are enhanced using the CLAHE (Contrast Limited Adaptive Histogram Equalization) technique. Data augmentations, including rotation, cropping, erasing, Gaussian noise, and Gaussian blur, are applied during training.

The U-Net model consists of an encoder and a decoder. The encoder extracts hierarchical features using convolutional layers, progressively reducing spatial dimensions while increasing feature abstraction. The decoder then reconstructs segmentation masks to identify forest areas. Skip connections are incorporated to transfer high-resolution spatial features from the encoder to the decoder, helping to preserve fine details and improve segmentation accuracy.

Tversky loss is chosen as the loss function instead of binary cross-entropy (BCE) due to a significant class imbalance between forest and non-forest areas.


## Results
For comparison, a baseline model is constructed to randomly predict each pixel as 1 (forest) or 0 (not forest). The trained U-Net model has **average IoU of 0.71** which is much better than the baseline IoU 0.38.


## Modules used
* `numpy`: Manipulates images
* `pandas`: Constructs a datatable from `meta_data.csv`
* `cv2`: Reads images into 2-D arrays, resizes images, converts masks to binary gray scale
* `matplotlib`: Plots images and visualizes data with `pyplot`
* `sklearn`: Splits and shuffles data into training and testing
* `torch`: Constructs neural networks including U-Net and CNN blocks within it


## Programs  included
* `forest_images_segmentation.ipynb`:
    * `image_process.py`: Preprocesses images and masks, constructs torch Dataset which implements augmentations to training dataset.
    * `model_unet.py`: Constructs a U-Net from torch.
    * `train_evaluation.py`: Constructs the pipelines of training and evaluation process: loads training data, validates validation data, evaluates test data.
        * `EarlyStopper` is consturcted to prevent over training and save the model at its minimum validation loss.
        * `TverskyLoss` is defined as the loss function.
