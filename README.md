# AI-for-crop-recognition-with-Satellite-data_Computer-Vision-competition
This repository contains the implementation of a deep learning model that predicts the presence of rice crops at a given location in the An Giang province, Mekong Delta, Vietnam. The model is based on the ResNet-50 architecture and trained on optical data from the Sentinel-2 satellite. This project was developed as part of the EY/Microsoft 2023 Open Science Data Challenge, with the goal of contributing to solutions for world hunger and improving food security.

## Introduction
The study of rice crops using satellite data is a well-established practice, with researchers and government organizations routinely employing such data for identifying crop locations and forecasting yields.

This project aims to predict the presence or non-presence of rice crops at a given location. The developed model uses satellite images from the European Copernicus Sentinel-2 program and is fine-tuned with images of regions in the An Giang province of Vietnam.

## Dataset
The dataset is prepared by downloading Sentinel-2 satellite images for different locations and filtering them based on a sparsity threshold after applying Cloud filtering. The images are then saved as HDF5 files containing various bands and indices, such as NDVI and NDMI. The dataset is then split into training and validation sets using K-fold cross-validation. The included Jupyter notebook contains helper functions for downloading, processing, and cleaning the images.

The input images used for training and evaluation are processed to include three channels: NDVI, NDMI, and NIR. These channels were chosen based on their relevance to vegetation content, which are important factors in identifying rice fields.
* NIR (Near Infrared) band: This band captures the reflectance of the near-infrared wavelength.
* NDVI (Normalized Difference Vegetation Index): This index is calculated using the Red and Near-Infrared (NIR) bands of the satellite images. It is widely used to assess the presence and health of vegetation.Index calculation: NDVI = (NIR-Red) / (NIR+Red)
* NDMI (Normalized Difference Moisture Index): This index is calculated using the Near-Infrared (NIR) and Short-Wave Infrared (SWIR) bands of the satellite images. It is used to estimate the water content in vegetation. Higher NDMI values indicate higher moisture content, while lower values signify less moisture or no vegetation.Index calculation: NDMI = (NIR – SWIR) / (NIR + SWIR)

By incorporating these channels as inputs, the model can learn to detect rice fields more effectively, leveraging the unique spectral characteristics of rice fields in the satellite imagery.

The input images were also normalized to be compatible with the pre-processing steps used during the pre-training of the ResNet-50 model on the ImageNet dataset. This normalization step is crucial to ensure that the fine-tuning process effectively leverages the pre-trained weights for the rice crop detection task.

## Model
The model used in this project is based on the ResNet-50 architecture, which is a popular deep learning model known for its ability to achieve high accuracy in various image classification tasks. The ResNet-50 model is composed of 50 layers and utilizes residual connections to facilitate learning by enabling gradients to flow more easily through the network during backpropagation.

The original ResNet-50 model has been fine-tuned specifically for the task of rice crop detection. The output layer of the model was replaced with a new fully connected layer to perform binary classification(Rice and Non-Rice). All layers in the model, except for the last fully connected layer, were frozen during the training.

# Results
1st place of Spain's EY participants. 46th place overall(among more than 1000 participants).
0.95 Accuracy achieved in the final benchmark

Certificate (obtained for those who achieved more than 90% of Accuracy in the competition)
![imagen](https://github.com/SergioMG97/Crop-detection-with-Satellite-data--Computer-Vision-competition/assets/76975149/44ef0b7a-2cdd-44f5-8efc-4a95245ca242)

