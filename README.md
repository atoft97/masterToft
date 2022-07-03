# Semantic segmentation of LiDAR and RGB images for autonomous terrain driving.
This the the codebase for the master thesis: Semantic segmentation of LiDAR and RGB images for autonomous terrain driving.

## Overview
The project is mainely based on the mask2former model found here: https://github.com/facebookresearch/Mask2Former.
The model is adjusted to work better with LiDAR data and the modied model is found in `Segmentation/mask2Fromer/mask2former/`. 
Some new dataset mappers and dataloaders which work with LiDAR data i numpy format is found in `Segmentation/mask2Fromer/`. 

The preprocessing of the dataset is found in the `utils` folder. 
The `semanticSegmentationDriving` folder contains the scirpts for navigating the terrain vehcile with ros based on segmented images. 


