# Diffusion Grain
Machine Learning Model for Grain Detection Based on DiffusionEdge 

![grainExample](/assets/fullImg_testFiber1_1.png)

## Overview

This repository contains code for training and sampling the diffusionEdge model, image augmentation and data manipulation scripts to assist with training, and a computer vision script to process the segmented output with manual correction and statistical analysis of the detected grains. 

## Grain Detection and Augmentation

The diffusionEdge model operates in two stages with separate weights. 

First stage (default BSDS weights): https://github.com/GuHuangAI/DiffusionEdge/releases/download/v1.1/first_stage_total_320.pt

Second stage (diffusionGrain weights): https://specialtymaterials.box.com/s/zagv8us45iq9szts9tac6hqxjahqowpx


See the original Github (https://github.com/GuHuangAI/DiffusionEdge/tree/main?tab=readme-ov-file) for detailed information about sampling with the model or training on your own data. 

A minimum of 400 images post-augmentation should be used for training, ideally more than 800. Data augmentation can be done with the provided augmentation script 'augmentation.py,' an easily adaptable framework with a wide array of procesing methods. Early training can be dramatically hastened by using a batch size of one until the gradients become unstable, at which point training can be resumed with a batch size of two. 

## Grain Processor

grainProcessor.py intakes the output of the diffusionGrain model, cleanes and skeletonizes the result, and presents the user a colored view of the identified grains. The user may click on extraneous segmentations to remove them from processing. Once complete ('d' on the keyboard twice), data for the width, heigh, area, and aspect ratio of each identified grain will be exported to a file. Default units are microns, defined by a scaling factor of nanometers per pixel. 
