# SRNet

This repository contains a MATLAB script (`srnet.m`) for training a deep residual neural network (ResNet) on an image dataset. The code is designed to load grayscale images, build a custom ResNet architecture, and train it using the Adam optimizer.

## Features

* **Data Loading**: Utilizes `imageDatastore` to automatically load and label images from a specified directory.
* * **Data Splitting**: Automatically splits the dataset into training (90%) and validation (10%) sets.
* * **Custom ResNet Architecture**: Builds a ResNet with specific layer configurations using `resnetLayers`:
* Image Input Size: 100x100x1 (Grayscale)
*   * Number of Classes: 2
*   * Stack Depth: `[3 4 23 3]`
*   * Number of Filters: `[64 128 256 512]`
* * **Training Configuration**:
 *   * Optimizer: Adam
     *   * Initial Learning Rate: 0.001
         *   * Max Epochs: 500
             *   * Mini-Batch Size: 128
                 *   * Validation Frequency: Every 20 iterations
                     *   * Visualizes training progress automatically
                         *    ## Prerequisites
                      
                         *    * MATLAB (with Deep Learning Toolbox)
                          
                              * ## Usage
                          
                              * 1. Update the image dataset path in `srnet.m` to point to your local dataset directory:
                                2.    ```matlab
                                         imds = imageDatastore('path_to_your_data','IncludeSubfolders',true,'Labelsource','foldernames');
                                         ```
                                      2. Run the `srnet.m` script in MATLAB to begin the training process. The script will automatically output the trained network `net` and the training information `info`.
                                      3. 
