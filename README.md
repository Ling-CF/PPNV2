We introduce here a predictive coding based model that aims to generate accurate and sharp future frames. Inspired by the predictive coding hypothesis and related works, the total model is updated through a combination of bottom-up and top-down information flows, which can enhance the interaction between different network levels. Most importantly, We propose and improve several artifacts to ensure that the neural networks generate clear and natural frames. Different inputs are no longer simply concatenated or added, they are calculated in a modulated manner to avoid being roughly fused. The downsampling and upsampling modules have been redesigned to ensure that the network can more easily construct images from Fourier features of low-frequency inputs. Additionally,  the training strategies are also explored and improved to generate believable results and alleviate inconsistency between the input predicted frames and ground truth. Our proposals achieve results that better balance pixel accuracy and visualization effect.

The arxiv paper is available [here](https://arxiv.org/abs/2301.05421)

![image](images/TotalNet.png) 

 

![image](images/additional_experiment.png)

### Dependencies
* PyTorch, version 1.12.1 or above
* opencv, version 4.6.0 or above
* numpy, version 1.23.4 or above
* skimage, version 0.19.3 or above
* lpips

Please download and process the relevant datasets first. In order to save time, we process the video sequence into data of size (T, C, H, W) in advance, where T represents the length of the sequence, and C, H, W represent the dimension, height and width of the image respectively. We have provided examples for pre-processing of each dataset in this project. In addition, it is recommended to create new folders named "models" and "metric" in the local project to save the training model and evaluation results. Or, you can save it to other paths, but you need to modify the save path specified in the program.

### Model Implementation
* FIRTorch.py, Using pytorch as a platform to implement a low-pass filter with learnable cutoff frequency
* NetworkBlock.py, implementation of the modulation module and upsampling, downsampling artifacts
* ConvLSTM.py, implementation of the convolutioinal LSTM unit
* PPNV2.py, implementation of the complete pyramidal predictive network

### Training and Testing
* train.py, for training
* test.py, for testing

### Others
* utils.py, construction of PyTorch dataset and calculation of loss, etc
