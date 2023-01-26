# Analyzing and Improving the Pyramidal Predictive Network for Future Video Frame Prediction

![image](images/TotalNet.png) 

The pyramidal predictive network (PPNV1) proposes an interesting temporal pyramid architecture and yields promising results on the task of future video-frame prediction. We expose and analyze its signal dissemination and characteristic artifacts, and propose corresponding improvements in model architecture and training strategies to address them. Although the PPNV1 theoretically mimics the workings of human brain, its careless signal processing leads to aliasing in the network. We redesign the network architecture to solve the problems. In addition to improving the unreasonable information dissemination, the new architecture also aims to solve the aliasing in neural networks. Different inputs are no longer simply concatenated, and the downsampling and upsampling artifacts have also been redesigned to ensure that the network can more easily construct images from Fourier features of low-frequency inputs. Finally, we further improve the training strategies, to alleviate the problem of input inconsistency during training and testing. Overall, the improved model is more interpretable, stronger, and the quality of its predictions is better. 

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
* train.py, using the improved adversarial training method proposed in this paper for training
* test.py, for testing

### Others
* utils.py, construction of PyTorch dataset and calculation of loss, etc
