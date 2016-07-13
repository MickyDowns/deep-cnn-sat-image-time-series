# Deep CNN and RNN for Satellite Image Time Series

## Overview

This project is **IN PROCESS**. It uses a hybrid deep neural network architecture to sequence satellite images. The images, taken over different days in Southern California, were jumbled. The computer vision challenge is to order images chronologically based only on characteristics of the image itself i.e., no meta data on days, flights, angles, etc. The images were taken from this "Draper Satellite Image Chronology" competition at Kaggle: https://www.kaggle.com/c/draper-satellite-image-chronology). 

To date, the code:

    *   Explores the image repository using tools from the OpenCV (computer vision) toolkit including redband subtraction, brightness normalization, Sobol edge detection and eve principle components analysis. 

    *   Stitches like images together using kaze, akaze and brisk feature marking algorithms. It calculates required perspective tranformations and warps images as required to create valid image pairs. The objective is to:
        
        - Capture overlap between images to be used by the convolutional neural network (CNN) to sequence images

        - Capture the transformation homography to be used by the reucrrent neural network (RNN) to sequence images

The next steps:

    *   Built the CNN.

    *   Build the RNN experimenting w/ LSTM if appropriate

    *   Build the logistic function to incorporate probability-based predictions i.e., if picture a < (before) b, and picture b < c, then picture a < c. 

    *   Build the logistic or neural function that incorporates the three lower-level algorithms i.e., takes as inputs the final hidden layer of the CNN, the final hidden layer of the RNN and the logistic prediction. 

## Technology

The project technology is Python2.7, Jupyter notebook, Keras and Theano. I'll include installation instructions when complete.

### Key files:

* sat_img_seq.ipynb: This notebook has everything you need to train and run the networks. It loads and tranforms images then builds, feeds and optimizes the networks. 
## Credits

    *   Data exploration steps borrow heavily from a script built by **Ben Kamphaus** for the Kaggle competition. It is an excellent introduction to the basics of image exploration: https://www.kaggle.com/bkamphaus/draper-satellite-image-chronology/exploratory-image-analysis

    *   Image stitching steps borrow heavily from scripts prepared by **the1owl** and **HawkWang**. They used similar means and tools (OpenCV in particular) to register and stitch images. Their scripts are here: https://www.kaggle.com/the1owl/draper-satellite-image-chronology/stitch-and-predict, and here: https://www.kaggle.com/yourwanghao/draper-satellite-image-chronology/align-images, respectively. 

    *   CNN architecture is based on "Learning to Compare Image Patches via Convolutional Neural Networks" by **Zagoruyko et al**: http://arxiv.org/abs/1504.03641

    *   RNN architecture is based on "Long-Term Recurrent Convolutional Networks for Visual Recogniation and Description" by **Donahue et al** https://arxiv.org/abs/1411.4389


## Related Concepts

**Layered learning** is used when mapping directly from inputs to outputs is not tractable e.g., too complex for current algorithms, networks, hardware. It starts with bottoms-up, hierarchical task decomposition. It then uses machine learning algorithms to exploit data at each level training function-specific models. The output of learning in one layer feeds into the next layer building complex behaviors. More here: http://www.cs.cmu.edu/~mmv/papers/00ecml-llearning.pdf.

**CNN** tbd.

**RNN** tbd.

**LSTM (long-short term memory)** is RNN with memory. Each neural network layer can be thought of as a composite of four types of neurons with specialized memory, input, output and forget functions. The net effect is a network that can learn not just from the input array, but a sequence of input arrays arranged in batches over time. Chris Olah provides a great overview here: http://colah.github.io/posts/2015-08-Understanding-LSTMs/. 

## Learnings