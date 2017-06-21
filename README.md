# Age and Gender Classification using Convolutional Neural Network
Implementation of paper [Age and Gender Classification using Convolutional Neural Network (June, 2015)](http://www.openu.ac.il/home/hassner/projects/cnn_agegender/CNN_AgeGenderEstimation.pdf)
using __caffe__.

## Requisites
__1. Caffe (Deep Learning Library)__  

__2. openCV (Computer Vision Library)__

## Directories
### 1. img  
Containing test images.  

### 2. model  
Containing __model files__(prototxt file), __weight files__(caffemodel file) and __mean file__(binaryproto file).  
You can also download those files and see details from [here](http://www.openu.ac.il/home/hassner/projects/cnn_agegender/).

### 3. src
Containing source files and header files. As you can notice intuitively, __GenderClassification.cpp__ and __GenderClassification.h__  
are for gender classification, __AgeClassification.cpp__ and __AgeClassification.h__ are for age classification and __Main.cpp__ has main.  

## How to run 
1. Build and make a exe file, 'AgeAndGenderClassification.exe'
2. Command is like below.  

```AgeAndGenderClassification.exe "GENDER_MODEL_FILE_PATH" "GENDER_WEIGHT_FILE_PATH" "AGE_MODEL_FILE_PATH" "AGE_WEIGHT_FILE_PATH" "MEAN_FILE_PATH" "TEST_IMG_PATH"```
