# da-faster-rcnn-PyTorch
An unofficial implementation of  'Domain Adaptive Faster R-CNN for Object Detection in the Wild ’


### Preparation

#### Requirements: Python=3.6 and Pytorch=0.4.0

1. Install [Pytorch](http://pytorch.org/)

2. Our code is conducted based on [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch),please setup the framework by it.

3. Download dataset
   
   - we use cityscape and cityscapes-foggy datasets respectly as source and target,the cityscapes dataset could be download [here](https://www.cityscapes-dataset.com/downloads/)  
   
   - the format of datasets is similar with VOC,you just need to split train.txt to train_s.txt and train_t.txt
   
   - you can also download the dataset here[]
   
  
