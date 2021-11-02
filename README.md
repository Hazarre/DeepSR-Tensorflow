## DeepSR 
This repository serves as a quickstart to deep super-resolution model implementation and evaluation. The models are implemented in accordance to the actual paper, but the training dataset and methods will deviate from them to streamline the framework. This repository uses widely use numpy and PIL package to create a dataset pipeline from div2k that can be used for different ML frameworks. 


## Example Using this repo
Run the following command to setup an Anaconda environtment: 
```
conda env create -f environment.yml
```
Download datasets 

Run dataset.py to generate ```.tfrecord``` format datasets. They will be stored inside the ```tfrecords/folder```. 


### Testing and Benchmarking
In each ```{MODEL_NAME}/```, implement ```inference.py``` to be called for standard testing in ```test.py``` and benchmarking ```benchmark.py```. The interface should take two arguments,  the low resolution images

### Standards
Dataset: 
- Training/Validation is DIV2k. 
- Testing uses Set5, Set14, BSDS100.

Evaluation metrics are performed on unit8 PIL images y-channel using the following skimage methods. 
```python
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2ycbcr, ycbcr2rgb
```

### Dateset 
Stored under the ```data/``` folder. 
Run ```dataset.py```


### Training 

### Inference / Transfer Learning




## Repo Organization
Directory Structure: 
```data/``` is for storing local image datasets, ```.png``` preferred.
```{MODEL_NAME}/``` contains each implemented model and their training.
```dataset.py``` is the pipeline for generating images.
```common.py``` contains global methods, parameters and constants. 
```benchmark.py``` 
Optional: 
```tfrecords``` will contains ```.tfrecords``` for faster data reading in tensorflow 
```output``` is the directory for inferenced image outcomes from various models for comparison

Design Philosophy: 
- Keras clean interface for ML with Tensorboard monitoring for easy debug.
- Standard Dataset, Dataset Pre/Post Processing, Evaluation Metric.
- OOP style model implementation.

## Model Implementations 
#### FSRCNN
From paper (Accelerating the Super-Resolution Convolutional Neural Network)[http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html]






### Reference: 
[1] srgan: https://github.com/krasserm/super-resolution
[2] esrgan: https://hub.tensorflow.google.cn/captain-pool/esrgan-tf2/1
[3] keras: https://keras.io/examples/vision/super_resolution_sub_pixel
[4] FSRCNN: http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html
