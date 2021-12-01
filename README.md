## DeepSR 
This repository serves as a quickstart to deep super-resolution model implementation and evaluation. The models are implemented in accordance to the actual paper, but the training dataset is chosen to be DIV2k to streamline the framework. This repository uses popular numpy, PIL and keras package to create a dataset pipeline from div2k that can be used for tensorflow. The evaluation will be done using methods from skimage.metrics. 

### Standards
Dataset (can be found [through this link](https://github.com/jbhuang0604/SelfExSR)): 
- Training/Validation is [DIV2k](https://data.vision.ee.ethz.ch/cvl/DIV2K/). 
- Testing uses [Set5, Set14, BSDS100](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html).

Evaluation metrics are performed on unit8 PIL images y-channel using the following skimage methods. 
```python
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2ycbcr, ycbcr2rgb
```
Current models are implemented for mainly x4 and some x2 upscaling factors.


## Example Using this repo
### Environment Setup
Clone the repo and run the following command to setup an Anaconda environtment: 
```
conda env create -f environment.yml
```

### Benchmark & Testing
Implement and resuse methods in ```benchmark.py```. To add your own SR model, implement the upscaler function and see examples in ```benchmark.py```. 
```python
def modelName_upscaler():
```
Any ```upscaler()``` can be called by benchmarking as well as SR images and video generators for comparison. For example, FSRCNN and MCDN are both trained by authors of this repo. SRGAN and ESRGAN uses downloaded from other repos. See how their upscalers() are implemented. 

To generate SR videos, see enhance_video(), upscale_video(),restore_video(). A little experiment will show that a the trained models are biased towards upscaling bicubic downscaled frames. Perhaps 





### Dataset 
Download the datasets and store them under the ```data/``` folder. 
Run ```dataset.py``` to generate ```.tfrecord``` format datasets. They will be stored inside ```tfrecords/```. (Rename the folder for different upscaling factors to keep tfrecords for different datasets.)


### Model & Training 
In each ```{MODEL_NAME}/```, implement ```model.py```. Train the model and save the weights to ```weights```. This will allow model to be later loaded in ```benchmark.py```. 



## Repo Organization
Directory Structure:  
```data/``` is for storing local image datasets, ```.png``` preferred.  
```{MODEL_NAME}/``` contains each implemented model and their training.  
```dataset.py``` is the pipeline for generating images.  
```common.py``` contains global methods, parameters and constants.   
```benchmark.py``` provides methods for obtaining SR metrics (PSNR and SSIM) and generating SR images side by side.   

Optional:   
```tfrecords``` will contains ```.tfrecords``` for faster data reading in tensorflow .  
```output``` is the directory for inferenced image outcomes from various models for comparison.  
```utils``` has python scripts for upscaling videos and combining 4 videos into one scene for comparison. 

Design Philosophy: 
- Keras clean interface for ML with Tensorboard monitoring for easy debug.
- Standard Dataset, Dataset Pre/Post Processing, Evaluation Metric.
- OOP style model implementation see FSRCNN.
- Functional style model implementation see SRGAN. 

## Model Implementations 
### FSRCNN
From paper [Accelerating the Super-Resolution Convolutional Neural Network](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html).  

### ASRCNN
A modified ASRCNN with two x2 transposed convolutions instead of one x4 at the end of the network. 

### SRGAN 
Cloned and modified from [Krassem's github repo](https://github.com/krasserm/super-resolution)

### ESRGAN
From [tensorflow ESRGAN](https://www.tensorflow.org/hub/tutorials/image_enhancing).

## Reference: 
[1] srgan: https://github.com/krasserm/super-resolution  
[2] esrgan: https://hub.tensorflow.google.cn/captain-pool/esrgan-tf2/1  
[3] keras: https://keras.io/examples/vision/super_resolution_sub_pixel  
[4] FSRCNN: http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html

## Contributors:
Henry Chang and [Sam Yang](https://github.com/sam19029xc). 