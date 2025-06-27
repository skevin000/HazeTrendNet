### Installation
The project is built with PyTorch 3.8, PyTorch 1.8.1. CUDA 10.2, cuDNN 7.6.5
For installing, follow these instructions:
~~~
conda install pytorch=1.8.1 torchvision=0.9.1 -c pytorch
pip install tensorboard einops scikit-image pytorch_msssim opencv-python
conda install pillow
~~~
**Please use the *pillow* package downloaded by Conda instead of pip.**


Install warmup scheduler:
~~~
cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..
~~~

For dehazing.
Computational complexity: 50.47 GFLOPs
total parameters: 5.45M

### Train on RESIDE-Indoor/Dense-Haze

~~~
cd ITS
python main.py --mode train --data_dir your_path/reside-indoor
~~~


### Train on RESIDE-Outdoor/NH-HAZE
~~~
cd OTS
python main.py --mode train --data_dir your_path/reside-outdoor
~~~


### Evaluation
The pre-trained models are located in the files.

#### Testing on SOTS-Indoor/Dense-Haze
~~~
cd ITS
python main.py --data_dir your_path/reside-indoor --test_model path_to_its_model
~~~
#### Testing on SOTS-Outdoor/NH-HAZE
~~~
cd OTS
python main.py --data_dir your_path/reside-outdoor --test_model path_to_ots_model
~~~

For training and testing, your directory structure should look like this

`Your path` <br/>
`├──reside-indoor` <br/>
     `├──train`  <br/>
          `├──gt`  <br/>
          `└──hazy`  
     `└──test`  <br/>
          `├──gt`  <br/>
          `└──hazy`  
`└──reside-outdoor` <br/>
     `├──train`  <br/>
          `├──gt`  <br/>
          `└──hazy`  
     `└──test`  <br/>
          `├──gt`  <br/>
          `└──hazy` 
