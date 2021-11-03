# DPANet
The implementation of "Learning Dual-Pixel Alignment for Defocus Deblurring".

# Prerequisites  
- The code has been test with following environment
  - Ubuntu 18.04
  - Python 3.7.9
  - PyTorch 1.7.0
  - cudatoolkit 10.0.130
  - NVIDIA TITAN RTX GPU

# Datasets
### Training datasets
  - [DPDD train set](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel)
### Testing datasets
  - [DPDD test set](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel)
  - [PIXEL test set](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel)

### Preparing (for DCNv2)
```shell
$ cd DPANet
$ python setup.py build develop
```

# Test
- Download our [pre-trained model](https://drive.google.com/drive/folders/1CcK7UzB1c4SnNwI9Rh9irTq2YeB8cvzK?usp=sharing) and put the `final.pth` into `./checkpoint` folder
```shell
$ cd DPANet
$ python test.py
```
For more results, you can refer to [More results on DPDD](https://drive.google.com/drive/folders/1hxeq0j8T6h80rR5bGV-JUysyqnj61cBK?usp=sharing).

# Train
- First organize the training dataset according to our code implementation
### Start training
```shell
$ cd DPANet
$ python train.py
```
