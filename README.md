# DPANet
The implementation of "Learning Dual-Pixel Alignment for Defocus Deblurring".

# Prerequisites  
- The code has been tested with the following environment
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

### Preparation (for DCNv2)
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
- First, crop the images of DPDD train set into 512*512 patches using the same settings as DPDNet. (You can use ```$ python ./image_to_patch_filter.py```from [DPDD](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel) to get the patches.)
- After getting the training patches, please organize the training dataset according to our code implementation:
```$ mkdir dpdd_datasets/dpdd_16bit```,
then move the train/test folder into the directory just created.
### Start training
```shell
$ cd DPANet
$ python train.py
```
During training, we first train DPANet with MSE loss. After that, we choose the checkpoint that gives the best result among all the epochs (about 300 epochs, [example ckpt trained with MSE](https://drive.google.com/drive/folders/1B8ynZ-MIuPezsHKBt2w9DWRUDQQO52uw?usp=sharing)) and finetune it with Charbonnier loss.

# Results
Here we give results of different methods on DPDD and PIXEL datasets.
  - [DPDD results](https://drive.google.com/drive/folders/1F0P24qFEdC3POO6wF8m7c17-0nKTsbvw?usp=sharing)
  - [PIXEL results](https://drive.google.com/drive/folders/1F0P24qFEdC3POO6wF8m7c17-0nKTsbvw?usp=sharing)
