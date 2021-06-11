# MagNet-MultitaskLearning

This is a PyTorch implementation based on [MagNet: a Two-Pronged Defense against Adversarial Examples](https://arxiv.org/pdf/1705.09064.pdf) 
by Dongyu Meng and Hao Chen, at CCS 2017. 

Different from MagNet, in this project we focus on detecting the adversarial examples.
So we don't apply classification branch behind the Reform part. We believe if we train the classification task and reconstruction 
task in a multi-task learning framework by sharing the same NN layers for feature embedding, the attack on classification will
 transfer to the reconstruction task. So the adversarial examples will lead to a larger reconstruction loss compared with the normal 
examples.

We evaluate the method on three datasets ([MNIST](http://yann.lecun.com/exdb/mnist/), [F-MNIST](https://github.com/zalandoresearch/fashion-mnist) and [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)) with four white box attack methods
([FGSM](https://arxiv.org/pdf/1412.6572.pdf), [JSMA](https://arxiv.org/pdf/1511.07528.pdf&xid=25657,15700023,15700124,15700149,15700186,15700191,15700201,15700237,15700242.pdf), [DeepFool](https://openaccess.thecvf.com/content_cvpr_2016/papers/Moosavi-Dezfooli_DeepFool_A_Simple_CVPR_2016_paper.pdf) and [C&W](https://arxiv.org/pdf/1608.04644.pdf?source=post_page---------------------------)).

## Requirements
Install PyTorch vis conda (python3) (you can also apply pip etc.)

`
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
`

We utilize [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
 for attack methods.

`
pip install adversarial-robustness-toolbox
`

## Running

### Training the classification and reconstruction model

Training with the multi-task learning method on MNIST dataset

`
python train.py --dataset mnist --id 1
`

Training without the multi-task learning method on MNIST dataset

`
python train.py --dataset mnist --id 2
`

Testing the reconstruction performance of the trained Auto-encoder

`
python train.py --dataset mnist --id 3
`

### Generate adversarial datasets

This step generates adversarial examples for the models trained above. You can set the 
attack method [here](https://github.com/AlanJiang98/MagNet-MultitaskLearning/blob/main/attackshans.py#L100)

`
python attackshans.py --dataset mnist
`

### Evaluate the detector

This step evaluates the detector performance

`
python defensemtl.py --dataset mnist  --tfp 0.005
`

## Credits
This project mainly refers to [MagNet](https://github.com/Trevillie/MagNet) and 
apply some modifications on it. 

If you want to research further, please cite their paper [MagNet: a Two-Pronged Defense against Adversarial Examples](https://arxiv.org/pdf/1705.09064.pdf) 

