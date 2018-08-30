# pytorch-feature-extraction
Image Feature extraction using Pytorch with VAE and AE methods

Run the Autoencoder:

`CUDA_VISIBLE_DEVICES=0 python3 Autoencoder.py --dataset-dir='/home/vikiqiu/data/ILSVRC2012/' --model='vgg16' --load-model`

encoder output channel = 32:
`CUDA_VISIBLE_DEVICES=0 python3 Autoencoder.py --dataset-dir='/home/vikiqiu/data/ILSVRC2012/' --model='vgg16' --feature-channel=32`
`CUDA_VISIBLE_DEVICES=0 python3 Autoencoder.py --dataset-dir='/home/vikiqiu/data/ILSVRC2012/' --model='vgg16' --load-model --feature-channel=32`
