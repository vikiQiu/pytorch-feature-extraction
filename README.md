# pytorch-feature-extraction
Image Feature extraction using Pytorch with VAE and AE methods

**Run the Autoencoder:**

`CUDA_VISIBLE_DEVICES=0 python3 Autoencoder.py --dataset-dir='/home/vikiqiu/data/ILSVRC2012/' --model='vgg16' --load-model`


**encoder output channel = 32:**

`CUDA_VISIBLE_DEVICES=0 python3 Autoencoder.py --dataset-dir='/home/vikiqiu/data/ILSVRC2012/' --model='vgg16' --feature-channel=32`

`CUDA_VISIBLE_DEVICES=0 python3 Autoencoder.py --dataset-dir='/home/vikiqiu/data/ILSVRC2012/' --model='vgg16' --load-model --feature-channel=32`

**Evaluate**

`CUDA_VISIBLE_DEVICES=2 python3 Autoencoder.py --dataset-dir='/home/vikiqiu/data/ILSVRC2012/' --model='vgg16' --load-model --feature-channel=32 --evaluate  `


**Run the VAE:**

`CUDA_VISIBLE_DEVICES=2 python3 VAE.py --dataset-dir='/home/vikiqiu/data/ILSVRC2012/' --model='vgg16' --load-model --feature-channel=32`

**Run AE class**
`CUDA_VISIBLE_DEVICES=2 python3 AE_class.py --dataset-dir='/home/vikiqiu/data/ILSVRC2012/' --load-model --batch-size=64 --feature-channel=32`

It will automatically download vgg16_bn_xx.pth from website.
