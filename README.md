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


**Run AE class:** A model with classifier and decoder both.  

`CUDA_VISIBLE_DEVICES=2 python3 AE_class.py --dataset-dir='/home/vikiqiu/data/ILSVRC2012/' --load-model --batch-size=64 --feature-channel=32`

`CUDA_VISIBLE_DEVICES=2 python3 AE_class.py --dataset-dir='/home/vikiqiu/data/ILSVRC2012/ILSVRC2012_img_train_subset' --test-dir='/home/vikiqiu/data/ILSVRC2012' --batch-size=64 --feature-channel=32 --alpha=0.1 --dataset="ImageNet1000-train-sub" --cover-dir="/home/vikiqiu/data/cover/cover0712" --load-model`

It will automatically download vgg16_bn_xx.pth from website.

**Run Vgg16 classifier:** To evaluate the performence pretrained vgg16_bn model provided by pytorch on ImageNet Validation dataset.

`CUDA_VISIBLE_DEVICES=2 python3 vgg_classifier.py --dataset-dir='/home/vikiqiu/data/ILSVRC2012/' --batch-size=64`
