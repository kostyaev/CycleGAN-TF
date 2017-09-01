# CycleGAN-TF
[Cycle-Consistent Adversarial Network](https://arxiv.org/pdf/1703.10593.pdf) implementation in tensorflow

## Train model
Run training with the following example command: 
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --name=cycle_gan_v1 --dataset=/storage/dataset --tensorboard=storage/tensorboard/ --batch_size=1 --save_freq=1000 --crop_size=128 --scale_size=144 --test_size=128 --ngf=64 --ndf=64 --ks=7 --pool_size=50 --normalization=instance --max_epochs=100 --decay_after=50
```

## Available data augmentations:
 - Mirror, random cropping;
 - Multi scale training;
 - Color augmentations: brightness, contrast, saturation.

## Reference
- The torch implementation of CycleGAN, https://github.com/junyanz/CycleGAN
