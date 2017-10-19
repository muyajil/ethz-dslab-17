# Paper Evaluation for deep image compression

## [Semantic Perceptual Image Compression using Deep Convolution Networks](https://arxiv.org/abs/1612.08712)(2017)
[Tensorflow code available](https://github.com/iamaaditya/image-compression-cnn)
Learn map of (multiple) "semantically-salient" regions and encode those regions at higher quality (JPEG).
Needs Image-level labeling if multiple object-types should be detected.
Could be interesting for star images (only one object class, because we have no labels).
Measure "visual quality" with PSNR, SSIM, VIFP, PSNR-HVSM, etc..
Use [[Conv+Relu]^2+MaxPool]^5+MS-ROI

## [End-to-end Optimized Image Compression](https://arxiv.org/abs/1611.01704)
[Matlab code (apparently) available](http://www.cns.nyu.edu/~lcv/iclr2017/)

## [An End-to-End Compression Framework Based on Convolutional Neural Networks](https://arxiv.org/abs/1708.00838)
[Matlab testing code "available"](https://github.com/compression-framework/compression_framwork_for_tesing)
[Image -> ComCNN -> JPEG-enc -> JPEG-dec -> RecCNN -> Reconstruction]
They develop a unified end-to-end learning algorithm to simultaneously learn ComCNN and RecCNN.
The problem with existing image coding standards is that the quantization is not differentiable, which makes backpropagation challenging.
They train both CNNs simultaneously to overcome this problem.
They use MatConvNet.
They measure PSNR, SSIM.

## [Variable Rate Image Compression with Recurrent Neural Networks](https://arxiv.org/abs/1511.06085)(2016)
## [Full Resolution Image Compression with Recurrent Neural Networks](https://arxiv.org/abs/1608.05148#)(2017)
[Pre-trained Tensorflow model partially available.](https://github.com/tensorflow/models/tree/2390974a/compression)
Both papers are from the same group. (Google)


## [Lossy Image Compression with Compressive Autoencoders](https://arxiv.org/abs/1703.00395)(2017)
No source code linked.
This paper is from Twitter.
They propose a new approach to the problem of optimizing autoencoders for lossy image compression.
Autoencoders are difficult to optimize directly due to the inherent non-differentiabilty of the compression loss.

## [CAS-CNN: A Deep Convolutional Neural Network for Image Compression Artifact Suppression](https://arxiv.org/abs/1611.07233)
No copen source code found.
This paper is from ETHZ.

## [Learning Convolutional Networks for Content-weighted Image Compression](https://arxiv.org/abs/1703.10553)
