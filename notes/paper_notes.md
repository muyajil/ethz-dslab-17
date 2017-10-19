Standard object detection cnns require multiple passes over the image to identify and locate multiple objects.
In a traditional cnn, there are two fully-connected (non-convolutional) layers as the final layers of the network. 
Moving-window methods are able to produce rectangular bounding boxes, but cannot produce object silhouettes.


## [Semantic Perceptual Image Compression using Deep Convolution Networks](https://arxiv.org/abs/1612.08712)(2017)
The paper presents a cnn architecture for lossy image compression, which generates a map that highlights (multiple) semantically-salient regions so that they can be encoded at higher quality as compared to background regions.
They improve the "visual quality" (PSNR, SSIM, VIFP, PSNR-HVSM, etc..) of standard jpeg by using a higher bit rate to encode image regions flagged by their model as containing content of interest and lowering the bit rate elsewhere in the image.
They only improve the JPEG encoder, so a standard JPEG decoder can be used.
In single pass, multiple objects can be detected by learning separate feature maps for each of a set of object classes and then summing over the top features.
Thapproach requires image-level labels of object classes. (We could use it for star images, with only one object category.)
Multi-Structure Region of Interest (MS-ROI), allows to effectively train on localization tasks independent of the number of classes
Experiments are done on the Kodak PhotoCD dataset and the MIT Saliency Benchmark dataset.
