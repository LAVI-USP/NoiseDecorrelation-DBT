


======

This repository contains the training and testing codes for the paper "[Suppressing noise correlation in digital breast tomosynthesis using convolutional neural network and virtual clinical trials]()", submitted to the IWBI 2022 conference. We used the OpenVCT from the University of Pennsylvania, available [here](https://sourceforge.net/p/openvct/wiki/Home/). Also, we used the neural network architecture from the original CycleGAN [repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Abstract:

It is well-known that x-ray systems featuring indirect detectors are affected by noise spatial correlation. This phenomenon might affect, in the case of digital breast tomosynthesis (DBT), the perception of small details in the image, such as microcalcifications. In this work, we propose the use of a deep convolutional neural network (CNN) to restore these images using the framework of a cycle generative adversarial network (cycle-GAN). To generate pairs of images for the training procedure, we used a virtual clinical trial (VCT) system. Two approaches were evaluated: in the first one, the network was trained to perform noise decorrelation by changing the frequency-dependency of the noise in the input image, but keeping the other characteristics of the input image, evaluated by means of the mean normalized squared error (MNSE). In the second approach, the network was trained to perform noise filtration and decorrelation, with the objective of generating an image with frequency-independent (white) noise and with MNSE equivalent to an acquisition with a radiation exposure four times greater than the input image. We found that in both cases the network successfully corrected the power spectrum (PS) of the input images.


## Some results:

Soon

## Reference:

If you use the codes, we will be very grateful if you refer to this [paper]():

> Soon

## Acknowledgments:

This work was supported by the São Paulo Research Foundation ([FAPESP](http://www.fapesp.br/) grant 2021/12673-6) and by the National Council for Scientific and Technological Development ([CNPq](http://www.cnpq.br/)) and by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior ([CAPES](https://www.gov.br/capes/pt-br) - finance code 001). We would like to thank the contribution of our lab members and the [Barretos Love Hospital](https://www.hcancerbarretos.com.br) for providing the images of DBT.


---
Laboratory of Computer Vision ([Lavi](http://iris.sel.eesc.usp.br/lavi/))  
Department of Electrical and Computer Engineering  
São Carlos School of Engineering, University of São Paulo  
São Carlos - Brazil

AI-based X-ray Imaging System ([AXIS](https://wang-axis.github.io))  
Department of Biomedical Engineering  
Rensselaer Polytechnic Institute  
Troy - USA
