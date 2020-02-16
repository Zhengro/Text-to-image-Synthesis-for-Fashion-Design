# Text-to-image-Synthesis-for-Fashion-Design
With the aim to study the respective impacts of **network architectures** and **training data** on the performance of text-to-image synthesis, two GAN-based algorithms are adopted, namely, Attentional Generative Network ([AttnGAN](https://github.com/taoxugit/AttnGAN)) and Stacked Generative Network ([StackGAN](https://github.com/hanzhanggit/StackGAN)). They are applied on two fashion datasets separately, i.e., [FashionGen](https://www.fashion-gen.com/) and [Fashion Synthesis](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/FashionSynthesis.html).

Please refer to the links above for the architectures of two models and their original implementations. Some statistics of two datasets referenced in our experiments are summarized in the following table.

 Terms | FashionGen | Fashion Synthesis
  :-------------------------:|:-------------------------:|:-------------------------:
  Number of samples | train: 260490 validation: 32528 test: 32528 | train: 70000 test: 8979
  Number of categories | 48 | 19
  Resolution | 256X256 | 128X128
  Poses | multiple | multiple
  Background | white | varied
  Description | detailed | brief

## Examples generated by AttnGAN
### Generated FashionGen examples
For each generated FashionGen example, the first row gives the **_three generated images_** from AttnGAN, followed by the **_corresponding real image_** in the validation set. The first two generated images are bilinearly upsampled to be of the same size as the third one for better visualization. The second row and last row illustrate top-5 words attended by the first and second attention model, respectively.

example 1 | example 2
:-------------------------:|:-------------------------:
![](/AttnGAN/FashionGen/sample1_gen.png)  | ![](/AttnGAN/FashionGen/sample11_gen.png) 

Given the same text from the validation set of FashionGen, six images are generated to illustrate the image diversity.

example 3 | example 4
:-------------------------:|:-------------------------:
![](/AttnGAN/FashionGen/diversity2_gen.png) | ![](/AttnGAN/FashionGen/diversity3_gen.png)
### Generated Fashion Synthesis examples
Similarly, for each generated Fashion Synthesis example, the first row gives the **_generated images from the first two stages_** of AttnGAN, followed by the **_corresponding real image_** in the test set. The first generated images is bilinearly upsampled to be of the same size as the last one for better visualization. The second row illustrates top-5 words attended by the single attention model.

example 5 | example 6
:-------------------------:|:-------------------------:
![](/AttnGAN/Fashion%20Synthesis/sample7_syn.png) | ![](/AttnGAN/Fashion%20Synthesis/sample10_syn.png)

Given the same text from the test set of Fashion Synthesis, six images are generated to illustrate the image diversity.

example 7 | example 8
:-------------------------:|:-------------------------:
![](/AttnGAN/Fashion%20Synthesis/diversity2_syn.png) | ![](/AttnGAN/Fashion%20Synthesis/diversity3_syn.png)
## Examples generated by StackGAN
We display the images synthesized by StackGAN given the same descriptions as example 1 and example 2 (i.e., example 1* and example 2*). For each example, the left image of size 64X64 is bilinearly upsampled to be of the same size as the right one for better visualization. Likewise, we display the images synthesized by StackGAN given the same descriptions as example 5 and example 6 (i.e., example 5* and example 6*).

example 1* | example 2* | example 5* | example 6*
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](/StackGAN/s1_gen.png)  | ![](/StackGAN/s11_gen.png) | ![](/StackGAN/s7_syn.png) | ![](/StackGAN/s10_syn.png)
