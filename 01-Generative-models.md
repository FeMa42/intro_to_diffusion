# Generative Models

Generative models, for real-world applications, must meet the following criteria:

   1. **High-quality sampling:** Models should generate high quality results, like clear speech or realistic images, crucial for user interaction.
   2. **Mode coverage and sample diversity:** Models must capture data's complexity and diversity without compromising quality.
   3. **Fast and computationally inexpensive sampling:** Fast and efficient generation is necessary, especially for real-time applications like image editing.

While many deep learning models prioritize quality, diversity and efficiency are equally vital. Representing data's full range prevents biases and highlights valuable outliers, like dangerous traffic scenarios. Decreasing complexity and sampling time allows real-time use, reduces environmental impact, and lowers energy consumption.

This challenge of balancing these criteria has been defined as the 'generative learning trilemma', as most existing methods struggle to meet all three simultaneously (see also https://developer.nvidia.com/blog/improving-diffusion-models-as-an-alternative-to-gans-part-1/).

<figure>
<img src="imgs/genM-trilemma.png" alt="GenModelsTrilema" width="600"/>
<figcaption> Generative learning trilemma (from: https://developer.nvidia.com/blog/improving-diffusion-models-as-an-alternative-to-gans-part-1/)</figcaption>
</figure>

# Score Matching and Diffusion Models
Score-based generative models and denoising diffusion models are a new class of generative models. While beeing derived from different principles, they are very closely related and can be viewed as two sides of the same coin. In this notebook we will focus more on the score-based generative models, but we will also discuss the diffusion process in the beginning and diffusion models at the end.

These Methods have several advantages over existing model families: 

## Advantages of Score-Based (Diffusion-Denoising) Models
- GAN-level **sample quality** without adversarial training
- **flexible** model architectures
- exact **log-likelihood** computation (This is actual pretty awesome. There is a close conncetions to normalizing flows if we use Score based models)
- **inverse problem solving** without re-training models


<figure>
<img src="imgs/generative-overview.png" alt="GenerativeModelOverview" width="800"/>
<figcaption> Generative learning Overview (from: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)</figcaption>
</figure>

**Resources:**

- A list of great resource can be found here: https://github.com/diff-usion/Awesome-Diffusion-Models

- **Diffusion-Denoising Models** For a very good introduction into diffusion models you can check out the blog post from Lilian Weng: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

- **Score-Based Generative Models** A good introduction is given by Yang Song here: https://yang-song.net/blog/2021/score/, https://www.youtube.com/watch?v=wMmqCMwuM2Q 

- CVPR 2022 Tutorial on **Denoising Diffusion-based Generative Modeling - Foundations and Applications**: https://cvpr2022-tutorial-diffusion-models.github.io/ 