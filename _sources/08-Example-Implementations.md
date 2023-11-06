# Implementation Examples

## LSGM: Score-based Generative Modeling in Latent Space

[LSGM: Score-based Generative Modeling in Latent Space](https://github.com/NVlabs/LSGM) trains a score-based generative model (a.k.a. a denoising diffusion model) in the latent space of a variational autoencoder. In the latent score-based generative model (LSGM), data is mapped to latent space via an encoder $q(z_0|x)$ and a diffusion process is applied in the latent space $(z_0 → z_1)$. Synthesis starts from the base distribution $p(z_1)$ and generates samples in latent space via denoising $(z_0 ← z_1)$. Then, the samples are mapped from latent to data space using a decoder $p(x|z_0)$. The model is trained end-to-end. It currently achieves state-of-the-art generative performance on several image datasets.

<figure>
<img src="https://github.com/NVlabs/LSGM/raw/main/img/LSGM.png" alt="LSGM" width="600"/>
<figcaption> LSGM trains a score-based generative model in the latent space of a variational autoencoder.</figcaption>
</figure>


## Hierarchical Text-Conditional Image Generation with CLIP Latents
[Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/abs/2204.06125)

<figure>
<img src="imgs/unClip.png" alt="unClip" width="600"/>
<figcaption>Hierarchical Text-Conditional Image Generation with CLIP Latents.</figcaption>
</figure>

The two-stage diffusion model **unCLIP** heavily utilizes the CLIP text encoder to produce text-guided images at high quality. Given a pretrained CLIP model and paired training data for the diffusion model, $(x,y)$, where $x$ is an image and $y$ is the corresponding caption, we can compute the CLIP text and image embedding, $c^t(y)$ and , $c^i(x)$ respectively. The unCLIP learns two models in parallel: 
- A prior model $p(c^{i}|y)$: outputs CLIP image embeddings $c^i$ given the text $y$
- A decoder $p(x|c^{i},[y])$: generates the image $x$ given CLIP image embedding $c^i$ and optionally the original text $y$.

## LION (NeurIPS 2022):

[LION: Latent Point Diffusion Models for 3D Shape Generation](https://nv-tlabs.github.io/LION/)
<figure>
<img src="imgs/lion.png" alt="LION" width="600"/>
<figcaption>Hirarchical two-stage VAE model using diffusion priors (diffusion process in latent space).</figcaption>
</figure>

They introduce the hierarchical Latent Point Diffusion Model (LION) for 3D shape generation. LION is set up as a variational autoencoder (VAE) with a hierarchical latent space that combines a global shape latent representation with a point-structured latent space. For generation, they train two hierarchical DDMs in these latent spaces. The hierarchical VAE approach boosts performance compared to DDMs that operate on point clouds directly, while the point-structured latents are still ideally suited for DDM-based modeling. 

## MeshDiffusion: Score-based Generative 3D Mesh Modeling

[MeshDiffusion: Score-based Generative 3D Mesh Modeling](https://arxiv.org/abs/2303.08133)
[MeshDiffusion Repository](https://github.com/lzzcd001/MeshDiffusion)
<figure>
<img src="imgs/mesh_teaser.jpg" alt="mesh_teaser" width="600"/>
<figcaption>MeshDiffusion: Score-based Generative 3D Mesh Modeling</figcaption>
</figure>


The paper presents a novel approach to generating realistic 3D shapes, a task that has applications in automatic scene generation and physical simulation. The authors argue that meshes are a more practical representation for 3D shapes than alternatives like voxels and point clouds, as they allow for easy manipulation of shapes and can fully leverage modern graphics pipelines.

Previous methods for generating meshes have relied on sub-optimal post-processing and often produce overly-smooth or noisy surfaces without fine-grained geometric details. To overcome these limitations, the authors propose a new method that takes advantage of the graph structure of meshes and uses a simple yet effective generative modeling method to generate 3D meshes.

The method involves representing meshes with deformable tetrahedral grids and training a diffusion model on this direct parameterization. The authors demonstrate the effectiveness of their model on multiple generative tasks.

## Zero-1-to-3: Zero-shot One Image to 3D Object
[Zero-1-to-3: Zero-shot One Image to 3D Object](https://zero123.cs.columbia.edu/)

<figure>
<img src="imgs/zero123.png" alt="zero123" width="600"/>
<figcaption>They learn a view-conditioned diffusion model that can subsequently control the viewpoint of an image containing a novel object (left). Such diffusion model can also be used to train a NeRF for 3D reconstruction (right).</figcaption>
</figure>

The paper introduces Zero-1-to-3, a framework for changing the camera viewpoint of an object given just a single RGB image. To perform novel view synthesis in this under-constrained setting, they capitalize on the geometric priors that large-scale diffusion models learn about natural images. their conditional diffusion model uses a synthetic dataset to learn controls of the relative camera viewpoint, which allow new images to be generated of the same object under a specified camera transformation. Even though it is trained on a synthetic dataset, our model retains a strong zero-shot generalization ability to out-of-distribution datasets as well as in-the-wild images, including impressionist paintings. Their viewpoint-conditioned diffusion approach can further be used for the task of 3D reconstruction from a single image. 

## Repository of Implementations for different Approaches of 3D Generative Models

[threestudio](https://github.com/threestudio-project/threestudio): is a unified framework for 3D content creation from text prompts, single images, and few-shot images, by lifting 2D text-to-image generation models.

<figure>
<img src="https://user-images.githubusercontent.com/19284678/245695565-f48eca9f-45a7-4092-a519-6bb99f4939e4.gif" alt="threestudio" width="700"/>
<figcaption>Examples of different models in the Threestudio repository.</figcaption>
</figure>

<figure>
<img src="https://user-images.githubusercontent.com/22424247/252861573-0783ad8c-02ba-419b-aea1-9f5ecb16ac1b.gif" alt="threestudio-zero123" width="500"/>
<figcaption>Examples of Zero-1-to-3 from the Threestudio repository.</figcaption>
</figure>
