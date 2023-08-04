# Summary: Score-Based Models

... some text ...


## Why Continuous-Time Diffusion Models aka Score based Models?
Despite differences, discrete-time and continuous-time models follow nearly identical generative processes. Continuous models are even more straightforward to handle:

   1. They are versatile and can be transformed to discrete models through time discretization.
   2. Their behavior can be described by well-studied SDEs.
   3. They use standard numerical SDE solvers.
   4. They can be converted to simple ordinary differential equations (ODEs).

As noted, diffusion models generate samples by reversing the diffusion process that maps a Gaussian base distribution to complex data. This mapping, in continuous-time diffusion models represented by the generative SDE, is often complex due to the neural network approximating the score function $\nabla_{x_{t}}\log p_{t}(x_{t})$. Solving it with numerical integration techniques can require 1000s of calls to deep neural networks for sample generation. Consequently, these models are slower than alternatives like generative adversarial networks (GANs), which only require a single network call.


## Another view on Score-Based Models

Introduced by Song et al. in “Noise-Conditioned Score Network” (2019 https://arxiv.org/abs/1907.05600) score-based models where motivated by the idea of using the score function $\nabla_x \log p(x)$ to draw samples from a distribution $p(x)$ using langevin dynamics avoiding the need to compute the normalizing constant $Z_\theta$. This idea is shortly explained here. 

For a given dataset $\{x_1, x_2, ... x_N\}$ with the underlying data distribution $p(x)$ (each point is independently drawn), a generative model is fitted to this data distribution in order to synthesize new samples at will by sampling from the distribution. In likelihood based models we represent the probability distribution as probability density function (p.d.f.): 

$$
p_{\theta}(x)=\frac{e^{-f_{\theta}(x)}}{Z_{\theta}}
$$

where $f_{\theta}(x) \in \mathbb{R}$ is a real valued function parameterized by learnable parameter $\theta$ and $Z_{\theta}$ is a normalizing constant dependant on $\theta$, such that $\int p_{\theta}(x)dx=1$. The function $f_{\theta}(x) \in \mathbb{R}$ is often called an unnormalized probabilistic model or energy based model. We can learn $p_{\theta}(x)$ by maximizing the log-likelihood of the data 

$$
\max_{\theta}\sum_{i=1}^{N}\log p_{\theta}(x).
$$

The energy function $f_{\theta}(x)$ is typically parameterized by a flexible neural network. When training it as a likelihood model, we need to know the normalizing constant $Z_\theta$ by computing complex high-dimensional integrals, which is typically intractable. To make this normalizing constant tractable we can restrict the model architecture (like in normalizing flows using invertible neural network models) or approximate this constant (for example in variational autoencoders). 

Alternatively we can model the score function to avoid the difficulty of the intractable normalizing constant. Given a probablity density function $p(\mathbf{x})$, we define the score as

$$
\nabla_\mathbf{x} \log p(\mathbf{x}).
$$

In constrast to the other models, when computing the score, we obtain $\nabla_\mathbf{x} \log p_\theta(\mathbf{x}) = -\nabla_\mathbf{x} f_\theta(\mathbf{x})$ (since the log derivative of $Z_\theta$ is zero) which does not require computing the normalizing constant $Z_\theta$. A **score-based model** $s_{\theta}(x)\approx \nabla_\mathbf{x} \log p(\mathbf{x})$ is learned to approximate the score function. The score-based model is independant of the normalizing constant. As a result any neural network that maps an input vector $\mathbf{x} \in \mathbb{R}^d$ to an output vector $\mathbf{y} \in \mathbb{R}^d$ can be used as a score-based model, as long as the output and input have the same dimensionality. This yields huge flexibility in choosing model architectures. We can estimate the score using the Fischer-divergence, which is defined as

$$
\mathbb{E}_{p(x)}[||\nabla_x \log p(x)-s_{\theta}(x)||_2^2].
$$

There are several methods called score matching, which minimize the Fischer-divergence without knowledge of the underlying data distribution just using data samples (like https://arxiv.org/pdf/1905.07088.pdf). 

## Langevin Dynamics

To draw samples from a trained score-based model $s_{\theta}(x)\approx \nabla_\mathbf{x} \log p(\mathbf{x})$ we can use langevin dynamics. 

Langevin dynamics provida a way to draw samples from a dystribution $p(x)$ using is score function $\nabla_x \log p(x)$ using a Marcov Chain Monte Carlo procedure. Using an arbitrary prior $x_0 \sim \pi(x)$, we can iterate with the following

$$
x_{i+1}\leftarrow x_i + \epsilon \nabla_x\log p(x)+\sqrt{2\epsilon z_i}, i=0,1,..., K, 
$$

where $z_i\sim N(0,I)$. For $\epsilon \rightarrow 0$ and $K \rightarrow \infty$, the obtained $x_K$ converges to a sample fro $p(x)$ under sime regularity conditions. 

