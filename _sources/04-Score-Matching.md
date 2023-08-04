# Score Matching

One very naive idea is to directly learn the score function using regression. Here a **score-based model** $s_{\theta}(x_t,t)$ is learned to approximate the score function. Any neural network that maps an input vector $\mathbf{x} \in \mathbb{R}^d$ to an output vector $\mathbf{y} \in \mathbb{R}^d$ can be used as a score-based model, as long as the output and input have the same dimensionality. This yields huge flexibility in choosing model architectures. We can estimate the score using the Fischer-divergence (https://arxiv.org/pdf/1905.07088.pdf), which is defined as:

$$
\min_{\theta}\mathbb{E}_{t \sim U(0,T)}\mathbb{E}_{x_t \sim q_t(x_t)}[||s_{\theta}(x_t,t) - \nabla_{x_t} \log q_t(x_t)||_2^2].
$$

The score-based model is trained by minimizing the above objective function. The first expectation can be approximated by sampling $t$ from a uniform distribution $U(0,T)$. However for the second expectation we need to be able to sample from the distribution $q_t(x_t)$ and estimate $\nabla_{x_t} \log q_t(x_t)$. However $\nabla_{x_t} \log q_t(x_t)$ is not tractable and we only have access to samples from the underlying data distribution. 

## Denoising Score Matching
We can use the samples from the data distribution and diffuse individual data points based on a variance preserving SDE. Here $q_t(x_t|x_0)=\mathcal{N}(x_t;\gamma_t x_0, \sigma_t^2\mathbf{I})$, with $\gamma_t = \exp(-\frac{1}{2}\int_0^t\beta(s)ds)$ and $\sigma_t^2 = 1-\exp(-\int_0^t\beta(s)ds)$. Now we can sample from the data distribution $x_0 \sim p(x_0)$ and then sample from the diffusion process $x_t \sim q_t(x_t|x_0)$ using the normal distribution. the resulting probaility density is now tractable and we can estimate the gradient of the log density. The score-based model $s_{\theta}(x_t,t)$ is trained by minimizing the following **Denoising Score Matching** objective (https://arxiv.org/abs/1907.05600, https://arxiv.org/abs/2011.13456):
$$
\min_{\theta}\mathbb{E}_{t \sim U(0,T)}\mathbb{E}_{x_0 \sim p(x_0)}\mathbb{E}_{x_t \sim q_t(x_t|x_0)}[||s_{\theta}(x_t,t) - \nabla_{x_t} \log q_t(x_t|x_0)||_2^2].
$$

Since we model the diffusion process using a normal distribution which is variance preserving our score based model will have learned the score of the marginal distribution $q_t(x_t)$ after training: 
$$
s_{\theta}(x_t,t) \approx \nabla_{x_t} \log q_t(x_t)
$$

Based on this objective, we can derive the following implementation:

**1. Noise Prediction**

**2. Loss Weightings**

**3. Variance Reduction and Numerical Stability**


