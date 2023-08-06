# Score Matching

One very naive idea is to directly learn the score function using regression. Here a **score-based model** $s_{\theta}(x_t,t)$ is learned to approximate the score function. Any neural network that maps an input vector $\mathbf{x} \in \mathbb{R}^d$ to an output vector $\mathbf{y} \in \mathbb{R}^d$ can be used as a score-based model, as long as the output and input have the same dimensionality. This yields huge flexibility in choosing model architectures. We can estimate the score using the Fischer-divergence (https://arxiv.org/pdf/1905.07088.pdf), which is defined as:

$$
\min_{\theta}\mathbb{E}_{t \sim U(0,T)}\mathbb{E}_{x_t \sim q_t(x_t)}[||s_{\theta}(x_t,t) - \nabla_{x_t} \log q_t(x_t)||_2^2].\tag{1}
$$

The score-based model is trained by minimizing the above objective function. The first expectation can be approximated by sampling $t$ from a uniform distribution $U(0,T)$. However for the second expectation we need to be able to sample from the distribution $q_t(x_t)$ and estimate $\nabla_{x_t} \log q_t(x_t)$. However $\nabla_{x_t} \log q_t(x_t)$ is not tractable and we only have access to samples from the underlying data distribution. 

## Denoising Score Matching
We can use the samples from the data distribution and diffuse individual data points based on a variance preserving SDE. Here $q_t(x_t|x_0)=\mathcal{N}(x_t;\gamma_t x_0, \sigma_t^2\mathbf{I})$, with $\gamma_t = \exp(-\frac{1}{2}\int_0^t\beta(s)ds)$ and $\sigma_t^2 = 1-\exp(-\int_0^t\beta(s)ds)$. Now we can sample from the data distribution $x_0 \sim p(x_0)$ and then sample from the diffusion process $x_t \sim q_t(x_t|x_0)$ using the normal distribution. the resulting probaility density is now tractable and we can estimate the gradient of the log density. The score-based model $s_{\theta}(x_t,t)$ is trained by minimizing the following **Denoising Score Matching** objective (https://arxiv.org/abs/1907.05600, https://arxiv.org/abs/2011.13456):

$$
\min_{\theta}\mathbb{E}_{t \sim U(0,T)}\mathbb{E}_{x_0 \sim p(x_0)}\mathbb{E}_{x_t \sim q_t(x_t|x_0)}[||s_{\theta}(x_t,t) - \nabla_{x_t} \log q_t(x_t|x_0)||_2^2]. \tag{2}
$$

Since we model the diffusion process using a normal distribution which is variance preserving our score based model will have learned the score of the marginal distribution $q_t(x_t)$ after training:

$$
s_{\theta}(x_t,t) \approx \nabla_{x_t} \log q_t(x_t)
$$

Based on the Denoising Score Matching objective (Equation 2), we can derive the following implementation:

**Step 1. Noise Prediction:**

Since we are using a normal distribution to model the diffusion process, we can rewrite $x_t$ as $x_t = \gamma_t x_0 + \sigma_t \epsilon$, where $\epsilon \sim \mathcal{N}(0,\mathbf{I})$. Now we can rewrite the Score function as:

$$
\nabla_{x_t} \log q_t(x_t|x_0) = - \nabla_{x_t} \frac{(x_t - \gamma_t x_0)^2}{2\sigma_t^2} = -\frac{(x_t - \gamma_t x_0)}{\sigma_t^2} = -\frac{(\gamma_t x_0 + \sigma_t \epsilon - \gamma_t x_0)}{\sigma_t^2} = -\frac{\epsilon}{\sigma_t}
$$

We can see that the score function is a function of the noise $\epsilon$ and $\sigma_t$. We can use a neural network to predict $s_{\theta}(x_t,t) := -\frac{\epsilon_{\theta}(x_t,t)}{\sigma_t}$. The objective function can now be rewritten as:

$$
\min_{\theta}\mathbb{E}_{t \sim U(0,T)}\mathbb{E}_{x_0 \sim p(x_0)}\mathbb{E}_{\epsilon \sim \mathcal{N}(0,\mathbf{I})}[\frac{1}{\sigma_t^2}||\epsilon - \epsilon_{\theta}(x_t,t)||_2^2]. \tag{3}
$$

> Note that we can also model $s_{\theta}(x_t,t)$ as defined before and change the objective to: 
>
> $$ \min_{\theta}\mathbb{E}_{t \sim U(0,T)}\mathbb{E}_{x_0 \sim p(x_0)}\mathbb{E}_{\epsilon \sim \mathcal{N}(0, \mathbf{I})}[\frac{1}{\sigma_t^2}||\sigma_t s_{\theta}(x_t,t) + \epsilon||_2^2]$$ 
> 

**Step 2. Loss Weighting:**

We can also add a weighting term to the objective function to balance the loss between the different time steps. This is done by adding a weighting term $\lambda(t)$ to the objective function: 

$$
\min_{\theta}\mathbb{E}_{t \sim U(0,T)}\mathbb{E}_{x_0 \sim p(x_0)}\mathbb{E}_{\epsilon \sim \mathcal{N}(0,\mathbf{I})}[\frac{\lambda(t)}{\sigma_t^2}||\epsilon - \epsilon_{\theta}(x_t,t)||_2^2]. \tag{4}
$$

The weighting term can defined as $\lambda(t) = \sigma_t^2$ to favour perceptual quality or $\lambda(t) = \beta(t)$ which is the negative ELBO and the same objective as for the denoising diffusion models if noise $\epsilon$ subtraction is used (discussed in the "Illustration of Diffusion Process"). 

You can check out "Elucidating the Design Space of Diffusion-Based Generative Models" (https://arxiv.org/abs/2206.00364) for more details on the different weighting schemes as well as training and model optimizations.

**Step 3. Variance Reduction and Numerical Stability:**

If we choose $\lambda(t) = \beta(t)$ the loss is heavily amplified for sampling $t$ close to $0$ since $\sigma_t^2 \rightarrow 0$ for $t \rightarrow 0$. This can lead to numerical instabilities and variance in the training process. To reduce the variance we can train with a small time cut-off $t_{cut}$ (for example for $t_{cut} \approx 10^{-5}$): $t \sim U(t_{cut},T)$. This is a very easy way to reduce the variance in the training process. Alternatively we can use importance sampling with a sampling distribution $p(t) \propto \frac{\lambda(t)}{\sigma_t^2}$ to reduce the variance.
