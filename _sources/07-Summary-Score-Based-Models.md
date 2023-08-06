# Summary: Score-Based Models

Despite differences, discrete-time and continuous-time models follow nearly identical generative processes. Continuous models are even more straightforward to handle:

   1. They are versatile and can be transformed to discrete models through time discretization.
   2. Their behavior can be described by well-studied SDEs.
   3. They use standard numerical SDE solvers.
   4. They can be converted to simple ordinary differential equations (ODEs).

As noted, diffusion models generate samples by reversing the diffusion process that maps a Gaussian base distribution to complex data. This mapping, in continuous-time diffusion models represented by the generative SDE, is often complex due to the neural network approximating the score function $\nabla_{x_{t}}\log p_{t}(x_{t})$. Solving it with numerical integration techniques can require 1000s of calls to deep neural networks for sample generation. Consequently, these models are slower than alternatives like generative adversarial networks (GANs), which only require a single network call.
