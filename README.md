# SelfAttentionNN_VMC_Correlated_Electron_problem

Self Attention Neural Network with Variational Monte Carlo in Correlated Electron Problem

This project implements a Variational Monte Carlo (VMC) method augmented by a neural network-based ansatz—such as SlaterNet or a self-attention model—to approximate quantum wave functions. The neural network maps an electron configuration **R** to a complex amplitude Ψ(**R**; θ), where θ are the trainable weights. The squared wave function |Ψ|² is treated as a probability density, and the Metropolis algorithm is used to sample many electron configurations accordingly. For each sample, the local energy $E_{\text{loc}} = \frac{\hat{H}\Psi}{\Psi}$ is computed, and the mean over samples yields the variational energy estimate $E(\theta) = \langle E_{\text{loc}} \rangle$. Gradients of this energy with respect to θ are computed via automatic differentiation, enabling gradient-based optimizers (e.g., Adam, natural gradient) to update θ and minimize energy. Repeating this sample → estimate → update loop allows the network to learn a correlated ground-state wave function, combining Monte Carlo efficiency with the expressiveness of deep learning.


rmb to update later