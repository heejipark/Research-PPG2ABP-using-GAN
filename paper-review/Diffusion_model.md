#### Discriminative model VS Generative model
<code>Discriminative models</code> usually predict separate quantities given the data. On the other hand, <code>Generatrive models</code> allow you to synthesize novel data that is different from the real data but still looks just as realistic.

Now, I'm going to write more detail about one of the Generative models, a Diffusion Model.

## Diffusion model
Diffusion models are inspired by non-equilibrium thermodynamics (비평형 열역학). The models define a Markov chain of diffusion step to add random noise to data and then learn the method to reverse the diffusion process in order to construct desired data samples from the noise.
Unlike VAE or flow models, diffusion models are trained with a fixed procedure and the latent variable has high dimensionality (same as the original data). [[reference]](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) <br/>
<img src="./../img/06/generative-overview.png" height=400 width=750>
<br/>


### What is Diffusion Models?
Diffusion-based generative models have been proposed with similar idea, including <i>diffusion probabilistic models</i>, <i>noise-conditional score network</i>, and <i>denoising diffusion probabilistic models</i>.


### Forward diffusion
The process can be described as gradually applying Gaussian noise to the image until it becomes entirely unrecognisable. The process of noise application can be formulated as the Markov chain of sequential diffusion steps.

Let's assume that the images have a certain starting distribution q(x0). We can sample just one image from this distribution called x0. We want to perform a chain of diffusion steps x0 -> x1 -> ... -> xT, each step disintegrating the image more and more. To do this, there is a noising schedule, where for every t = 1,2,...,T.


### Reverse diffusion
To get back to the starting distribution q(x0) from the noised sample we would have to marginalize over all of the ways we could arise at x0 from the noise, including all of the latent states inbetween - ∫q(x0:T)dx1:T - which is intractable. So, if we cannot calculate it, surely we can… approximate it!





##### Good Reference
paper :
- Originally proposed in 2015 : https://arxiv.org/pdf/1503.03585.pdf

Blog: 
- https://deepseow.tistory.com/37
- https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/
- https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
- https://jmtomczak.github.io/blog/10/10_ddgms_lvm_p2.html
- https://towardsdatascience.com/diffusion-models-made-easy-8414298ce4da
- With pythorch code: https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/

Blog in Korean:
- https://www.lgresearch.ai/kor/blog/view/?seq=190&page=1&pageSize=12
- https://developers-shack.tistory.com/8

