# README

```bash
# First of all, extract dataset as follows
tar xf traj.tar.xz
```

- This repo is an implementation of VQ-VAE-based image tokenizer for a model-based reinforcement learning.
- In the MBRL like [World Model](https://arxiv.org/abs/1803.10122), dynamic and behavior are estimated through the encoded observations.
- So, to implement the World Model-like MBRL, it is very important to encoding observations effectively.
- The dataset is collected from the [Car Racing](https://gymnasium.farama.org/environments/box2d/car_racing/) environment.

## References

1. https://github.com/CompVis/taming-transformers
2. https://github.com/eloialonso/iris
3. https://github.com/nadavbh12/VQ-VAE
