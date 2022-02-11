# 3DMNIST

I present a method for rendering novel views of a scene. The goal is to optimize a volume such that it can represent a sample from a probability distribution from almost all viewing angles. I tuned this idea to the 
[MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
, thus the name, 3DMNIST

#

#### watch this model get trained in real time. Buy running train.py
```python
python train.py
```

#### view previously optimized volumes by running

```python
python inference.py
```

#

## Summary

The vanilla Generative model inputs a latent code and outputs a sample from a probability distribution. The generator makes use of a discriminator, The discriminator is a simple classifier which is forced to recognize what is inside and outside of its training distribution. What is great about this process is its fully differentiable https://arxiv.org/abs/1406.2661.

#

#### The Main Idea.

I replaced the Generator model with a simple volume. The volume has 2 major attributes: 
Attenuation volume (almost means transparency)  and Absorbance (How much light gets absorbed).

Instead of a latent vector input, I pass a rotation matrix ``` euler_matrix([randn,randn,0]) ``` and the volume into a differentiable volumetric renderer. This outputs a 2D isometric projection of the volume.

```python
isometric_image =  renderer(volume,rotation_matrix) 
````

In order to train this model I introduced a perceptual discriminator which is trained using standard GAN losses. This forces all outputs of the renderer to represent the distribution


#### Some Interesting things happens to the output

Because all projected volumes are forced to resemble its training data, interesting things happen.


# 

#### Issues

**1. Mode Collapse.**  - During training, I discovered 3DMnist converges to zeros in most cases or other numbers which contain zeros. 
*I am currently researching ways to resolve this issue*



#


### Contributions

This research is very much open for contributions. Feel free to send Issues and PR
