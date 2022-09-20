# guided-diffusion-keras

Text to Image Diffusion model in Keras.


To try out the code you can use the notebook below in colab. It is set to train on the fashion mnist dataset.
You should be able to see resonable generatations within 5 epochs (5 minutes on a V100 GPU)
A notebook to train a model on fashion MNIST or Cifar 10/100 or a npz file with embeddings:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EoGdyZTGVeOrEnieWyzjItusBSes_1Ef?usp=sharing)

CIFAR 10/100
You can get reasonable results after 20 epochs for CIFAR 10 and 30 epochs for CIFAR 100.
Training 50-100 epochs is even better.
CFG = 2
[1, 2, 3]*64, depth=3, [0, 1, 0]


Fashion MNIST:
You can get reasonable results after 5-10 epochs.
Note: Fashion MNIST does not work well with high CFG - keep it at 1.
[1, 2, 3]*64, depth=2, [0, 1, 0] or [[0, 0, 0]]

Time for 1 epoch (50k examples) - About 1 minute (V100) / 2 minutes (P100) per epoch (V100)
So you can get a decent Fashion Mnist model in 5 minutes!!

Aesthetic 100k
You can get good results after 15 epochs
~ 10 minutes per epoch (V100)
[1, 2, 3, 4]*64, depth=3, [0, 0, 1, 0]

Credits:

- Original Unet implementation in this [excellent blog post](https://keras.io/examples/generative/ddim/) - most of the code and Unet architecture in `denoiser.py` is based on this. I have added 
additional text/CLIP/masking embeddings/inputs and cross/self attention.


