# Text to Image Model in Keras

Training a simple CLIP conditioned Text to Image Diffusion model in Keras. See below for examples with prompts. The first examples are simler and condition on labels instead of prompts. If you're starting out I recommend starting with these easier datasets first.

To try out the code you can use the notebook below in colab. It is set to train on the fashion mnist dataset.
You should be able to see resonable image generations within 5 epochs (5 minutes on a V100 GPU)!
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EoGdyZTGVeOrEnieWyzjItusBSes_1Ef?usp=sharing)

CIFAR 10/100
You can get reasonable results after 20 epochs for CIFAR 10 and 30 epochs for CIFAR 100.
Training 50-100 epochs is even better.
CFG = 2
[1, 2, 3]*64, depth=3, [0, 1, 0]

Time for 1 epoch (50k examples) - About 1 minute (V100) / 2 minutes (P100) per epoch (V100)

Aesthetic 100k
You can get good results after 15 epochs
~ 10 minutes per epoch (V100)
[1, 2, 3, 4]*64, depth=3, [0, 0, 1, 0]

"Prompt: An Italian Villaga Painted by Picasso"

<img width="796" alt="image" src="https://user-images.githubusercontent.com/13619417/191168891-195a4bcb-94f1-429e-9878-2008027aeb24.png">

Prompt: "a small village in the alps, spring, sunset"

<img width="791" alt="image" src="https://user-images.githubusercontent.com/13619417/191170681-17c3820b-7fe1-44ad-bb51-30a521d465f7.png">


Prompt: "City at night"

<img width="796" alt="image" src="https://user-images.githubusercontent.com/13619417/191169091-2c8fbf9e-054e-49bf-b37a-8565bf112396.png">

CLIP interpolation: "A minimalist living room" -> "A Field in springtime, painting"

<img width="799" alt="image" src="https://user-images.githubusercontent.com/13619417/191169543-6d940748-495b-429f-96a1-e10e1da6bf89.png">

CLIP interpolation: "A lake in the forest in the summer" -> "A lake in the forest in the winter"

<img width="798" alt="image" src="https://user-images.githubusercontent.com/13619417/191169640-a8eb9a6f-7808-447a-af5a-094fcc8450ae.png">

Credits:

- Original Unet implementation in this [excellent blog post](https://keras.io/examples/generative/ddim/) - most of the code and Unet architecture in `denoiser.py` is based on this. I have added 
additional text/CLIP/masking embeddings/inputs and cross/self attention.

