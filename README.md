# Text to Image Model in Keras

#### See [blogpost](https://apapiu.github.io/2022-10-06-diffusion_text_to_img_keras/) for more details.

#### NEW: [Kaggle notebook](https://www.kaggle.com/code/apapiu/train-latent-diffusion-in-keras-from-scratch) that trains a 128*128 Latent Diffusion model on the Kaggle kernel hardware (P100 GPU). This is should be similar to the code for Stable diffusion.

Codebase to train a CLIP conditioned Text to Image Diffusion model on Colab in Keras. See below for notebooks and examples with prompts.


Images generated for the prompt: `A small village in the Alps, spring, sunset` 

<img width="500" alt="image" src="https://user-images.githubusercontent.com/13619417/191170681-17c3820b-7fe1-44ad-bb51-30a521d465f7.png">

Images generated for the prompt: `Portrait of a young woman with curly red hair, photograph` 

<img width="500" alt="image" src="https://user-images.githubusercontent.com/13619417/192167167-5b308069-4483-451e-8aef-0a8dc1d1c10f.png">


(more exampes below - try with your own inputs in Colab here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/123iljowP_b5o-_6RjHZK8vbgBkrNn8-c?usp=sharing) )


## Table of Contents:
- [Into](#intro)
- [Notebooks](#notebooks)
- [Usage](#usage)
- [Model Setup](#model-setup)
- [Data](#data)
- [Examples](#examples)

## Intro

The goal of this repo is to provide a simple, self-contained codebase for Text to Image Diffusion that can be trained in Colab in a 
reasonable amount of time. 

While there are a lot of great resources around the math and usage of diffusion models I haven't found many specifically
focused on _training_ text to img diffusion models. 
Particularly the idea of training a Dall-E 2 or Stable Diffusion like model feels like a daunting task requiring immense 
computational resources and data. Turns out you can accomplish quite a lot with little resources and without having a PhD in thermodynamics!
The easiest way to get aquainted with the code is thru the notebooks below. 

#### Credits/Resources

- Original Unet implementation in this [excellent blog post](https://keras.io/examples/generative/ddim/) - most of the code and Unet architecture in `denoiser.py` is based on this. I have added 
additional text/CLIP/masking embeddings/inputs and cross/self attention.
- [Conditional CIFAR Model in Pytorch](https://colab.research.google.com/drive/1IJkrrV-D7boSCLVKhi7t5docRYqORtm3#scrollTo=TAUwPLG92r89)
- [Laion Aesthetics 6.5+ Dataset](https://laion.ai/blog/laion-aesthetics/) - The 625K image-text pairs with predicted aesthetics scores of 6.5 or higher was used for training.
- [Text 2 img package](https://github.com/hmiladhia/img2text) 
- [Variational Diffusion Models (Paper)](https://arxiv.org/abs/2107.00630)
- [DDIM Paper](https://arxiv.org/abs/2010.02502)

## Notebooks

If you are just starting out I recommend trying out the next two notebook in order. The first should be able to get you 
recognizable images on the Fashion Mnist dataset within minutes!

- Train Class Conditional Fashion MNIST/CIFAR [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16rJUyPn72-C30mZRUr-Oo6ZjYS89Z3yH?usp=sharing) 
  - To try out the code you can use the notebook above in colab. It is set to train on the fashion mnist dataset.
You should be able to see reasonable image generations withing 5 epochs (5-20 minutes depending on GPU)!
  - For CIFAR 10/100 - you just have to change the `file_name`. You can get reasonable results after 25 epochs for CIFAR 10 and 40 epochs for CIFAR 100.
Training 50-100 epochs is even better. 

- Train CLIP Conditioned Text to Img Model on 115k 64x64 images+prompts sampled from the Laion Aesthetics 6.5+ dataset. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EoGdyZTGVeOrEnieWyzjItusBSes_1Ef?usp=sharing) 
  - You can get recognizable results after ~15 epochs
  ~ 10 minutes per epoch (V100)
- Test Prompts on a model trained for about 60 epochs (~60 hours on 1 V100) on entire 600k Laion Aesthetics 6.5+. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/123iljowP_b5o-_6RjHZK8vbgBkrNn8-c?usp=sharing) 
  - This model has about 40 million parameters (150MB) and can be downloaded from [here](https://huggingface.co/apapiu/diffusion_model_aesthetic_keras/blob/main/model_64_65_epochs.h5)
  - The examples in this repo use this model
 
## Usage

The model architecture, training parameters, and generation parameters are specified in a yaml file see [here](https://github.com/apapiu/guided-diffusion-keras/tree/main/guided_diffusion/configs) for examples. If unsure you can use the base_model. The get_train_data is built to work with various known datasets. If you have
your own dataset you can just use that instead. `train_label_embeddings` is expected to be a matrix of embedding the model conditions on (usually some embedding of text but could be anything).


```python
config_path = "guided-diffusion-keras/guided_diffusion/configs/base_model.yaml"

trainer = Trainer(config_path)
print(trainer.__dict__)

train_data, train_label_embeddings = get_train_data(trainer.file_name) #OR get your own images and label embeddings in matrix form.
trainer.preprocess_data(train_data, train_label_embeddings)
trainer.initialize_model()
trainer.data_checks(train_data)
trainer.train()
```

## Model Setup

The setup is fairly simple. 

We train a denoising U-NET neural net that takes the following three inputs:
- `noise_level` (sampled from 0 to 1 with more values concentrated close to 0)
- image (x) corrupted with a level of random noise
  - for a given `noise_level` between 0 and 1 the corruption is as follows:
    - `x_noisy = x*(1-noise_level) + eps*noise_level where eps ~ np.random.normal`
- CLIP embeddings of a text prompt
  - You can think of this as a numerical representation of a text prompt (this is the only pretrained model we use). 

The output is a prediction of the denoised image - call it `f(x_noisy)`.

The model is trained to minimize the absolute error `|f(x_noisy) - x|` between the prediction and actual image
(you can also use squared error here). Note that I don't reparametrize the loss in terms of the noise here to keep things simple.

Using this model we then iteratively generate an image from random noise as follows:
    
         for i in range(len(self.noise_levels) - 1):

            curr_noise, next_noise = self.noise_levels[i], self.noise_levels[i + 1]

            # predict original denoised image:
            x0_pred = self.predict_x_zero(new_img, label, curr_noise)

            # new image at next_noise level is a weighted average of old image and predicted x0:
            new_img = ((curr_noise - next_noise) * x0_pred + next_noise * new_img) / curr_noise

The `predict_x_zero` method uses classifier free guidance by combining the conditional and unconditional
prediction: `x0_pred = class_guidance * x0_pred_conditional + (1 - class_guidance) * x0_pred_unconditional`

A bit of math: The approach above falls within the VDM parametrization see 3.1 in [Kingma et al.](https://arxiv.org/pdf/2107.00630.pdf):

$$ z_t = \alpha_t*x + \sigma_t*\epsilon,  \epsilon ~ n(0,1)$$

Where $z_t$ is the noisy version of $x$ at time $t$.

generally $\alpha_t$ is chosen to be $\sqrt{1-\sigma_t^2}$ so that the process is variance preserving. Here I chose $\alpha_t=1-\sigma_t$ so that we 
linearly interpolate between the image and random noise. Why? Honestly I just wondered if it was going to work :) also it simplifies the updating equation quite a bit and it's easier to understand what the noise to signal ratio will look like. The updating equation above is the DDIM model for this parametrization which simplifies to a simple weighted average. Note that the DDIM model deterministically maps random normal noise to images - this has two benefits: we can interpolate in the random normal latent space, it takes fewer steps generaly to get decent image quality.



Note that I use a lot of unorthodox choices in the modelling. Since I am fairly new to generative models I found this to be a 
great way to learn what is crucial vs. what is nice to have. I generally did not see any divergences in training which supports
the notion that **diffusion models are stable to train and are fairly robust to model choices**. The flipside of 
this is that if you introduce subtle bugs in your code (of which I am sure there are many in this repo) they are pretty hard
to spot. 


Architecture: TODO - add cross-attention description.


## Data

The text-to-img models use the [Laion 6.5+ ](https://laion.ai/blog/laion-aesthetics/) datasets. You can see
some samples [here](http://captions.christoph-schuhmann.de/2B-en-6.5.html). As you can 
see this dataset is _very_ biased towards landscapes and portraits. Accordingly, the model
does best at prompts related to art/landscapes/paintings/portraits/architecture.

The script `img_load_utils.py` contains some code to use the img2dataset package to 
download and store images, texts, and their corresponding embeddings. The Laion datasets are still
quite messy with a lot of duplicates, bad descriptions etc.


#### 115k Laion Sample: I have uploaded a 115k 64x64 pixels sample from the Laion 6.5+ dataset+their corresponding prompts and CLIP embeddings to huggingface [here](https://huggingface.co/apapiu/diffusion_model_aesthetic_keras/tree/main). 
This can be used to quickly prototype new generative models. This
dataset is also used in the notebook above.

TODO: add more info and script on how to preprocess the data and link to huggingface repo.
Talk about data quality issues.  


## Training Process and Colab Hints:

If you want to train the img-to-text model I highly recommend getting at least the Colab Pro or even the Colab
Pro+ - it's going to be hard to train the model on a K80 GPU, unfortunately. NOTE: Colab will
change its setup and introduce credits at the end of September - I will update this. 

Setting this training workflow on Google Colab wasn't too bad. My approach has been 
very low tech and Google Drive played a large role. Basically at the end of every epoch I save the model and the 
generated images on a small validation set (50-100 images) to Drive.

This has a few advantages:
- If I get kicked off my colab instance the model is not lost and I just need to restart the instance
- I can keep record of image generation quality by epoch and go back to check and compare models
  - Important point here - make sure to use the _same_ random seed for every epoch - this controls for the randomness  
- Drive saves the past 100 versions of a file so I can always use past model version within the past 100 epochs.
  - This is important since there is some variability in image quality from epoch to epoch
- It's low tech and you don't have to learn a new platform like wandb.
- Reading and saving data in from Drive in Colab is _very_ fast.

I have slowly moved some data/models on huggingface but this is WIP.

#### GPU Speed:
In terms of speed the GPUs go as follows:
`A100>V100>P100>T4>K80` with the A100 being the fastest and every subsequent GPU being roughly twice as slow as 
the one before it for training (e.g. P100 is about 4x slower than A100). While I did get the A100 a few times 
the sweet spot was really V100/P100 on Colab Pro+ since the risk of being time-outed decreased. With colab PRO+ ($50/month) I managed to train on V100/P100 continuously for 12-24 hours at a time.
  

 
### Validation:

I'm not an expert here but generally the validation of generative models
is still an open question. There are metrics like Inception Score, FID, and KID that measure
whether the distribution of generated images is "close" to the training distribution in some way.
The main issue with all of these metrics however is that a model that simply memorizes the training data
will have a perfect score - so they don't account for overfitting. They are also fairly hard to understand, need large 
sample sizes, and are computationally intensive. For all these reasons I have chosen not to use them for now. 

Instead I have focused on analyze the visual quality of generated images by uhm.. looking at them. This can quickly
devolved into a tea-lead reading exercise however. To combat this one come up with different strategies to test for 
quality and diversity. For example sampling from both generated and ground truth images and looking at them together
- either side by side or permuted is a reasonable way to check test for sample quality. 

To test for generalization I have mostly focused on interpolations in both the CLIP space and the 
random normal latent space. Ideally as you move from embedding to embedding you want to generated images
along the path to be meaningful in some way.

CLIP interpolation: "A lake in the forest in the summer" -> "A lake in the forest in the winter"

<img width="400" alt="image" src="https://user-images.githubusercontent.com/13619417/191169640-a8eb9a6f-7808-447a-af5a-094fcc8450ae.png">


Does the model memorize the training data? This is an important question that has 
lots of implications. First of all the models above don't have the capacity to memorize _all_ 
of the training data. For example: the model is about 150 MB but is trained on about
8GB of data. Second of all it might not be in the model's best interest to memorize things. 
After digging a bit around the predictions on the training data I did find _one_ example where
the model shamelessly copies a training example. Note this is because the image appears many times
in the training data.



### Examples:

Prompt: `An Italian Villaga Painted by Picasso`

<img width="650" alt="image" src="https://user-images.githubusercontent.com/13619417/192023316-b11a7a17-2359-4dc0-b727-c51bca167257.png">

`City at night`

<img width="650" alt="image" src="https://user-images.githubusercontent.com/13619417/192022599-0f971f63-f124-4964-8e87-6cba51cf05bb.png">

`Photograph of young woman in a field of flowers, bokeh`

<img width="650" alt="image" src="https://user-images.githubusercontent.com/13619417/192019522-b6f9231d-3e60-472d-b1b8-c43e05310de7.png">

`Street on an island in Greece`

<img width="650" alt="image" src="https://user-images.githubusercontent.com/13619417/192021896-596f35db-5131-4da8-9256-c26e9fa1594d.png">

`A Mountain Lake in the spring at sunset`

<img width="650" alt="image" src="https://user-images.githubusercontent.com/13619417/192023995-b102e30c-2e2f-499a-b5e0-0644aedcbf5c.png">

`A man in a suit in the field in wintertime`

<img width="650" alt="image" src="https://user-images.githubusercontent.com/13619417/192016937-44544116-f27d-43af-a6ce-86506bb44346.png">


CLIP interpolation: "A minimalist living room" -> "A Field in springtime, painting"

<img width="650" alt="image" src="https://user-images.githubusercontent.com/13619417/191169543-6d940748-495b-429f-96a1-e10e1da6bf89.png">

CLIP interpolation: "A lake in the forest in the summer" -> "A lake in the forest in the winter"

<img width="650" alt="image" src="https://user-images.githubusercontent.com/13619417/191169640-a8eb9a6f-7808-447a-af5a-094fcc8450ae.png">


