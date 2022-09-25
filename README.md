# Text to Image Model in Keras

Codebase to train a CLIP conditioned Text to Image Diffusion model on Colab in Keras. See below for notebooks and examples with prompts.

Images generated for the prompt: `A small village in the Alps, spring, sunset` 

<img width="500" alt="image" src="https://user-images.githubusercontent.com/13619417/191170681-17c3820b-7fe1-44ad-bb51-30a521d465f7.png">

Images generated for the prompt: `Portrait of a young woman with curly red hair, photograph` 

<img width="500" alt="image" src="https://user-images.githubusercontent.com/13619417/192167167-5b308069-4483-451e-8aef-0a8dc1d1c10f.png">


(more exampes below - try with your own inputs in Colab here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/123iljowP_b5o-_6RjHZK8vbgBkrNn8-c?usp=sharing) )

The goal of this repo is to provide a simple, self-contained codebase for Text to Image Diffusion that can be trained in Colab in a 
reasonable amount of time. 

While there are a lot of great resources around the math and usage of diffusion models I haven't found many specifically
focused on _training_ text to img diffusion models. 
Particulary the idea of _training_ a Dalle-2 or Stable Diffusion like model feels like a daunting task requiring immense 
computational resources and data. Turns out you can accomplish quite a lot with little resources and without having a PhD in thermodynamics!
The easiest way to get aquainted with the code is thru the notebooks below. 

#### Credits:

- Original Unet implementation in this [excellent blog post](https://keras.io/examples/generative/ddim/) - most of the code and Unet architecture in `denoiser.py` is based on this. I have added 
additional text/CLIP/masking embeddings/inputs and cross/self attention.
- https://colab.research.google.com/drive/1IJkrrV-D7boSCLVKhi7t5docRYqORtm3#scrollTo=TAUwPLG92r89
- Laion Aesthetics 6.5+
- Text 2 img package

### Notebooks:

If you are just starting out I recommend trying out the next two notebook in order. The first should be able to get you 
recognizable images on the Fashion Mnist dataset within minutes!

- Train Class Conditional Fashion MNIST: 
  - To try out the code you can use the notebook above in colab. It is set to train on the fashion mnist dataset.
You should be able to see reasonable image generations withing 5 epochs (5-20 minutes depending on GPU)!
  - For CIFAR 10/100 - you just have to change the `file_name`. You can get reasonable results after 25 epochs for CIFAR 10 and 40 epochs for CIFAR 100.
Training 50-100 epochs is even better. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16rJUyPn72-C30mZRUr-Oo6ZjYS89Z3yH?usp=sharing)

- Train CLIP Conditioned Text to Img Model on 115k 64x64 images+prompts sampled from the Laion Aesthetics 6.5+ dataset. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EoGdyZTGVeOrEnieWyzjItusBSes_1Ef?usp=sharing) 
  - You can get reasonable results after 15 epochs
  ~ 10 minutes per epoch (V100)
- Test Prompts on a model trained for about 60 epochs (~60 hours on 1 V100) on entire 600k Laion Aesthetics 6.5+. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/123iljowP_b5o-_6RjHZK8vbgBkrNn8-c?usp=sharing) 
  - This model has about 40 million parameters (150MB) 

### Model Setup:

The setup is fairly simple. 

We train a denoising U-NET neural net that takes the following three inputs:
- image (x) corrupted with a level of random noise
  - for a give `noise_level` between 0 and 1 the corruption is as follows:
    - `x_noisy = x*(1-noise_level) + z*noise_level where z ~ np.random.normal`
- noise_level (sampled from 0 to 1 with more values concentrated close to 0)
- CLIP embeddings of a text prompt
  - You can think of this as a numerical represenation of a prompt (this is the only pretrained model we use). 

and outputs a prediction of the denoised image - call it `f(x_noisy)`.

The model is trained to minimize the absolute error `|f(x_noisy) - x|` between the prediction and 
(you can also use squared error here). 

Using this model we then iteratively generate an image from random noise as follows:
    
         for i in range(len(self.noise_levels) - 1):

            curr_noise, next_noise = self.noise_levels[i], self.noise_levels[i + 1]

            # predict original denoised image:
            x0_pred = self.predict_x_zero(new_img, label, curr_noise)

            # new image at next_noise level is a weighted average of old image and predicted x0:
            new_img = ((curr_noise - next_noise) * x0_pred + next_noise * new_img) / curr_noise

There's some math behind this. DDIM like. TODO: explain more here.

Note that I use a lot of unorthodox choices in the model and in the diffusion process (linear weights for the noise, 
weird noise sampling, not scaling the attention). Since I am fairly new to this I found this to be a  
great way to learn what is crucial vs. what is nice to have. I generally obtained decent results
so I think the main lesson here is that **diffusion models are stable to train and are fairly robust to model choices**. Even tough I had no experience with generative 
models before this project I almost never encountered a divergence in training - so go ahead and experiment :) The flipside of this
is that if you introduce subtle bugs in your code (of which I am sure there are many in this repo) they are pretty hard
to spot. This is exacerbated by the fact that there are a lot of residual/skip connections in the model so the 
information will find a way to flow even if you block some pathways.


Architecture:
models train well 
### Validation:

I'm not an expert here but generally the validation of generative models
is still an open question. There are metrics like Inception Score, FID, and KID that measure
whether the distribution of generated images is "close" to the training distribution in some way.
The main issue with all of these metrics however is that a model that simply memorizes the training data
will have a perfect score - so they don't account for overfitting. They are also fairly hard to understand, need large 
sample sizes, and are computationally intensive. For all these reasons I have chosen not to use them. Instead 
I just look at the images generated by the model and test for:
- quality, diversity, generalization (see 20.4 in Murphy Book)

Quality and Diversity can be tested by looking at the generated images next to ground truth:

- Next to each other
- Permutation tests

However again a model that memorizes (some) of the training data can do well on Quality and Diversity tests above.

Generalization:
A good way to test for generaliztion - which is what we want really is to interpolate between different seeds and also between different CLIP embeddings.
You want a few things:

1. Smooth Transitions - this means the model has learnt to interpolate between images/embeddings


Does the model memorize the training data? This is an important question that has 
lots of implications. First of all the models above don't have the capacity to memorize _all_ 
of the training data. For example: the model is about 150 MB but is trained on about
8GB of data. Second of all it might not be in the model's best interest to memorize things. 
After digging a bit around the predictions on the training data I did find _one_ examaple where
the model shamelessly copies a training example. Note this is because the image appears many times
in the training data.


### Data:

The text-to-img models use the [Laion 6.5+ ](https://laion.ai/blog/laion-aesthetics/) datasets. You can see
some samples [here](http://captions.christoph-schuhmann.de/2B-en-6.5.html). As you can 
see this dataset is _very_ biased towards landscapes and portraits. Accordingly, the model
does best at prompts related to art/landscapes/paintings/portraits/architecture.

TODO: add more info and script on how to preprocess the data and link to huggingface repo.
Talk about data quality issues.  


### Training Process and Colab Hints:

If you want to train the img-to-text model I highly recommend getting at least the Colab Pro or even the Colab
Pro+ - it's going to be hard to train the model on a K80 GPU, unfortunately.

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

I managed to train on V100/P100 contiously for 12-24 hours at a time. 

GPUs:
I recommend V100/P100 for this tak. You might get A100 but you won't be able to keep an instance
on with a A100 for too long. Generally the speed of training is as follows:
- A100
- V100
- P1000
- T4
- K80

Every card is roughly about twice as fast as the one before it so 




### Examples:

Prompt: `An Italian Villaga Painted by Picasso`
<img width="750" alt="image" src="https://user-images.githubusercontent.com/13619417/192023316-b11a7a17-2359-4dc0-b727-c51bca167257.png">

`City at night`
<img width="750" alt="image" src="https://user-images.githubusercontent.com/13619417/192022599-0f971f63-f124-4964-8e87-6cba51cf05bb.png">

`Photograph of young woman in a field of flowers, bokeh`
<img width="750" alt="image" src="https://user-images.githubusercontent.com/13619417/192019522-b6f9231d-3e60-472d-b1b8-c43e05310de7.png">

`Street on an island in Greece`
<img width="750" alt="image" src="https://user-images.githubusercontent.com/13619417/192021896-596f35db-5131-4da8-9256-c26e9fa1594d.png">

`A Mountain Lake in the spring at sunset`
<img width="750" alt="image" src="https://user-images.githubusercontent.com/13619417/192023995-b102e30c-2e2f-499a-b5e0-0644aedcbf5c.png">

`A man in a suit in the field in wintertime`
<img width="750" alt="image" src="https://user-images.githubusercontent.com/13619417/192016937-44544116-f27d-43af-a6ce-86506bb44346.png">


CLIP interpolation: "A minimalist living room" -> "A Field in springtime, painting"

<img width="799" alt="image" src="https://user-images.githubusercontent.com/13619417/191169543-6d940748-495b-429f-96a1-e10e1da6bf89.png">

CLIP interpolation: "A lake in the forest in the summer" -> "A lake in the forest in the winter"

<img width="798" alt="image" src="https://user-images.githubusercontent.com/13619417/191169640-a8eb9a6f-7808-447a-af5a-094fcc8450ae.png">


