import os
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from tensorflow import keras
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from denoiser import get_network
from utils import batch_generator
from diffuser import Diffuser

#########
# CONFIG:
#########

image_size = 32
num_channels = 3

# architecture
embedding_dims = 32
embedding_max_frequency = 1000.0
channels = 32
channel_multiplier = [1, 2, 3,
                      # 4
                      ]
block_depth = 1
emb_size = 32  # CLIP/label embedding
num_classes = 110  # placeholder
attention_levels = [0, 0, 0
                    ]

precomputed_embedding = False

widths = [c * channels for c in channel_multiplier]

###train process config:
batch_size = 32
num_imgs = 64
learning_rate = 0.0003
model_name = "model_test"
from_scratch = True
file_name = "aesthetic_0.npz"

home_dir = "{model_name}".format(model_name=model_name)
if not os.path.exists(home_dir):
    os.mkdir(home_dir)

model_path = os.path.join(home_dir, model_name + ".h5")

(train_data, train_label_embeddings), (_, _) = cifar100.load_data()
train_label_embeddings += 1

# train_data, train_label_embeddings = get_data(npz_file_name=file_name, prop=0.6, captions=False)
# train_data, train_label_embeddings, caption_list = get_data(npz_file_name=file_name, prop=0.6)

print(train_data.shape)

if precomputed_embedding:
    labels_ohe = train_label_embeddings[:64]
else:
    labels_ohe = np.array([[i] * 8 for i in np.arange(8)]).flatten()[:, None]

np.random.seed(100)
rand_image = np.random.normal(0, 1, (64, image_size, image_size, num_channels))

if from_scratch:
    autoencoder = get_network(image_size,
                              widths,
                              block_depth,
                              num_classes=num_classes,
                              emb_size=emb_size,
                              attention_levels=attention_levels,
                              num_channels=num_channels,
                              precomputed_embedding=precomputed_embedding)

    autoencoder.compile(optimizer="adam", loss="mae")
else:
    autoencoder = keras.models.load_model(model_path)

from keras.utils.vis_utils import plot_model

plot_model(autoencoder, to_file=os.path.join(home_dir, 'model_plot.png'),
           show_shapes=True, show_layer_names=True)

print(autoencoder.count_params())
#
# pd.Series(train_data[:1000].ravel()).hist(bins=80)
# plt.show()
# #print(caption_list[:64])
#
# plt.rcParams["figure.figsize"] = (14, 14)
# imgs = train_data[:100]
# nrows = 10
# for i in range(len(imgs)):
#     ax = plt.subplot(nrows,nrows, i+1)
#     plt.imshow(imgs[i])
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
# plt.show()

train_generator = batch_generator(autoencoder,
                                  model_name,
                                  train_data[:1000],
                                  train_label_embeddings[:1000],
                                  batch_size,
                                  rand_image,
                                  labels_ohe,
                                  home_dir)

autoencoder.optimizer.learning_rate.assign(learning_rate)

eval_nums = autoencoder.fit(
    x=train_generator,
    epochs=1,
)
