import os
import yaml
import numpy as np
import pandas as pd

from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from keras.utils import plot_model
from tensorflow import keras
from matplotlib import pyplot as plt

from denoiser import get_network
from utils import batch_generator, plot_images, get_data, preprocess
from diffuser import Diffuser


class Trainer:
    def __init__(self, config_file):
        # Load YAML config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Unpack config into instance attributes
        for key, value in config.items():
            setattr(self, key, value)

        # Additional setups
        self._setup_paths_and_dirs()

    def _setup_paths_and_dirs(self):
        self.widths = [c * self.channels for c in self.channel_multiplier]
        self.captions_path = os.path.join(self.data_dir, "captions.csv")
        self.imgs_path = os.path.join(self.data_dir, "imgs.npy")
        self.embedding_path = os.path.join(self.data_dir, "embeddings.npy")

        if self.save_in_drive:
            from google.colab import drive
            drive.mount('/content/drive')
            drive_path = '/content/drive/MyDrive/'
            self.home_dir = os.path.join(drive_path, self.MODEL_NAME)
        else:
            self.home_dir = self.MODEL_NAME

        if not os.path.exists(self.home_dir):
            os.mkdir(self.home_dir)

        self.model_path = os.path.join(self.home_dir, self.MODEL_NAME + ".h5")

    def preprocess_data(self, train_data, train_label_embeddings):
        print(train_data.shape)
        self.train_data = train_data
        self.train_label_embeddings = train_label_embeddings
        self.image_size = train_data.shape[1]
        self.num_channels = train_data.shape[-1]
        self.row = int(np.sqrt(self.num_imgs))
        self.labels = self._get_labels(train_data, train_label_embeddings)

    def _get_labels(self, train_data, train_label_embeddings):
        if self.precomputed_embedding:
            return train_label_embeddings[:self.num_imgs]
        else:
            row_labels = np.array([[i] * self.row for i in np.arange(self.row)]).flatten()[:, None]
            return row_labels + 1

    def initialize_model(self):

        if self.from_scratch:
            self.autoencoder = get_network(self.image_size,
                                           self.widths,
                                           self.block_depth,
                                           num_classes=self.num_classes,
                                           attention_levels=self.attention_levels,
                                           emb_size=self.emb_size,
                                           num_channels=self.num_channels,
                                           precomputed_embedding=self.precomputed_embedding)

            self.autoencoder.compile(optimizer="adam", loss="mae")
        else:
            self.autoencoder = keras.models.load_model(self.model_path)

    def data_checks(self, train_data):
        print("Number of parameters is {0}".format(self.autoencoder.count_params()))
        pd.Series(train_data[:1000].ravel()).hist(bins=80)
        plt.show()
        print("Original Images:")
        plot_images(preprocess(train_data[:self.num_imgs]), nrows=int(np.sqrt(self.num_imgs)))
        plot_model(self.autoencoder, to_file=os.path.join(self.home_dir, 'model_plot.png'),
                   show_shapes=True, show_layer_names=True)
        print("Generating Images below:")

    def train(self):
        np.random.seed(100)
        self.rand_image = np.random.normal(0, 1, (self.num_imgs, self.image_size, self.image_size, self.num_channels))

        self.diffuser = Diffuser(self.autoencoder,
                                 class_guidance=self.class_guidance,
                                 diffusion_steps=35)

        if self.train_model:
            train_generator = batch_generator(self.autoencoder,
                                              self.model_path,
                                              self.train_data,
                                              self.train_label_embeddings,
                                              self.epochs,
                                              self.batch_size,
                                              self.rand_image,
                                              self.labels,
                                              self.home_dir,
                                              self.diffuser)

            self.autoencoder.optimizer.learning_rate.assign(self.learning_rate)

            self.eval_nums = self.autoencoder.fit(
                x=train_generator,
                epochs=self.epochs
            )


def get_train_data(file_name, captions_path=None, imgs_path=None, embedding_path=None):
    dataset_loaders = {
        "cifar10": cifar10.load_data,
        "cifar100": cifar100.load_data,
        "fashion_mnist": fashion_mnist.load_data,
        "mnist": mnist.load_data
    }

    if file_name in dataset_loaders:
        (train_data, train_label_embeddings), (_, _) = dataset_loaders[file_name]()

        # Add unconditional embedding
        train_label_embeddings = train_label_embeddings + 1

        if file_name in ["fashion_mnist", "mnist"]:
            train_data = train_data[:, :, :, None]
            train_label_embeddings = train_label_embeddings[:, None]

    else:
        captions = pd.read_csv(captions_path)
        train_data, train_label_embeddings = np.load(imgs_path), np.load(embedding_path)

    return train_data, train_label_embeddings
    #train_data, train_label_embeddings = get_data(npz_file_name=file_name, prop=0.6, captions=False)


if __name__=='__main__':

    trainer = Trainer('guided_diffusion/configs/base_model.yaml')

    train_data, train_label_embeddings = get_train_data(trainer.file_name)
    trainer.preprocess_data(train_data, train_label_embeddings)
    trainer.initialize_model()
    trainer.data_checks(train_data)
    trainer.train()
