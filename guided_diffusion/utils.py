import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from numpy.linalg import norm
from diffuser import Diffuser


def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    array = array.astype("float32") / 255.0
    array = array * 2 - 1

    return array


#######
# TRAIN UTILS:
#######

def add_noise(array, mu=0, std=1):
    # THIS IS IMPORTANT:
    x = np.abs(np.random.normal(0, std, 2 * len(array)))
    x = x[x < 3]
    x = x / 3
    x = x[:len(array)]
    noise_level_sqrt = x

    signal_level_sqrt = 1 - noise_level_sqrt  # - using linear weight works as well! np.sqrt(1-np.square(noise_level_sqrt))

    # reshape so that the multiplication makes sense:
    noise_level_reshape_sqrt = noise_level_sqrt[:, None, None, None]
    signal_level_reshape_sqrt = signal_level_sqrt[:, None, None, None]

    pure_noise = np.random.normal(0, 1, size=array.shape).astype("float32")

    noisy_data = array * signal_level_reshape_sqrt + np.random.normal(0, 1, size=array.shape) * noise_level_reshape_sqrt

    return noisy_data, noise_level_sqrt


def slerp(p0, p1, t):
    """spherical interpolation"""
    omega = np.arccos(np.dot(p0 / norm(p0), p1 / norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1


def interpolate_two_points(num_points=100):
    pA = np.random.normal(0, 1, (image_size * image_size * num_channels))
    pB = np.random.normal(0, 1, (image_size * image_size * num_channels))

    ps = np.array([slerp(pA, pB, t) for t in np.arange(0.0, 1.0, 1 / num_points)])
    rand_image = ps.reshape(len(ps), image_size, image_size, num_channels)

    return rand_image


def imshow(img):
    def norm_0_1(img):
        return (img + 1) / 2

    # img here is betweeen -1 and 1:
    if img.shape[-1] == 1:
        img = img.reshape(img.shape[0], img.shape[1])
    img = np.clip(img, -1, 1)
    plt.imshow(norm_0_1(img))


def plot_images(imgs, size=16, nrows=8, save_name=None):
    plt.rcParams["figure.figsize"] = (size, size)

    for i in range(len(imgs)):
        ax = plt.subplot(nrows, nrows, i + 1)
        imshow(imgs[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if save_name:
        save_to = "{0}.png".format(save_name)
        plt.savefig(save_to, dpi=200)
    plt.show()


def get_labels(num_classes=100, emb=True):
    labels_ohe = np.zeros(num_imgs)
    for i in range(10):
        labels_ohe[10 * i:10 * (i + 1)] = i

    if emb:
        text_labels = labels[labels_ohe]
        labels_ohe = np.vstack([get_embedding(label) for label in text_labels.values])

    return labels_ohe


def get_data(npz_file_name="ccm100_1k.npz", prop=0.6, captions=True):
    data = np.load("/content/drive/MyDrive/{0}".format(npz_file_name))

    if captions:
        train_data, train_label_embeddings, caption_list = data["arr_0"], data["arr_1"], data["arr_2"]
    else:
        train_data, train_label_embeddings = data["arr_0"], data["arr_1"]

    # eliminate if perc white pixels > 60%.
    white_pixels = (train_data >= 254).mean(axis=(1, 2, 3))
    mask = (white_pixels < 0.6)

    if captions:
        train_data, train_label_embeddings, caption_list = train_data[mask], train_label_embeddings[mask], caption_list[
            mask]
        return train_data, train_label_embeddings, caption_list
    else:
        train_data, train_label_embeddings = train_data[mask], train_label_embeddings[mask]
        return train_data, train_label_embeddings


def batch_generator(model, model_path, train_data, train_label_embeddings, epochs,
                    batch_size, rand_image, labels_ohe, home_dir, class_guidance=3):
    indices = np.arange(len(train_data))
    batch = []
    epoch = 0
    print("Training for {0}".format(epochs))
    while epoch < epochs:
        print("saving model:")
        model.save(model_path)

        print(" Generating images:")
        big_diffuser = Diffuser(model,
                                class_guidance=class_guidance,
                                diffusion_steps=35)

        imgs = big_diffuser.reverse_diffusion(rand_image, labels_ohe)
        img_path = os.path.join(home_dir, str(epoch))
        plot_images(imgs, save_name=img_path, nrows=int(np.sqrt(len(imgs))))

        print("new epoch {0}".format(epoch))
        # it might be a good idea to shuffle your data before each epoch
        np.random.shuffle(indices)
        for i in indices:
            batch.append(i)
            if len(batch) == batch_size:
                tr_batch = train_data[batch].copy()
                tr_batch = preprocess(tr_batch)

                # random dropout for CFG:
                s = np.random.binomial(1, 0.15, size=batch_size).astype("bool")
                train_label_dropout = train_label_embeddings[batch].copy()
                train_label_dropout[s] = np.zeros(shape=train_label_embeddings.shape[1])

                # Add Noise to Images:
                noisy_train_data, noise_level_train = add_noise(tr_batch, mu=0, std=1)
                noise_level_train = noise_level_train[:, None, None, None]  # for correct dim

                yield [noisy_train_data, noise_level_train, train_label_dropout], tr_batch
                batch = []
        epoch += 1


def get_common_words(n):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(min_df=10, stop_words="english")
    text = caption_list
    vectorizer.fit(text)

    word_counts = vectorizer.transform(text)
    word_counts = (word_counts.sum(0))
    word_counts = pd.DataFrame(word_counts.T, index=vectorizer.get_feature_names_out())
    return word_counts.sort_values(0, ascending=False).head(n)

# from google.colab import drive
# drive.mount('/content/drive')