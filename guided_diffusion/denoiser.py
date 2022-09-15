import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

#######
# MODEL:
#######

# https://github.com/taki0112/Self-Attention-GAN-Tensorflow
# also from stable diffusion:
def attention(qkv):
    q, k, v = qkv
    # f  [bs, h*w, c']
    # g  [bs, h*w, c']
    # h  [bs, h*w, c]

    # should we scale this?
    s = tf.matmul(k, q, transpose_b=True)  # # [bs, h*w, h*w]
    beta = tf.nn.softmax(s)  # attention map
    o = tf.matmul(beta, v)  # [bs, h*w, C]

    return o


def spatial_attention(img):
    filters = img.shape[3]
    orig_shape = ((img.shape[1], img.shape[2], img.shape[3]))
    print(orig_shape)
    img = layers.BatchNormalization()(img)

    # q/k/v could be from different

    # projections:
    q = layers.Conv2D(filters // 8, kernel_size=1, padding="same")(img)
    k = layers.Conv2D(filters // 8, kernel_size=1, padding="same")(img)
    v = layers.Conv2D(filters, kernel_size=1, padding="same")(img)
    k = layers.Reshape((k.shape[1] * k.shape[2], k.shape[3],))(k)

    q = layers.Reshape((q.shape[1] * q.shape[2], q.shape[3]))(q)
    v = layers.Reshape((v.shape[1] * v.shape[2], v.shape[3],))(v)

    # should we scale this?
    img = layers.Lambda(attention)([q, k, v])
    img = layers.Reshape(orig_shape)(img)

    # out_projection:
    img = layers.Conv2D(filters, kernel_size=1, padding="same")(img)
    img = layers.BatchNormalization()(img)

    return img


def cross_attention(img, text):
    filters = img.shape[3]
    orig_shape = ((img.shape[1], img.shape[2], img.shape[3]))
    print(orig_shape)
    img = layers.BatchNormalization()(img)
    text = layers.BatchNormalization()(text)

    # projections:
    q = layers.Conv2D(filters // 8, kernel_size=1, padding="same")(text)
    k = layers.Conv2D(filters // 8, kernel_size=1, padding="same")(img)
    v = layers.Conv2D(filters, kernel_size=1, padding="same")(text)

    q = layers.Reshape((q.shape[1] * q.shape[2], q.shape[3]))(q)
    k = layers.Reshape((k.shape[1] * k.shape[2], k.shape[3],))(k)
    v = layers.Reshape((v.shape[1] * v.shape[2], v.shape[3],))(v)

    # should we scale this?
    img = layers.Lambda(attention)([q, k, v])
    img = layers.Reshape(orig_shape)(img)

    # out_projection:
    img = layers.Conv2D(filters, kernel_size=1, padding="same")(img)
    img = layers.BatchNormalization()(img)

    return img


def sinusoidal_embedding(x):
    #TODO: remove the hardcoded values here:
    embedding_min_frequency = 1.0
    embedding_max_frequency = 1000.0
    embedding_dims = 32
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=keras.activations.swish
        )(x)
        # intermediate layer ... add mlp embedding of class plus timestamp here?
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth, use_self_attention=False):
    def apply(x):
        x, skips, emb_and_noise = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)

            if use_self_attention:
                o = spatial_attention(x)
                x = layers.Add()([x, o])
                cross_att = cross_attention(x, emb_and_noise)
                x = layers.Add()([x, cross_att])

            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth, use_self_attention=False):
    def apply(x):
        x, skips, emb_and_noise = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)

            if use_self_attention:
                o = spatial_attention(x)
                x = layers.Add()([x, o])
                cross_att = cross_attention(x, emb_and_noise)
                x = layers.Add()([x, cross_att])

        return x

    return apply


def get_network(image_size, widths, block_depth, num_classes,
                num_channels, emb_size, attention_levels,
                precomputed_embedding=True):
    noisy_images = keras.Input(shape=(image_size, image_size, num_channels))

    noise_variances = keras.Input(shape=(1, 1, 1))
    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    if precomputed_embedding:
        input_label = layers.Input(shape=emb_size)  # CLIP/glove embedding.
        emb_label = layers.Dense(emb_size // 2)(input_label)
        emb_label = layers.Reshape((1, 1, emb_size // 2))(emb_label)
    else:
        input_label = layers.Input(shape=1)  # label/word - integer encoded
        emb_label = layers.Embedding(input_dim=num_classes, output_dim=emb_size)(input_label)
        emb_label = layers.Reshape((1, 1, emb_size))(emb_label)

    emb_label = layers.UpSampling2D(size=image_size, interpolation="nearest")(emb_label)

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e, emb_label])

    emb_and_noise = layers.Concatenate()([emb_label, e])
    emb_and_noise = layers.BatchNormalization()(emb_and_noise)

    skips = []
    level = 0
    for width in widths[:-1]:
        use_self_attention = bool(attention_levels[level])
        x = DownBlock(width, block_depth, use_self_attention)([x, skips, emb_and_noise])

        emb_and_noise = layers.AveragePooling2D()(emb_and_noise)
        level += 1

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)
        if bool(attention_levels[level]):
            o = spatial_attention(x)
            x = layers.Add()([x, o])
            cross_att = cross_attention(x, emb_and_noise)
            x = layers.Add()([x, cross_att])

    for width in reversed(widths[:-1]):
        level -= 1

        emb_and_noise = layers.UpSampling2D(size=2, interpolation="bilinear")(emb_and_noise)
        use_self_attention = bool(attention_levels[level])
        x = UpBlock(width, block_depth, use_self_attention)([x, skips, emb_and_noise])

    x = layers.Conv2D(num_channels, kernel_size=1, kernel_initializer="zeros",
                      activation="linear"
                      )(x)

    return Model([noisy_images, noise_variances, input_label], x, name="residual_unet")