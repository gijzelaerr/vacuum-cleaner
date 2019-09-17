import tensorflow as tf

tf.enable_eager_execution()

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from io import BytesIO


def fits_open(path, IMG_SIZE):
    content = fits.open(str(path))[0].data.squeeze().astype(np.float32)
    if IMG_SIZE == 128:
        return content[IMG_SIZE // 2:-(IMG_SIZE // 2), IMG_SIZE // 2:-(IMG_SIZE // 2)]
    else:
        return content


def load_fits(fits_file):
    def internal(data):
        return fits.open(BytesIO(data))[0].data.squeeze().astype(np.float32)[..., np.newaxis]

    blob = tf.io.read_file(fits_file)
    return tf.py_func(internal, [blob], tf.float32)


def make_dataset(glob):
    ds = tf.data.Dataset.list_files(str(glob), shuffle=False)
    ds = ds.map(load_fits)
    return ds


def normalize(first, *others):
    """accepts a list of images, normalized to [-1, 1] relative to the first image"""
    min_ = tf.reduce_min(first)
    max_ = tf.reduce_max(first)
    f = lambda i: ((i - min_) / ((max_ - min_) / 2)) - 1
    return [min_, max_, f(first)] + list(map(f, others))


def denormalize(images, min_, max_):
    """scales image back to min_, max_ range"""
    return [((i + 1) / 2 * (max_ - min_)) + min_ for i in images]


def random_jitter(*images):
    if (tf.random.uniform(shape=()) > tf.to_float(0.5)) is not None:
        return [tf.image.flip_left_right(i) for i in images]
    else:
        return list(images)


def downsample(filters, size, apply_batchnorm=True):
    type_ = tf.keras.layers.Conv2D
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(type_(filters,
                     size,
                     strides=2,
                     padding='same',
                     kernel_initializer=initializer,
                     use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def Generator(IMG_SIZE, OUTPUT_CHANNELS):
    if IMG_SIZE == 256:
        down_stack_start = [
            downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
            downsample(128, 4),  # (bs, 64, 64, 128)
        ]
    elif IMG_SIZE == 128:
        down_stack_start = [
            downsample(128, 4, apply_batchnorm=False),  # (bs, 64, 64, 128)
        ]
    else:
        raise Exception

    down_stack = down_stack_start + [
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
    ]

    if IMG_SIZE == 256:
        up_stack.append(upsample(64, 4))  # (bs, 128, 128, 128)
    elif IMG_SIZE == 128:
        # do nothing
        ...
    else:
        raise Exception

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, IMG_SIZE, IMG_SIZE, 1)

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[None, None, 1])
    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, 1], name='input_image')
    tar = tf.keras.layers.Input(shape=[None, None, 1], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def discriminator_loss(loss_object, disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


def generator_loss(loss_object, disc_generated_output, gen_output, target, LAMBDA):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss


def convolve(convolved, convolver):
    if convolved.shape != convolver.shape:
        # probably convolving with a big psf
        convolver = convolver[:, 128:-128, 128:-128, :]
    kernel = tf.squeeze(convolver)[:, :, tf.newaxis, tf.newaxis]  # [height, width, in_channels, out_channels]
    return tf.nn.conv2d(convolved, filter=kernel, strides=[1, 1, 1, 1], padding="SAME")


def likelihood_loss(predicted, target, psf):
    convolved = convolve(predicted, psf)
    # return tf.reduce_sum(tf.multiply(predicted, convolved - 2 * target))
    return tf.reduce_sum(tf.tensordot(tf.squeeze(predicted), (tf.squeeze(convolved) - 2 * tf.squeeze(target)), axes=1))


def clean_loss(predicted, target, clean_beam):
    return l1(convolve(predicted, clean_beam), convolve(target, clean_beam))


def render(f, a, imgdata, title):
    i = a.pcolor(imgdata, cmap='cubehelix')
    f.colorbar(i, ax=a)
    a.set_title(title)


def generate_images(prediction, input_, target):
    f, ((a1, a2, a3)) = plt.subplots(1, 3, figsize=(15, 3))
    render(f, a1, tf.squeeze(prediction), 'prediction')
    render(f, a2, tf.squeeze(input_), 'input_')
    render(f, a3, tf.squeeze(target), 'target')
    plt.show()


def generate_plot(range_, title):
    fig, ax = plt.subplots()
    ax.plot(range_)
    ax.set(title=title)
    plt.show()


def train_step(loss_object, generator, generator_optimizer, discriminator_optimizer, discriminator, input_image, target, LAMBDA):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        predicted = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, predicted], training=True)


        gen_loss = generator_loss(loss_object, disc_generated_output, predicted, target, LAMBDA)
        disc_loss = discriminator_loss(loss_object, disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))


def train_step_likelihood(generator, generator_optimizer, input_image, target, bigpsf):
    with tf.GradientTape() as gen_tape:
        predicted = generator(input_image, training=True)
        likelyhood_loss = likelihood_loss(predicted, target, bigpsf) / 2e8 + 1
        l1_loss = tf.reduce_mean(tf.abs(target - predicted))
        total_loss = likelyhood_loss + l1_loss  # * LAMBDA

    print(float(likelyhood_loss), float(l1_loss))

    generator_gradients = gen_tape.gradient(total_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))


def l1(a, b):
    return tf.math.reduce_sum(tf.math.abs(a - b))

