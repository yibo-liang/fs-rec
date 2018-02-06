# This is a fork from Faceswap-GAN, with face recognition added so in the test part we don't swap every face, but only the trained one

from keras.models import Sequential, Model
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.applications import *
import keras.backend as K
from tensorflow.contrib.distributions import Beta
import tensorflow as tf
from keras.optimizers import Adam

# In[2]:


from image_augmentation import random_transform
from image_augmentation import random_warp, random_warp128
from utils import get_image_paths, load_images, stack_images
from pixel_shuffler import PixelShuffler
from instance_normalization import InstanceNormalization

# In[3]:


import time
import numpy as np
from PIL import Image
import cv2
import glob
from random import randint, shuffle
from IPython.display import clear_output
from IPython.display import display
import matplotlib.pyplot as plt

from keras_vggface.vggface import VGGFace

vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))

K.set_learning_phase(1)

channel_axis = -1
channel_first = False

IMAGE_SHAPE = (128, 128, 3)
nc_in = 3  # number of input channels of generators
nc_D_inp = 6  # number of input channels of discriminators

use_perceptual_loss = True
use_lsgan = True
use_instancenorm = False
use_mixup = True
mixup_alpha = 0.2  # 0.2

batchSize = 256
lrD = 1e-4  # Discriminator learning rate
lrG = 1e-4  # Generator learning rate


# <a id='5'></a>
# # 5. Define models

# In[9]:


def __conv_init(a):
    print("conv_init", a)
    k = RandomNormal(0, 0.02)(a)  # for convolution kernel
    k.conv_weight = True
    return k


conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)  # for batch normalization


# In[10]:


# def batchnorm():
#    return BatchNormalization(momentum=0.9, axis=channel_axis, epsilon=1.01e-5, gamma_initializer = gamma_init)

def inst_norm():
    return InstanceNormalization()


def conv_block(input_tensor, f, use_instance_norm=True):
    x = input_tensor
    x = SeparableConv2D(f, kernel_size=3, strides=2, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    if use_instance_norm:
        x = inst_norm()(x)
    x = Activation("relu")(x)
    return x


def conv_block_d(input_tensor, f, use_instance_norm=True):
    x = input_tensor
    x = Conv2D(f, kernel_size=4, strides=2, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    if use_instance_norm:
        x = inst_norm()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x


def res_block(input_tensor, f, dilation=1):
    x = input_tensor
    x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, use_bias=False, padding="same", dilation_rate=dilation)(
        x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, use_bias=False, padding="same", dilation_rate=dilation)(
        x)
    x = add([x, input_tensor])
    # x = LeakyReLU(alpha=0.2)(x)
    return x


def upscale_ps(filters, use_instance_norm=True):
    def block(x, use_instance_norm=use_instance_norm):
        x = Conv2D(filters * 4, kernel_size=3, use_bias=False, kernel_initializer=RandomNormal(0, 0.02),
                   padding='same')(x)
        if use_instance_norm:
            x = inst_norm()(x)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        return x

    return block


def Discriminator(nc_in, input_size=128):
    inp = Input(shape=(input_size, input_size, nc_in))
    # x = GaussianNoise(0.05)(inp)
    x = conv_block_d(inp, 64, False)
    x = conv_block_d(x, 128, True)
    x = conv_block_d(x, 256, True)
    x = conv_block_d(x, 512, True)
    out = Conv2D(1, kernel_size=4, kernel_initializer=conv_init, use_bias=False, padding="same", activation="sigmoid")(
        x)
    return Model(inputs=[inp], outputs=out)


def Encoder(nc_in=3, input_size=128):
    inp = Input(shape=(input_size, input_size, nc_in))
    x = Conv2D(32, kernel_size=5, kernel_initializer=conv_init, use_bias=False, padding="same")(inp)
    x = conv_block(x, 64, use_instance_norm=False)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = conv_block(x, 512)
    x = conv_block(x, 1024)
    x = Dense(1024)(Flatten()(x))
    x = Dense(4 * 4 * 1024)(x)
    x = Reshape((4, 4, 1024))(x)
    out = upscale_ps(512)(x)
    return Model(inputs=inp, outputs=out)


def Decoder_ps(nc_in=512, input_size=8):
    input_ = Input(shape=(input_size, input_size, nc_in))
    x = input_
    x = upscale_ps(256)(x)
    x = upscale_ps(128)(x)
    x = upscale_ps(64)(x)
    x = res_block(x, 64, dilation=2)

    out64 = Conv2D(64, kernel_size=3, padding='same')(x)
    out64 = LeakyReLU(alpha=0.1)(out64)
    out64 = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(out64)

    x = upscale_ps(32)(x)
    x = res_block(x, 32)
    x = res_block(x, 32)
    alpha = Conv2D(1, kernel_size=5, padding='same', activation="sigmoid")(x)
    rgb = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x)
    out = concatenate([alpha, rgb])
    return Model(input_, [out, out64])


# In[11]:


# <a id='6'></a>
# # 6. Load Models

# In[88]:




# <a id='7'></a>
# # 7. Define Inputs/Outputs Variables
#
#     distorted_A: A (batch_size, 64, 64, 3) tensor, input of generator_A (netGA).
#     distorted_B: A (batch_size, 64, 64, 3) tensor, input of generator_B (netGB).
#     fake_A: (batch_size, 64, 64, 3) tensor, output of generator_A (netGA).
#     fake_B: (batch_size, 64, 64, 3) tensor, output of generator_B (netGB).
#     mask_A: (batch_size, 64, 64, 1) tensor, mask output of generator_A (netGA).
#     mask_B: (batch_size, 64, 64, 1) tensor, mask output of generator_B (netGB).
#     path_A: A function that takes distorted_A as input and outputs fake_A.
#     path_B: A function that takes distorted_B as input and outputs fake_B.
#     path_mask_A: A function that takes distorted_A as input and outputs mask_A.
#     path_mask_B: A function that takes distorted_B as input and outputs mask_B.
#     path_abgr_A: A function that takes distorted_A as input and outputs concat([mask_A, fake_A]).
#     path_abgr_B: A function that takes distorted_B as input and outputs concat([mask_B, fake_B]).
#     real_A: A (batch_size, 64, 64, 3) tensor, target images for generator_A given input distorted_A.
#     real_B: A (batch_size, 64, 64, 3) tensor, target images for generator_B given input distorted_B.

# In[13]:


def cycle_variables(netG):
    distorted_input = netG.inputs[0]
    fake_output = netG.outputs[0]
    fake_output64 = netG.outputs[1]
    alpha = Lambda(lambda x: x[:, :, :, :1])(fake_output)
    rgb = Lambda(lambda x: x[:, :, :, 1:])(fake_output)

    masked_fake_output = alpha * rgb + (1 - alpha) * distorted_input

    fn_generate = K.function([distorted_input], [masked_fake_output])
    fn_mask = K.function([distorted_input], [concatenate([alpha, alpha, alpha])])
    fn_abgr = K.function([distorted_input], [concatenate([alpha, rgb])])
    fn_bgr = K.function([distorted_input], [rgb])
    return distorted_input, fake_output, fake_output64, alpha, fn_generate, fn_mask, fn_abgr, fn_bgr


# In[14]:


encoder = Encoder()
decoder_A = Decoder_ps()
decoder_B = Decoder_ps()

x = Input(shape=IMAGE_SHAPE)

netGA = Model(x, decoder_A(encoder(x)))
netGB = Model(x, decoder_B(encoder(x)))

# In[12]:


netDA = Discriminator(nc_D_inp)
netDB = Discriminator(nc_D_inp)

distorted_A, fake_A, fake_sz64_A, mask_A, path_A, path_mask_A, path_abgr_A, path_bgr_A = cycle_variables(netGA)
distorted_B, fake_B, fake_sz64_B, mask_B, path_B, path_mask_B, path_abgr_B, path_bgr_B = cycle_variables(netGB)
real_A = Input(shape=IMAGE_SHAPE)
real_B = Input(shape=IMAGE_SHAPE)

# <a id='8'></a>
# # 8. Define Loss Function
#
# LSGAN

# In[15]:


if use_lsgan:
    loss_fn = lambda output, target: K.mean(K.abs(K.square(output - target)))
else:
    loss_fn = lambda output, target: -K.mean(K.log(output + 1e-12) * target + K.log(1 - output + 1e-12) * (1 - target))


# In[16]:


def define_loss(netD, real, fake_argb, fake_sz64, distorted, vggface_feat=None):
    alpha = Lambda(lambda x: x[:, :, :, :1])(fake_argb)
    fake_rgb = Lambda(lambda x: x[:, :, :, 1:])(fake_argb)
    fake = alpha * fake_rgb + (1 - alpha) * distorted

    if use_mixup:
        dist = Beta(mixup_alpha, mixup_alpha)
        lam = dist.sample()
        # ==========
        mixup = lam * concatenate([real, distorted]) + (1 - lam) * concatenate([fake, distorted])
        # ==========
        output_mixup = netD(mixup)
        loss_D = loss_fn(output_mixup, lam * K.ones_like(output_mixup))
        # output_fake = netD(concatenate([fake, distorted])) # dummy
        loss_G = 1 * loss_fn(output_mixup, (1 - lam) * K.ones_like(output_mixup))
    else:
        output_real = netD(concatenate([real, distorted]))  # positive sample
        output_fake = netD(concatenate([fake, distorted]))  # negative sample
        loss_D_real = loss_fn(output_real, K.ones_like(output_real))
        loss_D_fake = loss_fn(output_fake, K.zeros_like(output_fake))
        loss_D = loss_D_real + loss_D_fake
        loss_G = 1 * loss_fn(output_fake, K.ones_like(output_fake))
        # ==========
    loss_G += K.mean(K.abs(fake_rgb - real))
    loss_G += K.mean(K.abs(fake_sz64 - tf.image.resize_images(real, [64, 64])))
    # ==========

    # Perceptual Loss
    if not vggface_feat is None:
        def preprocess_vggface(x):
            x = (x + 1) / 2 * 255  # channel order: BGR
            x -= [93.5940, 104.7624, 129.]
            return x

        pl_params = (0.02, 0.3, 0.5)
        real_sz224 = tf.image.resize_images(real, [224, 224])
        real_sz224 = Lambda(preprocess_vggface)(real_sz224)
        # ==========
        fake_sz224 = tf.image.resize_images(fake_rgb, [224, 224])
        fake_sz224 = Lambda(preprocess_vggface)(fake_sz224)
        # ==========
        real_feat55, real_feat28, real_feat7 = vggface_feat(real_sz224)
        fake_feat55, fake_feat28, fake_feat7 = vggface_feat(fake_sz224)
        loss_G += pl_params[0] * K.mean(K.abs(fake_feat7 - real_feat7))
        loss_G += pl_params[1] * K.mean(K.abs(fake_feat28 - real_feat28))
        loss_G += pl_params[2] * K.mean(K.abs(fake_feat55 - real_feat55))

    return loss_D, loss_G


# In[17]:


# ========== Define Perceptual Loss Model==========
if use_perceptual_loss:
    vggface.trainable = False
    out_size55 = vggface.layers[36].output
    out_size28 = vggface.layers[78].output
    out_size7 = vggface.layers[-2].output
    vggface_feat = Model(vggface.input, [out_size55, out_size28, out_size7])
    vggface_feat.trainable = False
else:
    vggface_feat = None

# In[19]:


loss_DA, loss_GA = define_loss(netDA, real_A, fake_A, fake_sz64_A, distorted_A, vggface_feat)
loss_DB, loss_GB = define_loss(netDB, real_B, fake_B, fake_sz64_B, distorted_B, vggface_feat)

loss_GA += 3e-3 * K.mean(K.abs(mask_A))
loss_GB += 3e-3 * K.mean(K.abs(mask_B))

# In[20]:


weightsDA = netDA.trainable_weights
weightsGA = netGA.trainable_weights
weightsDB = netDB.trainable_weights
weightsGB = netGB.trainable_weights

# Adam(..).get_updates(...)
training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsDA, [], loss_DA)
netDA_train = K.function([distorted_A, real_A], [loss_DA], training_updates)
training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(weightsGA, [], loss_GA)
netGA_train = K.function([distorted_A, real_A], [loss_GA], training_updates)

training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsDB, [], loss_DB)
netDB_train = K.function([distorted_B, real_B], [loss_DB], training_updates)
training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(weightsGB, [], loss_GB)
netGB_train = K.function([distorted_B, real_B], [loss_GB], training_updates)


# <a id='9'></a>
# # 9. Utils For Loading/Displaying Images

# In[21]:


def load_data(file_pattern):
    return glob.glob(file_pattern)


random_transform_args = {
    'rotation_range': 20,
    'zoom_range': 0.1,
    'shift_range': 0.05,
    'random_flip': 0.5,
}


def read_image(fn, random_transform_args=random_transform_args):
    image = cv2.imread(fn)
    image = cv2.resize(image, (256, 256)) / 255 * 2 - 1
    image = random_transform(image, **random_transform_args)
    warped_img, target_img = random_warp128(image)

    return warped_img, target_img


# In[22]:


# A generator function that yields epoch, batchsize of warped_img and batchsize of target_img
def minibatch(data, batchsize):
    length = len(data)
    epoch = i = 0
    tmpsize = None
    shuffle(data)
    while True:
        size = tmpsize if tmpsize else batchsize
        if i + size > length:
            shuffle(data)
            i = 0
            epoch += 1
        rtn = np.float32([read_image(data[j]) for j in range(i, i + size)])
        i += size
        tmpsize = yield epoch, rtn[:, 0, :, :, :], rtn[:, 1, :, :, :]


def minibatchAB(dataA, batchsize):
    batchA = minibatch(dataA, batchsize)
    tmpsize = None
    while True:
        ep1, warped_img, target_img = batchA.send(tmpsize)
        tmpsize = yield ep1, warped_img, target_img


# In[23]:


def showG(test_A, test_B, path_A, path_B):
    figure_A = np.stack([
        test_A,
        np.squeeze(np.array([path_A([test_A[i:i + 1]]) for i in range(test_A.shape[0])])),
        np.squeeze(np.array([path_B([test_A[i:i + 1]]) for i in range(test_A.shape[0])])),
    ], axis=1)
    figure_B = np.stack([
        test_B,
        np.squeeze(np.array([path_B([test_B[i:i + 1]]) for i in range(test_B.shape[0])])),
        np.squeeze(np.array([path_A([test_B[i:i + 1]]) for i in range(test_B.shape[0])])),
    ], axis=1)

    figure = np.concatenate([figure_A, figure_B], axis=0)
    figure = figure.reshape((4, 7) + figure.shape[1:])
    figure = stack_images(figure)
    figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
    figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)

    display(Image.fromarray(figure))


def showG_mask(test_A, test_B, path_A, path_B):
    figure_A = np.stack([
        test_A,
        (np.squeeze(np.array([path_A([test_A[i:i + 1]]) for i in range(test_A.shape[0])]))) * 2 - 1,
        (np.squeeze(np.array([path_B([test_A[i:i + 1]]) for i in range(test_A.shape[0])]))) * 2 - 1,
    ], axis=1)
    figure_B = np.stack([
        test_B,
        (np.squeeze(np.array([path_B([test_B[i:i + 1]]) for i in range(test_B.shape[0])]))) * 2 - 1,
        (np.squeeze(np.array([path_A([test_B[i:i + 1]]) for i in range(test_B.shape[0])]))) * 2 - 1,
    ], axis=1)

    figure = np.concatenate([figure_A, figure_B], axis=0)
    figure = figure.reshape((4, 7) + figure.shape[1:])
    figure = stack_images(figure)
    figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
    figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)

    display(Image.fromarray(figure))


print("GAN loaded.")
