import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, AveragePooling2D, BatchNormalization, Concatenate
from tensorflow.keras.layers import Conv2D, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Input, Lambda, MaxPooling2D
from tensorflow.keras.applications.imagenet_utils import decode_predictions
# from tensorflow.keras.utils.data_utils import get_file
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Activation, Add, Concatenate, Conv2D, Dense,
                                     GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda,
                                     Reshape, multiply)

def channel_attention(input_feature, ratio=8, name=""):
    channel = K.int_shape(input_feature)[-1]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=False,
                             bias_initializer='zeros',
                             name="channel_attention_shared_one_" + str(name))
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=False,
                             bias_initializer='zeros',
                             name="channel_attention_shared_two_" + str(name))

    avg_pool = GlobalAveragePooling2D()(input_feature)
    max_pool = GlobalMaxPooling2D()(input_feature)

    avg_pool = Reshape((1, 1, channel))(avg_pool)
    max_pool = Reshape((1, 1, channel))(max_pool)

    avg_pool = shared_layer_one(avg_pool)
    max_pool = shared_layer_one(max_pool)

    avg_pool = shared_layer_two(avg_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature, name=""):
    kernel_size = 7

    cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])

    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False,
                          name="spatial_attention_" + str(name))(concat)
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return multiply([input_feature, cbam_feature])


def cbam_block(cbam_feature, ratio=8, name=""):
    cbam_feature = channel_attention(cbam_feature, ratio, name=name)
    cbam_feature = spatial_attention(cbam_feature, name=name)
    return cbam_feature



class NoisyAnd(Layer):
    """Custom NoisyAND layer from the Deep MIL paper"""

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(NoisyAnd, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a = 10  # fixed, controls the slope of the activation
        self.b = self.add_weight(name='b',
                                 shape=(1, input_shape[3]),
                                 initializer='uniform',
                                 trainable=True)
        super(NoisyAnd, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        mean = tf.reduce_mean(x, axis=[1, 2])
        res = (tf.nn.sigmoid(self.a * (mean - self.b)) - tf.nn.sigmoid(-self.a * self.b)) / (
                tf.nn.sigmoid(self.a * (1 - self.b)) - tf.nn.sigmoid(-self.a * self.b))
        return res

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[3]

def conv2d_bn(x, filters, kernel_size, strides=1, padding='same', activation='relu', use_bias=False, name=None):
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    if block_type == 'block35':
        branch_0 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(branch_1, 32, 3)
        branch_2 = conv2d_bn(x, 32, 1)
        branch_2 = conv2d_bn(branch_2, 48, 3)
        branch_2 = conv2d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 128, 1)
        branch_1 = conv2d_bn(branch_1, 160, [1, 7])
        branch_1 = conv2d_bn(branch_1, 192, [7, 1])
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn(branch_1, 256, [3, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    mixed = Concatenate(name=block_name + '_mixed')(branches)
    up = conv2d_bn(mixed, K.int_shape(x)[3], 1, activation=None, use_bias=True, name=block_name + '_conv')

    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale},
               name=block_name)([x, up])
    if activation is not None:
        x = Activation(activation, name=block_name + '_ac')(x)
    return x


def InceptionResNetV2(input_shape=[224, 224, 3],
                      classes=2):
    img_input = Input(shape=input_shape)

    x = conv2d_bn(img_input, 32, 3, strides=2, padding='valid')
    print(x.shape)
    x = conv2d_bn(x, 32, 3, padding='valid')
    print(x.shape)
    x = conv2d_bn(x, 64, 3)
    print(x.shape)
    x = MaxPooling2D(3, strides=2)(x)
    print(x.shape)

    x = conv2d_bn(x, 80, 1, padding='valid')
    print(x.shape)
    x = conv2d_bn(x, 192, 3, padding='valid')
    print(x.shape)
    x = MaxPooling2D(3, strides=2)(x)
    print(x.shape)

    # Mixed 5b (Inception-A block):35 x 35 x 192 -> 35 x 35 x 320
    branch_0 = conv2d_bn(x, 96, 1)
    print(x.shape)
    branch_1 = conv2d_bn(x, 48, 1)
    print(x.shape)
    branch_1 = conv2d_bn(branch_1, 64, 5)
    print(x.shape)
    branch_2 = conv2d_bn(x, 64, 1)
    print(x.shape)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    print(x.shape)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    print(x.shape)
    branch_pool = AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]

    x = Concatenate(name='mixed_5b')(branches)
    x = cbam_block(x, name="cbam_1")

    # 10次Inception-ResNet-A block:35 x 35 x 320 -> 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx)

    # Reduction-A block:35 x 35 x 320 -> 17 x 17 x 1088
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 256, 3)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='valid')
    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(name='mixed_6a')(branches)

    # 20次Inception-ResNet-B block: 17 x 17 x 1088 -> 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)

    # Reduction-B block: 17 x 17 x 1088 -> 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 256, 1)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='valid')
    branch_2 = conv2d_bn(x, 256, 1)
    branch_2 = conv2d_bn(branch_2, 288, 3)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='valid')
    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(name='mixed_7a')(branches)
    x = cbam_block(x, name="cbam_2")

    # 10次Inception-ResNet-C block: 8 x 8 x 2080 -> 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(x,
                                   scale=0.2,
                                   block_type='block8',
                                   block_idx=block_idx)
    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=10)

    # 8 x 8 x 2080 -> 8 x 8 x 1536
    x = cbam_block(x,name="cbam_3")
    x = conv2d_bn(x, 1536, 1, name='conv_7b')
    x = cbam_block(x,name="cbam_4")
    print(x.shape)
    x = NoisyAnd(classes)(x)
    # x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input

    # 创建模型
    model = Model(inputs, x, name='inception_resnet_v2')

    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    model = InceptionResNetV2()
    model.summary()
    # fname = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
    # weights_path = get_file(fname,BASE_WEIGHT_URL + fname,cache_subdir='models',file_hash='e693bd0210a403b3192acc6073ad2e96')
    # model.load_weights(fname)
    # img_path = 'elephant.jpg'
    # img = image.load_img(img_path, target_size=(299, 299))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    #
    # x = preprocess_input(x)
    #
    # preds = model.predict(x)
    # print('Predicted:', decode_predictions(preds))
