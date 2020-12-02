import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import ZeroPadding2D, UpSampling2D, Conv2D, Lambda, BatchNormalization, LeakyReLU, Input, Add
from tensorflow.keras.utils import plot_model


def padding(inputs):
    return tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]])


def resdual_net(x, num_filters, num_blocks, name=None):
    x = Lambda(padding)(x)
    x = Conv2D(filters=num_filters, kernel_size=3, strides=2, padding='VALID', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    for i in range(num_blocks):
        y = Conv2D(filters=num_filters//2, kernel_size=1, strides=1, padding='VALID', use_bias=False)(x)
        y = BatchNormalization()(y)
        y = LeakyReLU(alpha=0.1)(y)

        y = Lambda(padding)(y)
        y = Conv2D(filters=num_filters, kernel_size=3, strides=1, padding='VALID', use_bias=False)(y)
        y = BatchNormalization()(y)
        y = LeakyReLU(alpha=0.1)(y)
        if i == num_blocks-1:
            x = Add(name=name)([x, y])
        else:
            x = Add()([x, y])
    return x


def DarknetConv2D_BN_Leaky(x, filters=32, kernel=3):
    if kernel == 3:
        x = Lambda(padding)(x)
        x = Conv2D(filters=filters, kernel_size=kernel, padding='VALID', use_bias=False)(x)
    elif kernel == 1:
        x = Conv2D(filters=filters, kernel_size=kernel, padding='VALID', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def yolo_v3(input_shape=(416, 416, 3), obj_c=3*(4+5)):
    inputs = Input(shape=input_shape, name='img_input')
    x = Lambda(padding)(inputs)
    x = Conv2D(filters=32, kernel_size=3, padding='VALID', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = resdual_net(x, 64, 1)
    x = resdual_net(x, 128, 2)
    x_8 = resdual_net(x, 128, 4, name='shortcut_8')
    x_16 = resdual_net(x_8, 256, 4, name='shortcut_16')
    x_32 = resdual_net(x_16, 512, 2, name='shortcut_32')

    x = DarknetConv2D_BN_Leaky(x_32, filters=512, kernel=1)
    x = DarknetConv2D_BN_Leaky(x, filters=512*2, kernel=3)
    x = DarknetConv2D_BN_Leaky(x, filters=512, kernel=1)
    x = DarknetConv2D_BN_Leaky(x, filters=512 * 2, kernel=3)
    x1 = DarknetConv2D_BN_Leaky(x, filters=512, kernel=1)
    x = DarknetConv2D_BN_Leaky(x1, filters=512 * 2, kernel=3)
    y1 = Conv2D(filters=obj_c, kernel_size=1)(x)

    x = DarknetConv2D_BN_Leaky(x1, filters=256, kernel=1)
    x = UpSampling2D(2)(x)
    x = tf.keras.layers.concatenate([x, x_16])

    x = DarknetConv2D_BN_Leaky(x, filters=256, kernel=1)
    x = DarknetConv2D_BN_Leaky(x, filters=256 * 2, kernel=3)
    x = DarknetConv2D_BN_Leaky(x, filters=256, kernel=1)
    x = DarknetConv2D_BN_Leaky(x, filters=256 * 2, kernel=3)
    x2 = DarknetConv2D_BN_Leaky(x, filters=256, kernel=1)
    x = DarknetConv2D_BN_Leaky(x2, filters=256 * 2, kernel=3)
    y2 = Conv2D(filters=obj_c, kernel_size=1)(x)

    x = DarknetConv2D_BN_Leaky(x2, filters=128, kernel=1)
    x = UpSampling2D(2)(x)
    x = tf.keras.layers.concatenate([x, x_8])

    x = DarknetConv2D_BN_Leaky(x, filters=128, kernel=1)
    x = DarknetConv2D_BN_Leaky(x, filters=128 * 2, kernel=3)
    x = DarknetConv2D_BN_Leaky(x, filters=128, kernel=1)
    x = DarknetConv2D_BN_Leaky(x, filters=128 * 2, kernel=3)
    x = DarknetConv2D_BN_Leaky(x, filters=128, kernel=1)
    x = DarknetConv2D_BN_Leaky(x, filters=128 * 2, kernel=3)
    y3 = Conv2D(filters=obj_c, kernel_size=1)(x)
    model = Model(inputs, [y3, y2, y1])
    model.summary()
    return model


if __name__ == '__main__':
    model = yolo_v3()
    plot_model(model, show_shapes=True, to_file='YOLOV3Net.png')
    import numpy as np
    img = np.zeros(shape=(1, 416, 416, 3))
    z = model(img)
    print(z)