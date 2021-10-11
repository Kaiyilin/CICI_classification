import sys, os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, 
    Dropout, 
    Activation, 
    BatchNormalization, 
    Conv3D, 
    Dense,
    GlobalAveragePooling3D,
    MaxPooling3D, 
    AveragePooling3D,
    concatenate
)
from tensorflow.keras.regularizers import l1_l2
import tensorflow.keras.backend as K


class DenseNet3Dbuilder(object):
    """ Building DenseNet"""

    @staticmethod
    def build(input_shape, 
              n_classes, 
              growth_rate, 
              repetitions,  
              bottleneck_ratio , 
              reg_factor):
        
        def bn_rl_conv(input, filters, kernel_size=(1,1,1), strides=1):
            
            x = BatchNormalization()(input)
            x = Activation('relu')(x)
            x = Conv3D(filters=filters, kernel_size=kernel_size, 
                       strides=strides, padding='same',
                       kernel_initializer='he_normal',
                       kernel_regularizer=l1_l2(l1=0,l2=reg_factor)
                       )(x)

            return x
        
        def dense_block(x, growth_rate, repetition):
            
            for _ in range(repetition):
                y = bn_rl_conv(x, filters=growth_rate * bottleneck_ratio)
                y = bn_rl_conv(y, filters=growth_rate, kernel_size=(3,3,3))
                x = tf.keras.layers.concatenate([y,x])

            return x
            
        def transition_layer(x):
            
            x = bn_rl_conv(x, K.int_shape(x)[-1] //2 )
            x = AveragePooling3D(pool_size=(2,2,2), strides=2, padding='same')(x)
            return x
        
        input = Input(input_shape)
        x = Conv3D(filters=96, 
                   kernel_size=(7,7,7), 
                   strides=2, 
                   padding='same')(input)

        x = MaxPooling3D(pool_size=(3,3,3),
                        strides=2, 
                        padding='same')(x)
        
        for repetition in repetitions:
            
            d = dense_block(x=x, growth_rate=growth_rate, repetition=repetition)
            x = transition_layer(d)
        x = GlobalAveragePooling3D()(d)
        output = Dense(units=n_classes, 
                       activation='softmax')(x)
        
        model = Model(input, output)
        return model

    @staticmethod
    def build_densenet_121(input_shape, 
                           n_classes, 
                           growth_rate, 
                           bottleneck_ratio=4, 
                           reg_factor=1e-4):
                        
        return DenseNet3Dbuilder.build(input_shape, n_classes, 
                                       growth_rate, [6, 12, 24, 16], 
                                       bottleneck_ratio, reg_factor=reg_factor)

    @staticmethod
    def build_densenet_169(input_shape, 
                           n_classes, 
                           growth_rate, 
                           bottleneck_ratio=4, 
                           reg_factor=1e-4):

        return DenseNet3Dbuilder.build(input_shape, n_classes, 
                                       growth_rate, [6, 12, 32, 32], 
                                       bottleneck_ratio, reg_factor=reg_factor)

    @staticmethod
    def build_densenet_201(input_shape, 
                           n_classes, 
                           growth_rate, 
                           bottleneck_ratio=4, 
                           reg_factor=1e-4):

        return DenseNet3Dbuilder.build(input_shape, n_classes, 
                                       growth_rate, [6, 12, 48, 32], 
                                       bottleneck_ratio, reg_factor=reg_factor)
    

    @staticmethod
    def build_densenet_264(input_shape, 
                           n_classes, 
                           growth_rate, 
                           bottleneck_ratio=4, 
                           reg_factor=1e-4):

        return DenseNet3Dbuilder.build(input_shape, n_classes, 
                                       growth_rate, [6, 12, 64, 48], 
                                       bottleneck_ratio, reg_factor=reg_factor)