import keras
from keras.layers import Input, Dense, Dropout, Lambda
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.engine.topology import Layer
import numpy as np



# Defines a resnet network with 4096 feature output vector.
def resnet_raw(input_shape=(550, 550, 3), pooling='max'):
    input_layer = Input(input_shape, name='input')
    resnet = ResNet50(include_top=False, input_shape=input_shape, pooling=pooling) (input_layer)

    hidden = Dense(4096, activation='relu') (resnet)
    hidden = Dropout(0.5) (hidden)
    hidden = Dense(4096, activation='relu') (hidden)
    
    model = Model(inputs=[input_layer], outputs=[hidden], name='resnet')
    
    return model


# Uses resnet raw to define a softmax output to use for training
def resnet_softmax(num_classes, input_shape=(550, 550, 3), pooling='max'):
    input_layer = Input(input_shape, name='input')
    
    resnet = resnet_raw(input_shape=input_shape, pooling=pooling) (input_layer)
    hidden = Dropout(0.5) (resnet)
    
    output = Dense(num_classes, activation='softmax') (hidden)
    
    model = Model(inputs=[input_layer], outputs=[output])

    return model