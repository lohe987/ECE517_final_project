import sys
from random import randint
import cv2 as cv
import tensorflow as tf
import numpy as np 
import os



def main(argv):

    borderType = cv.BORDER_REPLICATE
    window_name = "copyMakeBorder Demo"
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(type(y_train))
    # Loads an image
    src = cv.imread('vid1_1k/out_0001.jpg', cv.IMREAD_COLOR)
    tgt = cv.imread('vid1_4k/out_0001.jpg', cv.IMREAD_COLOR)
    # Check if image is loaded fine

    
    top = 5 # Pad 5 pixels on each dimensions
    bottom = top
    left = 5
    right = left
        
    dst = cv.copyMakeBorder(src, top, bottom, left, right, borderType)
    
    R = dst[:,:,0]
    G = dst[:,:,1]
    B = dst[:,:,2]
    
    R_4k = tgt[:,:,0]
    G_4k = tgt[:,:,1]
    B_4k = tgt[:,:,2]
    
    x_train = []
    y_train = []
    
    for i in range(5,1085):
        for j in range(5,1925):
            x_train.append(R[i-2:i+3,j-2:j+3])
            x_train.append(G[i-2:i+3,j-2:j+3])
            x_train.append(B[i-2:i+3,j-2:j+3])

            y_train.append(R_4k[i-5,j-4])
            y_train.append(G_4k[i-5,j-4])
            y_train.append(B_4k[i-5,j-4])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = x_train.reshape(x_train.shape[0], 5, 5, 1)

    print(x_train[0])
    print(y_train[0])
    print(x_train.shape)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32,kernel_size=(3,3), strides=(1,1),activation='relu',input_shape=(5,5,1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='softmax')
    ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test,  y_test, verbose=2)   

    return 0

if __name__ == "__main__":
    main(sys.argv[1:])