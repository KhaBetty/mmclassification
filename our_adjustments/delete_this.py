import cv2
import numpy as np
import glob
import os
import random

def pad_vector(vector, how, depth, constant_value=0):
    vect_shape = vector.shape[:2]
    if how == 'up':
        pp = np.full(shape=(depth, vect_shape[1]), fill_value=constant_value)
        pv = np.vstack(tup=(pp, vector))
    elif how == 'down':
        pp = np.full(shape=(depth, vect_shape[1]), fill_value=constant_value)
        pv = np.vstack(tup=(vector, pp))
    elif how == 'left':
        pp = np.full(shape=(vect_shape[0], depth), fill_value=constant_value)
        pv = np.hstack(tup=(pp, vector))
    elif how == 'right':
        pp = np.full(shape=(vect_shape[0], depth), fill_value=constant_value)
        pv = np.hstack(tup=(vector, pp))
    else:
        return vector
    return pv

def mover (image, how, depth, constant_value=0):
    if how == 'up':
        image[:depth,:]=constant_value
    if how == 'down':
        image[-1*depth:,:]=constant_value
    if how == 'left':
        image[:,:depth]=constant_value
    if how == 'right':
        image[:,-1*depth:]=constant_value
    return image

def cropper  (image, how, depth, constant_value=0):
    if how == 'left':
        image=image [depth:-depth,:-2*depth]
    if how == 'right':
        image = image[depth:-depth, 2*depth:]
    if how == 'up':
        image = image[:-2*depth, depth:-depth]
    if how == 'down':
        image = image[2*depth:, depth:-depth]
    return image

def resized_creator(image, size, depth):
    standard_size=232
    edited_image = image
    edited_image = cv2.resize(edited_image, (standard_size, standard_size), interpolation=cv2.INTER_LINEAR)
    edited_image = edited_image[depth:-depth, depth:-depth]
    edited_image = cv2.resize(edited_image, (size, size), interpolation=cv2.INTER_LINEAR)
    return edited_image
if __name__ == '__main__':
    path='/home/maya/projA/mmclassification/our_adjustments/dog_image.jpg'
    size=32
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    edited_image = cv2.resize(image, (232, 232), interpolation=cv2.INTER_LINEAR) #232 image
    moved_image = np.asarray(cropper(edited_image, how='up', depth=0), dtype='uint8')
   # moved_image2 = np.asarray(cropper(edited_image, how='down', depth=4), dtype='uint8')

    resized_image = resized_creator(edited_image, 32, 50)
   # resized_image2 = cv2.resize(moved_image2, (size, size), interpolation=cv2.INTER_LINEAR)

    cv2.imshow('orig', edited_image)
    cv2.imshow('bla', resized_image)
   # cv2.imshow('blabla', resized_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(image.shape)
    print(moved_image.shape)
    print(resized_image.shape)