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

def random_direction():
    directions=['left','right','up','down']
    return directions[random.randrange(0,4)]

def image_compare(image1, image2):
	print(np.max((image1 - image2)))

#load the paths to all the picture
def load_image_paths(path):
	result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path)
			  for f in filenames if os.path.splitext(f)[1] == '.jpg']

	return result

#get the paths and load the images with Imread
def load_images(paths_list):
    list_of_images=[]
    #print("list size" + str(len(paths_list)))
    for path in paths_list:
        image = cv2.imread(path)
        list_of_images.append(image)
    return list_of_images

#get the list of images and return a tuple: 1st element is resized images, 2nd element is shifted and resized images
def resize_images_and_save(list_of_images,path_resized,path_shifted):

    resized=[]
    resized_and_shifted=[]
    i=1
    for image in list_of_images:
        resized_image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
        #print(image.shape)
        #print(type(image))
        direction = random_direction()
        temp=np.asarray(pad_vector(vector=np.asmatrix(image[:,:,0]), how=direction, depth=1), dtype='uint8')
        shifted_image=np.empty((temp.shape[0],temp.shape[1],3))
        shifted_image[:,:,0] = np.asarray(pad_vector(vector=np.asmatrix(image[:,:,0]), how=direction, depth=1), dtype='uint8')
        shifted_image[:,:,1] = np.asarray(pad_vector(vector=np.asmatrix(image[:, :, 1]), how=direction, depth=1),dtype='uint8')
        shifted_image[:,:,2] = np.asarray(pad_vector(vector=np.asmatrix(image[:, :, 2]), how=direction, depth=1), dtype='uint8')
        #print(resized_image.shape)
        #print(shifted_image.shape)
        #print(type(resized_image))
        #print(type(shifted_image))
        #cv2.imshow('resized_image', image)
        #cv2.waitKey(0)

        shifted_resized_image=cv2.resize(shifted_image, (128, 128), interpolation=cv2.INTER_LINEAR)
        resized.append(resized_image)
        resized_and_shifted.append(shifted_resized_image)
        cv2.imwrite(path_resized + str(i) +'.jpg', resized_image)
        cv2.imwrite(path_shifted + str(i) +'.jpg', shifted_resized_image)
        i = i+1

    return resized, resized_and_shifted

if __name__ == '__main__':
    path_list = load_image_paths(r"C:\Users\eyalg\Desktop\pics\dataset\validation_set\dogs")
    images = load_images(path_list)
    tup = resize_images_and_save(images, r"C:\Users\eyalg\Desktop\pics\color\resized\validation_set\dogs\dog", r"C:\Users\eyalg\Desktop\pics\color\shifted_and_resized\validation_set\dogs\dog")
#load from - r"C:\Users\eyalg\Desktop\pics\dataset\training_set\dogs"
#save resize to - r"C:\Users\eyalg\Desktop\pics\resized\training_set\dogs"
#save shifted to - r"C:\Users\eyalg\Desktop\pics\shifted_and_resized\training_set\dogs"







# image = cv2.imread('dog_gray_resized.jpg', cv2.IMREAD_GRAYSCALE)
# image2 = cv2.imread('dog_shifted_resized.jpg', cv2.IMREAD_GRAYSCALE)
# image_compare(image, image2)

# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #shift
# #shifted_image = pad_vector(vector=np.asmatrix(image), how='left', depth=3)
# #cv2.imwrite('dog_gray.jpg',image)# np.asarray(shifted_image))
# #resize
# resized_image = cv2.resize(image, (128,128), interpolation = cv2.INTER_LINEAR)
# cv2.imwrite('dog_shifted_resized.jpg', resized_image)  # np.asarray(shifted_image))


