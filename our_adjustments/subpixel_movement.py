import cv2
import numpy as np


def pad_vector(vector, how, depth, constant_value=0):
	vect_shape = vector.shape[:2]
	if (how == 'upper') or (how == 'top'):
		pp = np.full(shape=(depth, vect_shape[1]), fill_value=constant_value)
		pv = np.vstack(tup=(pp, vector))
	elif (how == 'lower') or (how == 'bottom'):
		pp = np.full(shape=(depth, vect_shape[1]), fill_value=constant_value)
		pv = np.vstack(tup=(vector, pp))
	elif (how == 'left'):
		pp = np.full(shape=(vect_shape[0], depth), fill_value=constant_value)
		pv = np.hstack(tup=(pp, vector))
	elif (how == 'right'):
		pp = np.full(shape=(vect_shape[0], depth), fill_value=constant_value)
		pv = np.hstack(tup=(vector, pp))
	else:
		return vector
	return pv


def image_compare(image1, image2):
	print(np.max((image1 - image2)))

if __name__ == '__main__':
	image = cv2.imread('dog_gray_resized.jpg', cv2.IMREAD_GRAYSCALE)
	image2 = cv2.imread('dog_shifted_resized.jpg', cv2.IMREAD_GRAYSCALE)
	image_compare(image, image2)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #shift
# #shifted_image = pad_vector(vector=np.asmatrix(image), how='left', depth=3)
# #cv2.imwrite('dog_gray.jpg',image)# np.asarray(shifted_image))
# #resize
# resized_image = cv2.resize(image, (128,128), interpolation = cv2.INTER_LINEAR)
# cv2.imwrite('dog_shifted_resized.jpg', resized_image)  # np.asarray(shifted_image))
