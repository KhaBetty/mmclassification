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
    directions = ['left', 'right', 'up', 'down']
    return directions[random.randrange(0, 4)]


def image_compare(image1, image2):
    print(np.max((image1 - image2)))


# load the paths to all the picture
def load_image_paths(path):
    result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path)
              for f in filenames if os.path.splitext(f)[1] == '.jpg']

    return result


# get the paths and load the images with Imread
def load_images(paths_list):
    list_of_images = []
    # print("list size" + str(len(paths_list)))
    for path in paths_list:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        list_of_images.append(image)
    return list_of_images


def sub_pixel_creator(image, path, num_of_images, image_num):
    path_images = path + str(image_num)
    os.mkdir(path_images)  # take of the image extension
    for i in range(1, num_of_images + 1):
        edited_image = image
        for _ in range(3):  # move randomly 3 times
            direction = random_direction()
            edited_image = np.asarray(pad_vector(vector=np.asmatrix(edited_image[:, :]), how=direction, depth=1),
                                      dtype='uint8')  # gray scale image , two dim
        edited_image = cv2.resize(edited_image, (64, 64), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(path_images + '/' + str(i) + '.jpg', edited_image)

def resized_creator(image, path, image_num):
    edited_image = image
    edited_image = cv2.resize(edited_image, (64, 64), interpolation=cv2.INTER_LINEAR)
    path = path + str(image_num) + '.jpg'
    cv2.imwrite(path, edited_image)



# get the list of images and return a tuple: 1st element is resized images, 2nd element is shifted and resized images
def resize_images_and_save(list_of_images, path_resized, path_shifted):
    resized = []
    resized_and_shifted = []
    num_moved_images = 4

    for counter, image in enumerate(list_of_images):
        sub_pixel_creator(image, path_shifted, num_moved_images, counter + 1)
        resized_creator(image, path_resized, counter + 1)

        # resized_image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
        # print(image.shape)
        # print(type(image))
        #
        # direction = random_direction() temp=np.asarray(pad_vector(vector=np.asmatrix(image[:,:,0]), how=direction,
        # depth=1), dtype='uint8') shifted_image=np.empty((temp.shape[0],temp.shape[1],3)) shifted_image[:,:,
        # 0] = np.asarray(pad_vector(vector=np.asmatrix(image[:,:,0]), how=direction, depth=1), dtype='uint8')
        # shifted_image[:,:,1] = np.asarray(pad_vector(vector=np.asmatrix(image[:, :, 1]), how=direction, depth=1),
        # dtype='uint8') shifted_image[:,:,2] = np.asarray(pad_vector(vector=np.asmatrix(image[:, :, 2]),
        # how=direction, depth=1), dtype='uint8') #print(resized_image.shape) print(shifted_image.shape) print(type(
        # resized_image)) print(type(shifted_image)) cv2.imshow('resized_image', image) #cv2.waitKey(0)
        #
        # shifted_resized_image=cv2.resize(shifted_image, (64, 64), interpolation=cv2.INTER_LINEAR)
        # resized.append(resized_image)
        # resized_and_shifted.append(shifted_resized_image)
        # cv2.imwrite(path_resized + str(i) + '.jpg', resized_image)
        # cv2.imwrite(path_shifted + str(i) +'.jpg', shifted_resized_image)

    #return resized, resized_and_shifted


if __name__ == '__main__':
    # orig_img =['cats', 'dogs']
    #     #['test_set/cats','test_set/dogs', 'training_set/dogs', 'training_set/cats','validation_set/dogs', 'validation_set/cats']

    path_list = load_image_paths("/home/maya/Pictures/projA_pics/dataset_balanced/test_set/cats")
    images = load_images(path_list)

    #tup = resize_images_and_save(images, '/home/maya/Pictures/projA_pics/resized/test_set/cats/cat',
             #                '/home/maya/Pictures/projA_pics/subpixel_trial/test_set/cats/cat')

    sets = ["training_set","validation_set","test_set"]
    animals =["cat" , "dog"]
    for data_set in sets:
        for animal in animals:
            path_list = load_image_paths("/home/maya/Pictures/projA_pics/dataset_balanced" + '/' + data_set + '/' + animal + 's')
            images = load_images(path_list)
            resized_path = ("/home/maya/Pictures/projA_pics/resized" + '/' + data_set +'/' + animal + 's/' + animal)
            subpixel_path = ("/home/maya/Pictures/projA_pics/subpixel" + '/' + data_set + '/' + animal + 's/' + animal)
            resize_images_and_save(images,resized_path,subpixel_path)


# load from - r"C:\Users\eyalg\Desktop\pics\dataset\training_set\dogs"
# save resize to - r"C:\Users\eyalg\Desktop\pics\resized\training_set\dogs"
# save shifted to - r"C:\Users\eyalg\Desktop\pics\shifted_and_resized\training_set\dogs"
