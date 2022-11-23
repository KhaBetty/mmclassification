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

def resized_creator(image, path, image_num, size):
    edited_image = image
    edited_image = cv2.resize(edited_image, (size, size), interpolation=cv2.INTER_LINEAR)
    path = path + str(image_num) + '.jpg'
    cv2.imwrite(path, edited_image)
def original_grey_creator(image, path, image_num):
    edited_image = image
    path = path + str(image_num) + '.jpg'
    cv2.imwrite(path, edited_image)



# get the list of images and return a tuple: 1st element is resized images, 2nd element is shifted and resized images

if __name__ == '__main__':

    main_path="/home/maya/Pictures/projA_pics/"
    num_subpixel_images = 4

    sets = ["training_set","validation_set","test_set"]
    animals =["cat" , "dog"]
    for data_set in sets:
        for animal in animals:

            path_list = load_image_paths(main_path + "dataset_balanced" + '/' + data_set + '/' + animal + 's')
            images = load_images(path_list)
            orig_grey_path = (main_path + "orig_color" + '/' + data_set +'/' + animal + 's/' + animal)
            resized_path = (main_path + "resized_32" + '/' + data_set +'/' + animal + 's/' + animal)
            subpixel_path = (main_path + "subpixel" + '/' + data_set + '/' + animal + 's/' + animal)
            for counter, image in enumerate(images):
               # sub_pixel_creator(image, subpixel_path, num_subpixel_images, counter + 1)
               # resized_creator(image, resized_path, counter + 1, 32)
                original_grey_creator(image, orig_grey_path, counter + 1)
    # resize_images_and_save(images,orig_grey_path,resized_path,subpixel_path)





# load from - r"C:\Users\eyalg\Desktop\pics\dataset\training_set\dogs"
# save resize to - r"C:\Users\eyalg\Desktop\pics\resized\training_set\dogs"
# save shifted to - r"C:\Users\eyalg\Desktop\pics\shifted_and_resized\training_set\dogs"
