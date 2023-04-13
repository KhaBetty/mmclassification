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

def image_compare(image1, image2):
    print(np.max((image1 - image2)))


# load the paths to all the picture
def load_image_paths(path):
    result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path)
              for f in filenames if os.path.splitext(f)[1] == '.jpg']

    return result


# get the paths and load the images with Imread
def load_images(paths_list, color=None):
    list_of_images = []
    # print("list size" + str(len(paths_list)))
    for path in paths_list:
        if color:
            image = cv2.imread(path)
        else:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        list_of_images.append(image)
    return list_of_images


def sub_pixel_creator(image, path, num_of_images, image_num, size,depth):
    path_images = path + str(image_num)
    os.mkdir(path_images)  # take of the image extension
    directions_list = ['left', 'up', 'down', 'right']

    for i in range(1, num_of_images + 1):
        edited_image = image
        directions = []
        dir1=random.randrange(0, 4)
        dir2 = random.randrange(0, 4)

        # make sure that the directions do not cancel each other
        while dir1 == 3-dir2:
            dir1 = random.randrange(0, 4)
            dir2 = random.randrange(0, 4)
        directions.append(directions_list[dir1])
        directions.append(directions_list[dir2])
        print(directions)
        for j in range(2):  # move randomly 2 times
            direction = directions[j]
            edited_image = np.asarray(pad_vector(vector=np.asmatrix(edited_image[:, :]), how=direction, depth=depth),
                                      dtype='uint8')  # gray scale image , two dim
        edited_image = cv2.resize(edited_image, (size, size), interpolation=cv2.INTER_LINEAR)
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

#this function creates 4 subpixel images where the first is moved right, the second left, third up, fourth down
def sub_pixel_fixed_movements_creator(image, path, image_num, size,depth,eight_channels=None):
    path_images = path + str(image_num)
    directions = ['right', 'left', 'up', 'down']
    os.mkdir(path_images)  # take of the image extension
    for i in range(1, 5):
        edited_image = image
        direction = directions[i-1]
        edited_image = np.asarray(pad_vector(vector=np.asmatrix(edited_image[:, :]), how=direction, depth=depth),
                                      dtype='uint8')  # gray scale image , two dim
        edited_image = cv2.resize(edited_image, (size, size), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(path_images + '/' + str(i) + '.jpg', edited_image)
    dir1=['right', 'left']
    dir2=['up', 'down']
    counter=5

    if eight_channels:
        for rl in dir1:
            for ud in dir2:
                edited_image = image
                edited_image = np.asarray(pad_vector(vector=np.asmatrix(edited_image[:, :]), how=rl, depth=depth//2),
                                          dtype='uint8')  # gray scale image , two dim
                edited_image = np.asarray(pad_vector(vector=np.asmatrix(edited_image[:, :]), how=ud, depth=depth//2),
                                          dtype='uint8')  # gray scale image , two dim
                edited_image = cv2.resize(edited_image, (size, size), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(path_images + '/' + str(counter) + '.jpg', edited_image)
                counter = counter + 1


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
            orig_grey_path = (main_path + "orig_grey" + '/' + data_set +'/' + animal + 's/' + animal)
            resized_path = (main_path + "resized_32_rgb" + '/' + data_set +'/' + animal + 's/' + animal)
            subpixel_path = (main_path + "subpixel_32_4_depth_4_dir_2" + '/' + data_set + '/' + animal + 's/' + animal)
            subpixel_fixed_path = (main_path + "subpixel_32_4_fixed_depth_1" + '/' + data_set + '/' + animal + 's/' + animal)
            for counter, image in enumerate(images):
                sub_pixel_creator(image, subpixel_path, num_subpixel_images, counter + 1,32,4)
              # sub_pixel_fixed_movements_creator(image, subpixel_fixed_path,counter,32,1)
               #resized_creator(image, resized_path, counter + 1, 32)
               # original_grey_creator(image, orig_grey_path, counter + 1)
    # resize_images_and_save(images,orig_grey_path,resized_path,subpixel_path)





# load from - r"C:\Users\eyalg\Desktop\pics\dataset\training_set\dogs"
# save resize to - r"C:\Users\eyalg\Desktop\pics\resized\training_set\dogs"
# save shifted to - r"C:\Users\eyalg\Desktop\pics\shifted_and_resized\training_set\dogs"
