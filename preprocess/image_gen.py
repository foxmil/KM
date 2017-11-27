import random
from .datagen import *
from .image import *
import keras


# Saves all image representations to a lmdb writer.
def save_img_reps(lmdb_writer, img, class_num, num_classes, preprocess=None):
    # If a preprocess function is specfied then preprocess the image.
    if preprocess is not None:
        img = preprocess(img.astype('float32'))

    # Generate different equivalent representations of each image
    reps = gen_img_reps(img)
    
    # Generate the image out data
    y = keras.utils.to_categorical(class_num, num_classes=num_classes)

    # Save each representation to the lmdb and its appropriate label.
    for x in reps:
        lmdb_writer.write_point([x], [y])



# Save images in subdirectories corresponding to their label into a lmdb.
# This also will save all image representations into he lmdb.
def save_img_lmdb(lmdb_path, img_dir, batch_size=32, ext='tif', rgb=True,
                  preprocess=None):
    # Find the total number of classes in the directory to use
    num_classes = find_num_classes(img_dir, ext=ext)
    
    # If the number of classes is < 2 then throw an error since a softmax will not
    # properly work.
    if num_classes < 2:
        raise IOError('The number of classes must be > 1 for proper categorical encoding.')
    
    # Generate a list of all of the images.
    img_list = generate_img_list(img_dir, ext=ext)
    random.shuffle(img_list)
    print('Found ' + str(len(img_list)) + ' images in ' + str(num_classes) + ' classes.')
    
    # Calculate the size needed size for the lmdb and intitalize it.
    img = load_img(img_list[0])
    lmdb_size = 4*img.shape[0]*img.shape[1]*len(img_list)*11
    lmdb_size = 2*lmdb_size if img.shape[0]*img.shape[1] else lmdb_size
    lmdb_size = 3*lmdb_size if rgb is True else lmdb_size
    lmdb_size += 4*num_classes*len(img_list)
    print(lmdb_size/1024/1024)
    
    lmdb = kmdbWriter(lmdb_path, batch_size=batch_size, map_size=lmdb_size)
    
    
    # Since we now have a random list of image paths we will load each one, generate representations
    # and save them to the lmdb. We will also preprocess the image if it was specified.
    class_num = 0
    for img_path in img_list:
        img = load_img(img_path)
        
        save_img_reps(lmdb, img, class_num, num_classes, preprocess=preprocess)
#        print(lmdb.cur_key)
   
    
    lmdb.close()

