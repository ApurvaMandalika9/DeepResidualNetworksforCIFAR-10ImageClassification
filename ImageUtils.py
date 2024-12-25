import numpy as np
import random

""" This script implements the functions for data augmentation and preprocessing.
"""

def parse_record(record, training):
    """ Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image

def preprocess_image(image, training):
    """ Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    if training:
        ### YOUR CODE HERE
        # Resize the image to add four extra pixels on each side.
        
        ### YOUR CODE HERE
        image = np.pad(image,((2,2),(2,2),(0,0)),'constant')
        ### YOUR CODE HERE
        # Randomly crop a [32, 32] section of the image.
        # HINT: randomly generate the upper left point of the image
        
        ### YOUR CODE HERE
        [top_left_x,top_left_y] = np.random.randint(low=0,high=4+1,size=(2,))
        image = image[top_left_x:top_left_x+32,top_left_y:top_left_y+32,:]
        ### YOUR CODE HERE
        # Randomly flip the image horizontally.
        if random.random() > 0.5:
            image = np.flip(image,axis=1)
        ### YOUR CODE HERE

    ### YOUR CODE HERE
    # Subtract off the mean and divide by the standard deviation of the pixels.
    mean = np.mean(image,axis=(1,2),keepdims=True)
    std = np.std(image,axis=(1,2),keepdims=True)
    image = (image-mean)/(std+1e-5)
    ### YOUR CODE HERE

    return image