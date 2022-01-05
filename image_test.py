from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from math import ceil

def load():
    """Loads greyscale test image array of size (3, 4096), scaled to interval [0,1]"""
    directory = "data/image_test/"
    filenames = "1 2 3".split()
    extension = ".bmp"

    data = np.empty((len(filenames), 64*64))
    for n in range(len(filenames)):
        path = directory+filenames[n]+extension
        imgarray = np.array(Image.open(path))
        data[n,:] = imgarray.flatten()/255
    return data

    

def show_images(imgarray):
    """Display original images next to images given in argument, for comparison purposes"""
    orig_images = load().reshape((3,64,64))
    new_images = imgarray.reshape((3,64,64))
    sqdiff_orig = np.abs(orig_images - np.roll(orig_images, shift=-1, axis=0))
    sqdiff_new = np.abs(new_images - np.roll(new_images, shift=-1, axis=0))
    images = np.concatenate((orig_images, sqdiff_orig, new_images, sqdiff_new), axis=0)
    # images = np.concatenate((orig_images, new_images), axis=0)
    n_images = len(images)
    rows = images.shape[0]//3
    titles = ['Image (%d)' % i for i in range(1,n_images + 1)]

    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(rows, ceil(n_images/float(rows)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

if __name__ == "__main__":
    images = load().reshape((3,64,64))
    show_images(images)