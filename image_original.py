import skimage as ski
import numpy as np
import random
from PIL import Image

def load():
    """Loads greyscale image array of size (1000, 2500), scaled to interval [0,1]. 13 base images, 77 samples each, final sample discarded."""

    random.seed(0)

    directory = "data/image_original/"
    filenames = "1 2 3 4 5 6 7 8 9 10 11 12 13".split()
    extension = ".tiff"

    data = np.empty((1001, 50*50))
    for n in range(len(filenames)):
        path = directory+filenames[n]+extension
        imgarray = np.array(Image.open(path))
        print("Got imgarray of shape ",imgarray.shape)
        print(imgarray)
        for m in range(77):
            width,height = imgarray.shape
            x = random.randrange(width-50)
            y = random.randrange(height-50)
            data[n*77+m] = imgarray[x:x+50,y:y+50].flatten()/255
            # print(f"Populated index {n*77+m}")
    print("Return shape: ",data[:-1].shape)
    return data[:-1] # Return only 1000 samples

if __name__ == '__main__':
    load()