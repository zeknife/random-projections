# Authors: Alexander Åblad & Erland Ekholm (see comments for specific contributions)
# 2022-01-07
# DD2434

# Ex: Initialize DCT(50) for input image arrays of d=2500 (sidelength of square image)
# DCT_obj.transform(image_dataset_array, dimensions_to_reduce_to), i.e. 800 filters out 1700 DCT frequencies/features

import numpy as np

class DCT:
    def __init__(self, dct_size):
        self.dct_size = dct_size

    def transform(self, imgarray, k):
        # Erland Ekholm
        # Input 1000x2500 array, k: dim reduction (n dims to reduce to)

        self.W = self.generate_W(k, self.dct_size)
        self.dct = self.basis(self.dct_size)

        for row in range(imgarray.shape[0]):
            transformed_row = self.transform_internal(imgarray[row])
            imgarray[row] = transformed_row
        return imgarray

    def transform_internal(self, imgblock):
        # Erland Ekholm

        reshaped_imgblock = imgblock.reshape((self.dct_size, self.dct_size))
        enc_imgblock = self.encode(reshaped_imgblock)
        dec_imgblock = self.decode(enc_imgblock)

        return dec_imgblock.flatten()

    def basis(self, m):
        # Alexander Åblad
        # DCT Basis matrix. Used together with Image in matrix mult. to calculate coefficients

        B = np.zeros((m, m))
        for p in range(m):
            for q in range(m):
                # Branchless implementation
                B[p, q] = (q==0) * 1/np.sqrt(m) + (q>0) * np.sqrt(2/m)*np.cos((np.pi*(2*p+1)*q)/(2*m))
        return B

    def encode(self, I):
        # Alexander Åblad
        # Input: image

        I -= 0.5    # Centre around 0 as cosine is negative and positive
        C = self.dct.T@I@self.dct
        C *= self.W  # Masks out higher frequencies
        return C

    def decode(self, C):
        # Erland Ekholm
        # Input: coefficient matrix

        I = self.dct@C@self.dct.T
        I += 0.5    # Re-offset from centre
        return I

    def generate_W(self, n_ones, size):
        # Alexander Åblad
        # Input n_ones: dimensionality reduction
        # Size: size of W-matrix
        # Creates simple, approximate masking matrix for feature selection
        # (i.e. not always completely symmetrical in corners of triangle of ones)

        W = np.zeros((size, size))
        step = 0
        count = 0
        rowstop = int(np.sqrt(n_ones * 2))

        for row in range(0, rowstop):
            for i in range(0, rowstop - step):
                W[row][i] = 1
                count += 1
                if count == n_ones:
                    return W
            step += 1
        return W
