import numpy as np
import rp
import image_original
import image_test
import matplotlib.pyplot as plt
# from sklearn.random_projection import GaussianRandomProjection

def dist(a,b):
    # Axis -1 to allow for both vectors and arrays
    return np.linalg.norm(a-b, axis=-1)

def imgtest():
    testarr = image_test.load()

    proj = rp.GaussianRP(100, testarr.shape[-1])
    # proj2 = GaussianRandomProjection(n_components=64)
    # proj2.fit(testarr)

    test_mapped = proj.transform(testarr)
    test_inversed = proj.fast_inverse(test_mapped)

    print("Distance scaling:",proj.dist_factor)

    print("Original shape:",testarr.shape)
    print("Mapped shape:",test_mapped.shape)


    print("Distances before projection:")
    print(dist(testarr[0], testarr[1]))
    print(dist(testarr[1], testarr[2]))
    print(dist(testarr[0], testarr[2]))

    print("Distances after projection:")
    print(dist(test_mapped[0], test_mapped[1]))
    print(dist(test_mapped[1], test_mapped[2]))
    print(dist(test_mapped[0], test_mapped[2]))

    print("Scaled distances after projection:")
    print(proj.dist_factor*dist(test_mapped[0], test_mapped[1]))
    print(proj.dist_factor*dist(test_mapped[1], test_mapped[2]))
    print(proj.dist_factor*dist(test_mapped[0], test_mapped[2]))

    image_test.show_images(test_inversed)

def imgoriginal():
    K = 800 # Max dimensionality to test
    rng = np.random.default_rng()
    rnd_pairs = rng.choice(1000, (2,100), replace=False)
    # print(f"{rnd_pairs=}")

    imgarr = image_original.load()
    dist_before = dist(imgarr[rnd_pairs[0]], imgarr[rnd_pairs[1]])
    # print("Distances before reduction: ")
    # print(dist_before)

    # print("pairs:",rnd_pairs[0],rnd_pairs[1])
    # print("pairs shape:",rnd_pairs[0].shape)

    distGauss = np.empty((K, 100))
    distSparse = np.empty((K, 100))
    for k in range(K):
        rGaussian = rp.GaussianRP(k+1, imgarr.shape[-1])
        rSparse = rp.SparseRP(k+1, imgarr.shape[-1])
        projGauss = rGaussian.transform(imgarr)
        projSparse = rSparse.transform(imgarr)

        distGauss[k] = dist(projGauss[rnd_pairs[0]], projGauss[rnd_pairs[1]])
        distSparse[k] = dist(projSparse[rnd_pairs[0]], projSparse[rnd_pairs[1]])
        # print("newdist shape:",newdist.shape)
        #  = newdist
        print(f"k={k+1}")
        # print(rSparse.mapping)
        # print(dist_after[k])
    errorGauss = np.abs(distGauss - dist_before)
    errorSparse = np.abs(distSparse - dist_before)
    meanGauss = np.mean(errorGauss, axis=1)
    meanSparse = np.mean(errorSparse, axis=1)
    # print("Distance errors:")
    # print(errorSparse)
    # print(errorSparse.shape)
    # print("Mean errors:")
    # print(meanSparse)
    # print(meanSparse.shape)
    points = np.arange(K, step=20)
    plt.plot(points+1, meanGauss[points], 'k+')
    plt.plot(points+1, meanSparse[points], 'k*')
    plt.title('Error using RP, SRP')
    plt.show()


if __name__ == '__main__':
    imgoriginal()
    # imgtest()