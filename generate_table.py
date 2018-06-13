import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

def get_color(bi, gi, ri):
    """ Get a set of colors whose discrete indices are the same
    Arguments:
        bi, gi, ri: blue, green, red index
    Return:
        D = [w1,...,w512], w: color
    """
    D = np.zeros((512, 3))

    windex = 0
    for b in range(bi*8, (bi+1)*8):
        for g in range(gi*8, (gi+1)*8):
            for r in range(ri*8, (ri+1)*8):
                D[windex, ...] = [b, g, r]
                windex += 1
    
    return D / 255.0


def knn(K, w, Z):
    """ K-NN
    Arguments:
        K: number of nearest neighbors
        w: color
        Z: salient color names
    Return:
        K nearest neighbors of w from Z
    """
    distances = norm(Z-w, axis=1)
    k_neighb = np.argsort(distances)[:K]

    return Z[k_neighb]


def pzw(z, w):
    z_knnkw = knn(K, w, Z)
    if z in z_knnkw:
        numerator_bot = 0.0
        for zl in z_knnkw:
            if np.array_equal(zl, z) is False:
                numerator_bot += np.square(norm(zl-w))
        numerator_bot /= (K-1)
        numerator = np.exp(- np.square(norm(z-w)) / numerator_bot)

        denominator = 0.0
        for zp in z_knnkw:
            denominator_bot = 0.0
            for zs in z_knnkw:
                if np.array_equal(zs, zp) is False:
                    denominator_bot += np.square(norm(zs-w))
            denominator_bot /= (K-1)
            denominator += np.exp(- np.square(norm(zp-w)) / denominator_bot)

        return numerator / denominator
    else:
        return 0


def pwd(w, D):
    mu = np.mean(D, axis=0)
    numerator = np.exp(- ALPHA * np.square(norm(w - mu)))
    denominator = np.sum(np.exp(-ALPHA * np.square(norm(D-mu, axis=1))))
    
    return numerator / denominator


def pzd(z, D):
    # The same as paper, but really slow
    # p = 0.0
    # for w in D:
        # p += pzw(z, w) * pwd(w, D)

    mu = np.mean(D, axis=0)
    p = 512 * pzw(z,mu) * pwd(mu, D)

    return p


K = 3
ALPHA = 1.
# Salient colour names
Z = [[255,0,0], [255,255,0], [0,255,0], [0,255,255], [0,0,255], [255,0,255],
     [128,0,0], [128,128,0], [0,128,0], [0,128,128], [0,0,128], [128,0,128],
     [0,0,0], [128,128,128], [192,192,192], [255,255,255]]
Z = np.array(Z) / 255.0


if __name__ == '__main__':
    
    P = np.zeros((32,32,32,16))
    for bi in tqdm(range(32)):
        for gi in range(32):
            for ri in range(32):
                for zi, z in enumerate(Z):
                    D = get_color(bi, ri, gi)
                    P[bi, gi, ri, zi] = pzd(z, D)


    np.save("P_mat.npy", P)


    # P = np.load("P_mat.npy")

    # test_D = get_color(0,0,0)
    # test = 0.0
    # for z in Z:
    #     # test += pzd(z, test_D)
    #     print(pzd(z, test_D))

    # print(test)