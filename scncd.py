import cv2
import numpy as np
from matplotlib import pyplot as plt
import time


class SCNCD():
    def __init__(self):
        self.table = np.load("/home/peng/Documents/scncd/P_mat_5.npy")
        
    def compute(self, img):
        """ Compute weighted salient color names feature descriptor.
        Param:
            img: input image in [b,g,r] format (the same as opencv)
        Return:
            16 dimensional color name feature descriptor.
        """
        descriptor = np.zeros(16)
        row = np.arange(img.shape[0])
        col = np.arange(img.shape[1])
        col, row = np.meshgrid(col, row)
        mu_row = img.shape[0]/2.2
        mu_col = img.shape[1]/2.
        sigma_row = img.shape[0]/8.
        sigma_col = img.shape[1]/8.
        w_matrix = np.exp(- (((row-mu_row)/sigma_row)**2 + ((col-mu_col)/sigma_col)**2) / 2)
        
        discrete_index = img / 8
        img_cn = np.zeros((img.shape[0], img.shape[1], 16))
        img_cn[:, :, ...] = self.table[discrete_index[:,:,0], discrete_index[:,:,1], discrete_index[:,:,2], ...]
        descriptor = np.sum(w_matrix.reshape((-1,1)) * img_cn.reshape((-1,16)), axis=0)
        descriptor /= np.linalg.norm(descriptor)

        return descriptor

    def color_names(self):
        """ Return the salient color names.
        """
        # Salient colour names
        Z = [[255,0,0], [255,255,0], [0,255,0], [0,255,255], [0,0,255], [255,0,255],
            [128,0,0], [128,128,0], [0,128,0], [0,128,128], [0,0,128], [128,0,128],
            [0,0,0], [128,128,128], [192,192,192], [255,255,255]]
        return (np.array(Z) / 255.0).tolist()

    def visualize_feat(self, feat):
        """ Visualize the color name descriptor in bar plot.
        """
        colours = [color[::-1] for color in self.color_names()]
        x = np.arange(16)
        plt.bar(x, height=feat, color=colours, edgecolor=(0,0,0))

    def visualize_weight(self, img):
        """ Visualize the weighted matrix when computing scncd.
        """
        # start_time = time.time()

        mu_row = img.shape[0]/2.2
        mu_col = img.shape[1]/2.
        sigma_row = img.shape[0]/6
        sigma_col = img.shape[1]/6
        row = np.arange(img.shape[0])
        col = np.arange(img.shape[1])
        col, row = np.meshgrid(col, row)
        w_matrix = np.exp(- (((row-mu_row)/sigma_row)**2 + ((col-mu_col)/sigma_col)**2) / 2)

        # clock_time = time.time() - start_time
        # print("Calculate Weight matrix time: {} sec.".format(clock_time))

        plt.imshow(w_matrix, cmap='winter')

        return w_matrix
        


# scncd = SCNCD()
# img = cv2.imread("person.png")
# feat = scncd.compute(img)
# plt.subplot(231), plt.imshow(img[:,:,::-1])
# plt.subplot(232), scncd.visualize_feat(feat)

# plt.subplot(233), plt.imshow(scncd.visualize_weight(img))

# img2 = cv2.imread("person2.png")
# feat2 = scncd.compute(img2)
# plt.subplot(234), plt.imshow(img2[:,:,::-1])
# plt.subplot(235), scncd.visualize_feat(feat2)
# plt.subplot(236), plt.imshow(scncd.visualize_weight(img2))

# cosine = np.dot(feat, feat2)
# print(cosine)
# plt.show()

# size = 10
# sigma_x = 20
# sigma_y = 20

# x = np.linspace(-10, 10, size)
# y = np.linspace(-10, 10, size)
# x, y = np.meshgrid(x, y)
# print(x)
# z = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-(x**2/(2*sigma_x**2)
#      + y**2/(2*sigma_y**2))))
# print(z.shape)
# # plt.contourf(x, y, z, cmap='Blues')
# plt.imshow(z, cmap='winter')
# # plt.colorbar()
# plt.show()

# z = np.arange(64*34).reshape((64,34))
# plt.imshow(z, cmap='winter')
# plt.show()