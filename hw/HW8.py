import cv2
import numpy as np

image = cv2.imread('./8-5.jpg', cv2.IMREAD_COLOR)
cv2.imshow('test image', image)
cv2.waitKey(0)

B = image[:, :, 0]
B = np.reshape(B, (1, B.shape[0]*B.shape[1]))
G = image[:, :, 1]
G = np.reshape(G, (1, G.shape[0]*G.shape[1]))
R = image[:, :, 2]
R = np.reshape(R, (1, R.shape[0]*R.shape[1]))

Y = (0.257 * R) + (0.504 * G) + (0.098 * B) + 16
V = (0.439 * R) - (0.368 * G) - (0.071 * B) + 128
U = -(0.148 * R) - (0.291 * G) + (0.439 * B) + 128


def getCorrelation(Y, K):

    Ymean = np.mean(Y)
    Kmean = np.mean(K)
    Ymin = Y - Ymean
    Kmin = K - Kmean

    Ysum = 0
    Ksum = 0

    for i in range(Ymin.shape[1]):
        Ysum = Ysum + Ymin[0][i] * Ymin[0][i]
    for i in range(Kmin.shape[1]):
        Ksum = Ksum + Kmin[0][i] * Kmin[0][i]
    Ysum = np.divide(Ysum, Ymin.shape[1])
    Ksum = np.divide(Ksum, Kmin.shape[1])
    Ysum = np.sqrt(Ysum)
    Ksum = np.sqrt(Ksum)

    cov = np.divide(np.dot(Ymin, Kmin.T), Ymin.shape[1])
    return np.divide(cov, Ysum * Ksum)

correlR = getCorrelation(Y, R)
correlG = getCorrelation(Y, G)
correlB = getCorrelation(Y, B)

print(" Y-R = ", correlR, "\n Y-G = ", correlG, "\n Y-B = ", correlB)
