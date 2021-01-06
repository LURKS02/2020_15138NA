from __future__ import print_function
import numpy as np
from scipy import io
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import cv2
from time import time
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.datasets import fetch_lfw_people
import operator

mat_file = io.loadmat('C://Users/dihye/PycharmProjects/NA/project1/YaleB_32x32.mat')
print(mat_file)
people = fetch_lfw_people(min_faces_per_person=60,resize=0.3,color=False)

images = people.images
print(images.shape)
images_vec = images.reshape(images.shape[0], images.shape[1]*images.shape[2])
print(images_vec.shape)

mean = np.zeros((1, images.shape[1]*images.shape[2]))
for i in images_vec:
    mean = np.add(mean, i)
mean = np.divide(mean, float(images.shape[0])).flatten()
print(mean.shape)

normalised_faces = np.ndarray(shape=(images_vec.shape[0], images_vec.shape[1]))
for i in range(images_vec.shape[0]):
    normalised_faces[i] = np.subtract(images_vec[i], mean)
normalised_faces = normalised_faces.T

covariance = np.cov(normalised_faces)
covariance = np.divide(covariance, normalised_faces.shape[0])

pca = PCA(n_components=normalised_faces.shape[0])
pca.fit(covariance)

eig_vals = pca.explained_variance_
eig_vecs = pca.components_


eig_pairs = [(eig_vals[index], eig_vecs[:,index]) for index in range(len(eig_vals))]
sorted(eig_pairs, key=operator.itemgetter(0), reverse=True)

sort_eigvals = [eig_pairs[index][0] for index in range(len(eig_vals))]
sort_eigvecs = [eig_pairs[index][1] for index in range(len(eig_vals))]

#상위 30개의 eigenfaces
reduced_vec = np.array(sort_eigvecs[:30])
print(reduced_vec.shape)

def getCoefficients (image, mean, vectors):
    coeff = []
    normalised_vec = image - mean
    for i in range(30):
        coeffval = np.dot(normalised_vec, vectors[i])
        coeff.append(coeffval)
    return coeff

coeffs = getCoefficients(images_vec[80], mean, reduced_vec)
print(coeffs)

generate_face = np.zeros((1, images_vec.shape[1]))
for i in range (30):
    generate_face = generate_face + coeffs[i] * reduced_vec[i]
generate_face = generate_face + mean

test_image_show = generate_face.reshape(images.shape[1], images.shape[2])
plt.imshow(test_image_show, cmap='gray')
plt.show()


'''
test_image = images_vec[0]
test_image = normalised_faces[0]
test_image_show = test_image.reshape(images.shape[1], images.shape[2])
plt.imshow(test_image_show, cmap='gray')
plt.show()
'''




#mat_image = mat_vector.reshape(mat_vector.shape[0], 32, 32).astype(np.uint8)


#showImage = mean.reshape(images.shape[1], images.shape[2]).astype(np.uint8)
#showImage = normalised_faces[600].reshape(images.shape[1], images.shape[2])
#showImage = images[600]
#showImage = mean.reshape(images.shape[1], images.shape[2])
#plt.imshow(showImage, cmap='gray')
#plt.show()


#resizeImage = cv2.resize(showImage, None, None, 10, 10, cv2.INTER_CUBIC)
#cv2.imshow('image', resizeImage)
#cv2.waitKey(0)

'''
cv2.imshow('image',loadImage)
cv2.waitKey(0)'''



