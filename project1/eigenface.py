import matplotlib
import numpy as np
import prettytable as pt
import ipykernel
import matplotlib.pyplot as plt
import operator

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 75

plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 14

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"
plt.rcParams['text.latex.preamble'] = "\\usepackage{subdepth}, \\usepackage{type1cm}"

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from time import time
from sklearn.datasets import fetch_lfw_people


dataset = fetch_lfw_people(min_faces_per_person=10, resize=0.3)

n_images = dataset.images.shape[0]
height = dataset.images.shape[1]
width = dataset.images.shape[2]
j = 0
list = []
for i in (dataset.target):
    if (i == 22):
        list.append(j)
    j = j + 1

X = dataset.data
print(X.shape)

dimension = X.shape[1]

def plot_faces(images, n_row=2, n_col=5):
    plt.figure(figsize=(1.5 * n_col, 2.2 * n_row))
    plt.subplots_adjust(0.6, 0.5, 1.5, 1.5)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((height, width)), cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
    plt.tight_layout()
    plt.show()

def plot_face(image, n_row=1, n_col=1):
    plt.figure(figsize=(1.5*n_col, 2.2*n_row))
    plt.subplots_adjust(0.6,0.5,1.5,1.5)
    plt.subplot(n_row,n_col,1)
    plt.imshow(image.reshape((height,width)), cmap=plt.cm.gray)
    plt.xticks(())
    plt.yticks(())
    plt.tight_layout()
    plt.show()

mean = np.zeros((1, dimension))
for i in X:
    mean = np.add(mean, i)
mean = np.divide(mean, float(n_images)).flatten()
mean = mean.reshape(1,dimension)


X1 = [2225, 227, 1343, 803, 2400]
X2 = [4110, 1636, 216, 116, 121]
X3 = [315, 560, 836, 1035, 1172]
X4 = [136, 293, 322, 390, 434]
X5 = [20, 172, 473, 742, 898]
X6 = [335, 466, 563, 833, 908]
X7 = [9, 73, 692, 703, 967]
X8 = [472, 493, 497, 614, 854]
X9 = [16, 304, 359, 620, 886]
X10 = [1715, 1716, 1892, 2403, 2502]

X = np.subtract(X, mean)

# 평균 + 일반 사진 보여주기


X_train, X_test = train_test_split(X, test_size=0.25, random_state=42)

n_components = 10

t0 = time()

pca = PCA(n_components=n_components, svd_solver='randomized')
pca.fit(X)

eigenfaces = pca.components_

#plot_faces(eigenfaces[:10])

'''
minus = X[:10]
minus = minus - mean
plot_faces(minus)

c= X[0]-mean
c= np.dot(c, eigenfaces[0])
c = c * eigenfaces[0]
c = 0.1 * eigenfaces[0] + 1* eigenfaces[4] - mean
'''

def getCoefficients (image, mean, vectors):
    coeff = []
    normalised_vec = image
    for i in range(n_components):
        coeffval = np.dot(normalised_vec, vectors[i])
        coeff.append(coeffval)
    return coeff

coarr = []
for i in range(X.shape[0]):
    coeffs = getCoefficients(X[i], mean, eigenfaces)
    coarr.append(coeffs)


for i in (X5):
    print(np.round(coarr[i],2))



arr = []
for i in range(290,300):
    generate_face = np.zeros((1, X.shape[1]))
    for j in range (n_components):
        generate_face = generate_face + np.dot(coarr[i][j], eigenfaces[j])
    arr.append(generate_face)

plot_faces(arr + mean)
