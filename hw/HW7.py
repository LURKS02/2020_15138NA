import cv2
import numpy as np
import math
import time

DCTSize = 16
coeffSize = 64

def large_coeffs(dct):
    abs_image = np.abs(dct.flatten())
    sort_image = sorted(abs_image, reverse=True)
    index = sort_image[coeffSize]

    large_coeffs_image = dct
    large_coeffs_image[abs(large_coeffs_image) < index] = 0

    return large_coeffs_image


def DCT(image, row_start, row_end, col_start, col_end):
    for ch in range(3):
        block_image = image[row_start:(row_end + 1), col_start:(col_end + 1), ch]
        #block_image = np.float32(block_image) / 255.0

        dct_image = np.zeros((block_image.shape[0], block_image.shape[1]))

        for v in range(DCTSize):
            for u in range(DCTSize):
                sum = 0
                for y in range(DCTSize):
                    for x in range(DCTSize):
                        Sxy = block_image[y][x]
                        cos1 = v * np.pi * np.divide((2 * y + 1),(2 * DCTSize))
                        cos1 = np.cos(cos1)
                        cos2 = u * np.pi * np.divide((2 * x + 1),(2 * DCTSize))
                        cos2 = np.cos(cos2)
                        sum = sum + Sxy * cos1 * cos2
                Cv = 1
                Cu = 1
                if (v == 0):
                    Cv = np.divide(1, np.sqrt(2))
                if (u == 0):
                   Cu = np.divide(1, np.sqrt(2))
                dct_image[v][u] = np.divide(2, DCTSize) * Cv * Cu * sum


        #dct_image = cv2.dct(np.float32(block_image) / 255.0)
        large_coeffs_dct_image = large_coeffs(dct_image)
        idct_image = np.zeros((block_image.shape[0], block_image.shape[1]))

        for x in range(DCTSize):
            for y in range(DCTSize):
                sum = 0
                for v in range(DCTSize):
                    for u in range(DCTSize):

                        Cv = 1
                        Cu = 1
                        if (v == 0):
                            Cv = np.divide(1, np.sqrt(2))
                        if (u == 0):
                            Cu = np.divide(1, np.sqrt(2))
                        Fvu = large_coeffs_dct_image[v][u]
                        cos1 = v * np.pi * np.divide((2 * y + 1),(2 * DCTSize))
                        cos1 = np.cos(cos1)
                        cos2 = u * np.pi * np.divide((2 * x + 1),(2 * DCTSize))
                        cos2 = np.cos(cos2)
                        sum = sum + Cv * Cu * Fvu * cos1 * cos2


                idct_image[y][x] = np.divide(2, DCTSize) * sum

        #idct_image = cv2.idct(top_image) * 255

        #idct_image = idct_image * 255

        idct_image[idct_image > 255] = 255
        idct_image[idct_image < 0] = 0
        image[row_start:(row_end+1), col_start:(col_end+1), ch] = idct_image

    return image


fn1 = '2.jpg'
img1 = cv2.imread(fn1, cv2.IMREAD_COLOR)

print(img1.shape)
row_size = np.shape(img1)[0]
col_size = np.shape(img1)[1]

row_count = math.ceil(row_size / DCTSize)
col_count = math.ceil(col_size / DCTSize)

print(row_count, col_count)
Max_size = 0

if (row_count >= col_count):
    Max_size = row_count * DCTSize

elif (row_count <= col_count):
    Max_size = col_count * DCTSize

Result = np.zeros((Max_size, Max_size, 3))

for i in range(0, 3):
    for j in range(0, col_size):
        for k in range(0, row_size):
            Result[k][j][i] = img1[k][j][i]

start = time.time()
for i in range(row_count):
    for j in range(col_count):
        for ch in range(3):
            row_start = 0 + i * DCTSize
            row_end = (DCTSize - 1) + i * DCTSize
            col_start = 0 + j * DCTSize
            col_end = (DCTSize - 1) + j * DCTSize

            block = DCT(Result, row_start, row_end, col_start, col_end)

            Result[row_start:row_end + 1, col_start:col_end + 1, ch] = block[row_start:row_end + 1, col_start:col_end + 1, ch]
            print(i, j)


img1[:row_size, :col_size] = Result[:row_size, :col_size]
resize_img = cv2.resize(img1, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_AREA)
print("수행 시간 : ", time.time() - start)
cv2.imshow('image', resize_img)
cv2.waitKey(0)