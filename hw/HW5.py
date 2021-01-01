import pandas as pd
from matplotlib import pyplot
from scipy.optimize import curve_fit
from numpy import arange
import numpy as np
from numpy.linalg import inv

'''
dataframe = pd.DataFrame([[-2.9,35.4],[-2.1,19.7],[-0.9,5.7],[1.1,2.1],[0.1,1.2]
                          ,[1.9,8.7],[3.1,25.7],[4.0,41.5]])
data1 = pd.DataFrame([[-2.9,35.4],[-2.1,19.7],[-0.9,5.7],[1.1,2.1],[0.1,1.2]
                          ,[1.9,8.7]]).values
data2 = pd.DataFrame([[-0.9,5.7],[1.1,2.1],[0.1,1.2]
                          ,[1.9,8.7],[3.1,25.7],[4.0,41.5]]).values

x1 = data1[:,0]
y1 = data1[:,1]
x2 = data2[:,0]
y2 = data2[:,1]

def objective(x, a, b, c):
    return (a * x) + (b * x**2) + c

popt, _ = curve_fit(objective, x1, y1)
a, b, c = popt
print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))
pyplot.scatter(x1, y1)

#일정 범위 내에서 일정 간격으로 값을 반환
#start값은 포함, stop값은 제외

x_line = arange(-5, 5, 0.1)
#x_line = arange(min(x), max(x), 0.1)
y_line = objective(x_line, a, b, c)
pyplot.plot(x_line, y_line, '--', color='red')

popt, _ = curve_fit(objective, x2, y2)
a, b, c = popt
print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))
pyplot.scatter(x2, y2)

#일정 범위 내에서 일정 간격으로 값을 반환
#start값은 포함, stop값은 제외

x_line = arange(-5, 5, 0.1)
#x_line = arange(min(x), max(x), 0.1)
y_line = objective(x_line, a, b, c)
pyplot.plot(x_line, y_line, '--', color='blue')
pyplot.show()
'''


# x = [-2.9, -2.1, -0.9, 1.1, 0.1, 1.9, 3.1, 4.0]
x = [-0.9, 1.1, 0.1, 1.9, 3.1, 4.0]
y = np.array([[35.4],[19.7],[5.7],[2.1],[1.2],[8.7],[25.7],[41.5]])
yarr = np.array([[5.7],[2.1],[1.2],[8.7],[25.7],[41.5]])

a = []
b = []
c = []
for i in range(6):
    a.append(x[i]*x[i])
    b.append(x[i])
    c.append(1)

A = np.array([[a[0],b[0],c[0]],[a[1],b[1],c[1]],[a[2],b[2],c[2]],[a[3],b[3],c[3]],
                [a[4],b[4],c[4]],[a[5],b[5],c[5]]])

print("matrix A : \n", A)

print()

transposeA = A.T
arr2 = np.dot(transposeA, A)
arr2inv = inv(arr2)
arr3 = np.dot(arr2inv, transposeA)

print("pseudo-inverse of A : \n", arr3)

print()

finalarr = np.dot(arr3, yarr)

def f1(x):
    return 3.29527748*x**2 - 2.23383154*x + 0.95240824
def f2(x):
    return 3.02711584*x**2 - 1.89923592*x + 1.32758734

print("x : \n", finalarr)

print()

sum1 = (35.4 - f1(-2.9))**2 + (19.7 - f1(-2.1))**2
sum2 = (35.4 - f2(-2.9))**2 + (19.7 - f2(-2.9))**2
print("g1 [x = -2.9, -2.1] error = ", sum1,
      "\ng2 [x = -2.9, -2.1] error = ", sum2)

print()

sum1 = (25.7 - f1(3.1))**2 + (41.5 - f1(4.0))**2
sum2 = (25.7 - f2(3.1))**2 + (41.5 - f2(4.0))**2
print("g1 [x = 3.1, 4.0] error = ", sum1,
      "\ng2 [x = 3.1, 4.0] error = ", sum2)


