import time
a = 1
b = -4.443
c = -9.696284
d = 36.03005106
e = -23.67284825

#원함수
def f(x) :
    return (a * x**4) + (b * x**3) + (c * x**2) + (d * x) + e
#미분함수
def ff(x) :
    return (4 * a * x**3) + (3 * b * x**2) + (2 * c * x) + d
#2번 미분함수
def fff(x) :
    return (12 * a * x**2) + (6 * b * x) + 2 * c

#초기값 설정
s = -2

start = time.time()
n = 0
err = 1000000

while err > 10**(-6) and fff(s) > 0:
    n = n + 1
    vf = ff(s)
    df = fff(s)

    # 새로운 점
    temp = s - vf/df
    err = abs(temp - s)
    print(n, "번 반복 | 초기값 x1 = ", s, ", 새로운 값 x2 = ", temp, ", err : ", err)
    s = temp

print("x = ", temp)
print("수행 시간 = ", time.time()-start)
