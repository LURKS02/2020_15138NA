import time
a, b, c, d, e = map(float, input("항 입력").split())

#원함수
def f(x) :
    return (a * x**4) + (b * x**3) + (c * x**2) + (d * x) + e
#미분함수
def ff(x) :
    return (4 * a * x**3) + (3 * b * x**2) + (2 * c * x) + d

#초기값 설정
s = 6

start = time.time()
n = 0
err = 1000000

while err > 10**(-6):
    n = n + 1
    vf = f(s)
    df = ff(s)

    # 새로운 점
    temp = s - vf/df
    err = abs(temp - s)
    print(n, "번 반복 | 초기값 x1 = ", s, ", 새로운 값 x2 = ", temp, ", err : ", err)
    s = temp

print("x = ", temp)
print("수행 시간 = ", time.time()-start)