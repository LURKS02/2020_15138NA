import time
a, b, c, d, e = map(float, input("항 입력").split())

n = 0
def f(x) :
    return (a * x**4) + (b * x**3) + (c * x**2) + (d * x) + e

print(f(-1000))
print(f(1000))

while True:
    #근의 범위 지정
    s1 = -1000
    s2 = 1000
    if f(s1)*f(s2) < 0 :
        p = s1
        q = s2
        break

#두 추측값의 중간값
mid = p + (q-p)/2

err = abs(f(mid))
start = time.time()


while err > 10**(-6):
    n = n + 1
    print(n, "번 반복 | x1 = ", p, ", x2 = ", q, " | f(x1) = ", f(p), ", f(x2) = ", f(q), " err : ", err)

    # p ~ mid 사이에 근이 있으므로 q를 mid로 변경
    if f(mid)*f(p) < 0:
        q = mid
    # mid ~ q 사이에 근이 있으므로 p를 mid로 변경
    else :
        p = mid
    mid = p + (q-p)/2
    err = abs(f(mid))

print("x = ", mid)
print("수행 시간 = ", time.time()-start)