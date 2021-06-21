import math

fact = math.factorial
choose = math.comb


def s2(n, k):
    summation = 0
    for i in range(k + 1):
        summation += (-1)**i * choose(k, i) * (k - i)**n
    return (1/fact(k)) * summation


def bastard(E, T):
    total = 0
    for k in range(1, E):
        total += choose(E, k) * s2(2*T, k)
    return total
