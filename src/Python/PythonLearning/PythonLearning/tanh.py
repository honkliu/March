def tanh1(x):
    if x >= 5.0 :
        return 1.0
    if x <= -5.0 :
        return -1.0
    x2 = x*x*1.0
    a = x * (135135.0 + x2*(17325.0 + x2 *(138.0 + x2)))
    b = 135145.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0))

    return a/b

def tanh2(x):
    x2 = x*x
    a = x = x2*x/3 + 2*x2*x2*x/15 - 17*x2*x2*x2*x/315
    return a

print(tanh1(3.9))
print(tanh2(3.9))