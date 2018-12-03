from math import exp, sin, cos

def naive_Newton(f, dfdx, x, eps):
    while abs(f(x)) > eps:
        x = x - float(f(x))/dfdx(x)
    return x

def f(x):
    return (1 + x**2)*exp(-x)+sin(x)

def dfdx(x):
    return 2*exp(-x)*x-exp(-x)*(1 + x**2)+cos(x)

if __name__ == '__main__':

    res = naive_Newton(f, dfdx, 10, 0.001)
    print ("Result =", res)    
    print ("f(x) =", f(res))