import numpy as np
import matplotlib.pyplot as plt


# Реализация функций численного дифференцирование и вывод графиков,
# демонстрирующих погрешность численного метода дифференцирования


# Функция численного дифференцирования 1-го порядка
def diff1(x_0, h, f):
    return (f(x_0+h)-f(x_0))/h

# Функция численного дифференцирования 2-го порядка
def diff2(x_0, h, f):
    return (-3*f(x_0)+4*f(x_0+h)-f(x_0+2*h))/(2*h)

# Дифференцируемая функция
def g(x):
    return x*x*np.sin(x)

# Точное значение производной
def g_deriv(x):
    return 2*x*np.sin(x) + np.cos(x)*x**2

def abs_err(x, h, diff):
    return np.abs(g_deriv(x)-diff(x,h,g))

def g_derivv(x):
    return 2*np.sin(x) + np.cos(x)*x*4-np.sin(x)*x**2

def g_deriv3(x):
    return 6*np.cos(x) - np.sin(x)*x*6-np.cos(x)*x**2

def get_h_opt():
    return np.sqrt(4*np.finfo(float).eps/abs(g_derivv(2.98147))), (6*np.finfo(float).eps/abs(g_deriv3(4)))**(1/3)

def main():

    print(g_deriv3(2), g_deriv3(4), g_deriv3(2.02463))
    h2_opt=(6*np.finfo(float).eps/abs(g_deriv3(4)))**(1/3)
    print(h2_opt)

    '''
    print(g_derivv(2), g_derivv(3), g_derivv(2.98147))
    h1_opt=np.sqrt(4*np.finfo(float).eps/abs(g_derivv(2.98147)))
    print(h1_opt)
    '''
    x=2
    plt.rcParams.update({'font.size': 20})
    fig, axes = plt.subplots(1, 1, figsize=(13, 8))
    h = np.logspace(-16, 0, 150)
    h_for_sq = np.logspace(-4.5, 0, 100)
    h_for_lft = np.logspace(-15, -9, 100)

    #diff_1_line = axes.loglog(h, abs_err(x, h, diff1), 'o', label='diff1 error')
    #diff_1_opt = axes.loglog(h_opt, abs_err(x, h1_opt, diff1), 'o', label='diff1 opt error', color='red')

    diff_2_line = axes.loglog(h, abs_err(x, h, diff2), 'o', label='diff2 error', color='tab:orange')
    diff_2_opt = axes.loglog(h_opt, abs_err(x, h2_opt, diff2), 'o', label='diff2 opt error', color='red')
    axes.grid(linestyle='--')
    axes.set_xlabel(r'$h$', fontsize=20)
    axes.set_ylabel(r'$E$', fontsize=20)
    #axes.loglog(h_for_sq, 5*h_for_sq, '-', label=r'$O(h)$', color='tab:blue')
    axes.loglog(h_for_sq, 10*h_for_sq**2, '-', label=r'$O(h^2)$', color='tab:orange')
    axes.loglog(h_for_lft, 1.5e-15/h_for_lft, '-', label=r'$O(h^{-1})$', color='green')


    plt.legend(loc='lower left', fontsize=20)
    plt.tight_layout()
    plt.show()




