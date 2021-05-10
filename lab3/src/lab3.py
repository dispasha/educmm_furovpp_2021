import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import scipy.optimize as opt
import timeit as tit
import sympy as sp
import random

def f(t,y):
    return np.array([y[1],np.cos(t)-np.sin(y[0])-0.1*y[1]])
                    
def runge_kutta(x_0, t_n, f, h):
    m=int(t_n/h)+1
    w = np.zeros((m,2))
    w[0,:] = x_0[0]
    for i in range(1,m):
        k_1 = h*f(i*h,w[i-1])
        k_2 = h*f(i*h+h/2,w[i-1]+k_1/2)
        k_3 = h*f(i*h+h/2,w[i-1]+k_2/2)
        k_4 = h*f(i*h+h,w[i-1]+k_3)
        w[i] = w[i-1]+(k_1+2*k_2+2*k_3+k_4)/6
    return w


def adams_moulton(x_0, t_n, f, h, t_0 = 0):
    m=int(t_n/h)+1
    w = np.zeros((m,2))
    w[:3,:] = runge_kutta(x_0,h*3,f,h)[:3,:] 
    ft__0,ft__1,ft__2 = f(t_0,w[0]),f(t_0+h,w[1]),f(t_0+2*h,w[2])
    for i in range(3,m):
        f_right = w[i-1]+h/24*(19*ft__2-5*ft__1+ft__0)
        f_min = lambda x: x-h*3/8*f(t_0+i*h,x)-f_right
        w[i] = opt.root(f_min, w[i-1]).x
        ft__0,ft__1,ft__2 = ft__1,ft__2,f(t_0+i*h,w[i])
    return w

def solve_ints(j):
    s=sp.symbols('s')
    mul=1
    for k in range(1,5):
        if k==j: continue
        mul*=(s+k-2)/(k-j)
    return sp.integrate(mul, (s,0,1))

def milne_simpson(x_0, t_n, f, h):
    m=int(t_n/h)+1
    w = np.zeros((m,2))
    w[:4,:] = runge_kutta(x_0,h*4,f,h)[:4,:]
    ft_0,ft_1,ft_2 = f(h,w[1]),f(2*h,w[2]),f(3*h,w[3])
    for t in range(4,m):
        w_ = w[t-4]+4*h*(2*ft_0-ft_1+2*ft_2)/3
        w[t] = w[t-2]+h*(f(t*h,w_)+4*ft_2+ft_1)/3
        ft_0,ft_1,ft_2 = ft_1,ft_2,f(t*h,w[t])
    return w

#for j in range(1,5):
#    print(f'a{j}={solve_ints(j)}')
runge, adam, sim= False,False,False
t_n=100
h=0.1
np.random.seed(1404)
d0dt=np.random.uniform(1.85, 2.1, size=20)
t = np.arange(0, t_n+h, h)
if runge:
    fig, axes = plt.subplots ( figsize=(15,10))
    for i in d0dt:
        runge_res= runge_kutta([[0,i]],t_n,f,h)[:,0]
        axes.plot(t, runge_res, '-', linewidth = 1)
    axes.grid()
    axes.set_xlabel(r'$t$')
    axes.set_ylabel(r'$\pi\theta$')
    axes.set_title(r'Рунге-Кутта')
    many_zeros = np.linspace(-8 * np.pi, 7 * np.pi, 16)
    names = np.linspace(-8, 7, 16)
    names=names.astype(int)
    axes.set_yticks(many_zeros)
    axes.set_yticklabels(names)
    plt.savefig("vych.png", dpi=300)

if adam:
    fig, axes = plt.subplots ( figsize=(15,10))
    for i in d0dt:
        adams_res= adams_moulton([[0,i]],t_n,f,h)[:,0]
        axes.plot(t, adams_res, '-', linewidth = 1)
    axes.grid()
    axes.set_xlabel(r'$t$')
    axes.set_ylabel(r'$\pi\theta$')
    axes.set_title(r'Адамс-Моултон')
    many_zeros = np.linspace(-8 * np.pi, 11 * np.pi, 20)
    names = np.linspace(-8, 11, 20)
    names=names.astype(int)
    axes.set_yticks(many_zeros)
    axes.set_yticklabels(names)
    plt.savefig("vych1.png", dpi=300)

if sim:
    t_n=200
    t = np.arange(0, t_n+h, h)
    fig, axes = plt.subplots ( figsize=(15,10))
    for i in d0dt:
        sim_res= milne_simpson([[0,i]],t_n,f,h)[:,0]
        axes.plot(t, sim_res, '-', linewidth = 1)
    axes.grid()
    axes.set_xlabel(r'$t$')
    axes.set_ylabel(r'$\pi\theta$')
    axes.set_title(r'Милн-Симпсон')
    many_zeros = np.linspace(-8 * np.pi, 11 * np.pi, 20)
    names = np.linspace(-8, 11, 20)
    names=names.astype(int)
    axes.set_yticks(many_zeros)
    axes.set_yticklabels(names)
    plt.savefig("vych2.png", dpi=300)

if 0:
    d0dt=15
    h=np.arange(0.010,0.08,0.01)
    h=[0.1]
    fig, axes = plt.subplots ( figsize=(15,10))
    for i in h:    
        runge_res= adams_moulton([[0,d0dt]],t_n,f,i)[:,0]
        t = np.linspace(0, t_n+i, len(runge_res))
        axes.plot(t, runge_res, '-', linewidth = 1, label=f'h={round(i,4)}')
    axes.grid()
    #axes.legend(loc='lower right', fontsize=11)
    axes.set_xlabel(r'$t$')
    axes.set_ylabel(r'$\pi\theta$')
    axes.set_title(r'Начальная скорость 15')
    many_zeros = np.linspace(0 * np.pi, 47 * np.pi, 48)
    names = np.linspace(0, 47, 48)
    names=names.astype(int)
    axes.set_yticks(many_zeros)
    axes.set_yticklabels(names)
    plt.savefig("vych.png", dpi=300)

if 0:
    x = np.linspace(1,200,20)
    t_rk = np.zeros_like(x)
    t_am = np.zeros_like(x)
    t_ms = np.zeros_like(x)
    t_mam = np.zeros_like(x)

    for tn in range(len(x)):
        t = tit.default_timer()
        runge_kutta([[0,1.85]],x[tn],f,0.03)
        t_rk[tn] = tit.default_timer()-t

        t = tit.default_timer()
        adams_moulton([[0,1.85]],x[tn],f,0.218)
        t_am[tn] = tit.default_timer()-t

        t = tit.default_timer()
        milne_simpson([[0,1.85]],x[tn],f,0.196)
        t_ms[tn] = tit.default_timer()-t
        
        print("\r\r\r\r%3d%%"%(int(tn*100/(len(x)-1))),end="")
    plt.plot(x,t_rk,label = "Рунге-Кутта")
    plt.plot(x,t_am,label = "Адамс-Моултон")
    plt.plot(x,t_ms,label = "Милн-Симпсон")

    plt.xlabel("$t_{инт}$",fontsize=12)
    plt.ylabel("$t_{вып}$",rotation=0,fontsize=12)
    plt.gcf().set_size_inches(6, 4)
    plt.tight_layout(pad=0.5)
    plt.legend(loc = 2)

    plt.savefig("vych3.png", dpi=300)

if 0:
    ad = adams_moulton([[0, 1.85]], 100, f, 0.1)
    fig, ax = plt.subplots(1, 1, figsize=(14, 8)) 
    ax.plot(ad[:,0], ad[:,1], '-', markersize=2)
    ax.set_xlabel(r'$\theta$', fontsize=16)
    ax.set_ylabel(r'$d\theta/dt$', fontsize=16)

    ax.grid()
    plt.savefig("vych4.png", dpi=300)

if 0:
    random.seed(1404)
    colors = list(mcolors.CSS4_COLORS)
    np.random.shuffle(colors)
    y_tett = np.linspace(-12.56, 12.56, 5)
    y_dtett = np.linspace(-5.0, 5.0, 5)
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    for i in range(0, len(y_tett)):
        for j in range(0, len(y_dtett)):
            ad = adams_moulton([[y_tett[i], y_dtett[j]]], 200, f, 0.07)
            for l in range(len(ad)-2, 1, -1):
                if (abs(ad[-1, 0] - ad[l, 0]) < 0.05):
                    tim = l - (len(ad)-l)
                    break
            ax.plot(ad[tim:,0], ad[tim:,1], colors[round(ad[-1,0] / (2*np.pi))], '-', markersize=2)
            #ax.scatter(ad[0,0], ad[0,1], c = colors[round(ad[-1,0] / (2*np.pi))])
    many_zeros = np.linspace(-16 * np.pi, 16 * np.pi, 17)
    names = np.linspace(-16, 16, 17)
    names=names.astype(int)
    ax.set_xticks(many_zeros)
    ax.set_xticklabels(names)
    ax.set_xlabel(r'$\pi\theta$')
    ax.set_ylabel(r'$d\theta/dt$')
    ax.grid()
    plt.savefig("vych5.png", dpi=300)

if 1:
    random.seed(1404)
    colors = list(mcolors.CSS4_COLORS)
    np.random.shuffle(colors)
    y_tett = np.linspace(-12.56, 12.56, 121)
    y_dtett = np.linspace(-5.0, 5.0, 121)
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.set_xlabel(r'$\theta$', fontsize=16)
    ax.set_ylabel(r'$d\theta/dt$', fontsize=16)

    for i in range(0, len(y_tett)):
        for j in range(0, len(y_dtett)):
            ad = adams_moulton([[y_tett[i], y_dtett[j]]], 200, f, 0.07)
            ax.scatter(ad[0,0], ad[0,1], c = colors[round(ad[-1,0] / (2*np.pi))])
        plt.savefig("vych6.png", dpi=300)
    ax.grid()
    plt.savefig("vych6.png", dpi=300)
