import numpy as np
import matplotlib.pyplot as plt
from numerical import diff1, diff2, get_h_opt

def f_deriv(x):
    return(4*x**5 + 16*x**4 + 10*x**3 - 14*x**2 + 16*x + 5) / ((x + 2)**2)

class AutoDiffNumber:
    def __init__(self, a, b):
        self.a=a
        self.b=b
    def __add__(self, other):
        if isinstance(other, AutoDiffNumber):
            return AutoDiffNumber(self.a+other.a,self.b+other.b)
        else:
            return AutoDiffNumber(self.a+other,self.b)
        
    def __mul__(self, other):
        if isinstance(other, AutoDiffNumber):
            return AutoDiffNumber(self.a*other.a,self.b*other.a+self.a*other.b)
        else:
            return AutoDiffNumber(self.a*other,self.b*other)
        
    def __sub__(self, other):
        if isinstance(other, AutoDiffNumber):
            return AutoDiffNumber(self.a-other.a,self.b-other.b)
        else:
            return AutoDiffNumber(self.a-other,self.b)

    def __truediv__(self, other):
        if isinstance(other, AutoDiffNumber):
            return AutoDiffNumber(self.a/other.a,((self.b*other.a-self.a*other.b)/other.a**2))
        else:
            return AutoDiffNumber(self.a/other,self.b/other)
        
    def __radd__(other, self):
        if isinstance(other, AutoDiffNumber):
            return AutoDiffNumber(self.a+other.a,self.b+other.b)
        else:
            return AutoDiffNumber(self.a+other,self.b)
        
    def __rmul__(self, other):
        if isinstance(other, AutoDiffNumber):
            return AutoDiffNumber(self.a*other.a,self.b*other.a+self.a*other.b)
        else:
            return AutoDiffNumber(self.a*other,self.b*other)
        
    def __rsub__(other, self):
        if isinstance(other, AutoDiffNumber):
            return AutoDiffNumber(self.a-other.a,self.b-other.b)
        else:
            return AutoDiffNumber(self.a-other,self.b)

    def __rtruediv__(other, self):
        if isinstance(other, AutoDiffNumber):
            return AutoDiffNumber(self.a/other.a,((self.b*other.a-self.a*other.b)/other.a**2))
        else:
            return AutoDiffNumber(self.a/other,self.b/other)    
        
    def __pow__(self, p):
        return AutoDiffNumber(self.a**p,(self.b*p*self.a**(p-1)))

def f(x):
    return (x**5 + 2*x**4 - 3*x**3 + 4*x**2 - 5)/(x + 2)
        
def forward_autodiff(fun_args):
    dual = AutoDiffNumber(fun_args[1], 1)
    f_dual = fun_args[0](dual)
    return(f_dual.b)

h1_opt, h2_opt=get_h_opt()
h1_opt, h2_opt=10e-16, 10e-16
h1_opt, h2_opt=10e-2, 10e-2

points = np.random.uniform(low=-1, high=1, size=(100,))
points.sort()

autodiff = []
deriv = []
diff1_p = []
diff2_p = []


for i in range(0, 100):
    autodiff.append(forward_autodiff([f, points[i]]))
    deriv.append(f_deriv(points[i]))
    diff1_p.append(diff1(points[i], h1_opt, f))
    diff2_p.append(diff2(points[i], h2_opt, f))
    
fig, ax = plt.subplots(1, 1, figsize=(13, 8)) 
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(16)
plt.rc('grid', linestyle="--")
ax.plot(points, autodiff, '-', label = 'autodiff')
#ax.plot(points, deriv, '--', label = 'deriv')
ax.plot(points, diff1_p, '--', label = f'{h1_opt}')
ax.plot(points, diff2_p, '+', label = f'{h2_opt}')
ax.grid()
ax.set_xlabel(r'$x$', fontsize = 15)
ax.set_ylabel(r'$f$', fontsize = 20)
ax.legend(loc='upper left', borderaxespad=0.1, fontsize=18)
plt.tight_layout()
plt.show()
