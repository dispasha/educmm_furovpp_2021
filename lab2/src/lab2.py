import numpy as np
import matplotlib.pyplot as plt
import docx
from docx import Document


def fft_Ak(y_nodes):
    N = len(y_nodes)
    if N <= 1: return y_nodes
    E = fft_Ak(y_nodes[0::2])
    E = np.concatenate((E,E))
    O = fft_Ak(y_nodes[1::2])
    O = np.exp(-2j*np.pi*np.arange(N)/N)*np.concatenate((O,O))
    return E+O

def fft_coef(y_nodes):
    N=len(y_nodes)
    return (-1)**np.arange(N)/ (N+1)*fft_Ak(y_nodes)

def poly_regression(x_nodes, y_nodes, degree, l=0):
    X = np.vander(x_nodes, N=degree+1, increasing=True) 
    return np.linalg.inv(X.T @ X + l * np.identity(degree+1)) @ X.T @ y_nodes

def quad_error(orig, pred):
    return ((orig-pred).T @ (orig-pred))/len(orig)


def plot_regression(ax, x_nodes, y_nodes, fitting=None):
    global errors
    x_for_plotting = np.linspace(np.min(x_nodes), np.max(x_nodes),200)
    if fitting is not None:
        ax.plot(x_for_plotting, fitting, 'g', linewidth=3, color="red")
    ax.plot(x_nodes, y_nodes, 'bo', alpha=0.2, markersize=4)
    ax.tick_params(labelsize=16)
    ax.set_xlabel(r"$x$", fontsize=16)
    ax.set_ylabel(r"$y$", fontsize=16)
    


global errors

errors=[[],[]]

#обработка данных
data = np.loadtxt("data.txt", usecols=(1,2,3))[17:]
x_full = data[:,0]+(data[:,1]-1)/12.0
y_full = data[:,2]


#вывод данных
#fig,axes = plt.subplots(1, figsize=(13,8))
#plot_regression(axes, x_full, y_full)

#формирование x_train_valid
np.random.seed(1404)
data_rnd = data.copy()
np.random.shuffle(data_rnd)
m=len(data)//2
x_train = (data_rnd[:m,0]+(data_rnd[:m,1]-1)/12.0)
y_train = data_rnd[:m,2]
x_valid = (data_rnd[m:,0]+(data_rnd[m:,1]-1)/12.0)
y_valid = data_rnd[m:,2]

#вывод x_train_valid
'''
fig,axs = plt.subplots(2, figsize=(12,12))
axs[0].set_title("$D_{train}$",loc="left",fontsize=20)
axs[1].set_title("$D_{valid}$",loc="left",fontsize=20)
plot_regression(axs[0], x_train, y_train)
plot_regression(axs[1], x_valid, y_valid)'''

x_norm_train = 2*(x_train-np.min(x_train))/(np.max(x_train)-np.min(x_train))-1
x_norm_valid = 2*(x_valid-np.min(x_valid))/(np.max(x_valid)-np.min(x_valid))-1
x_norm_plot = np.linspace(-1,1,200)
print(x_norm_plot)
'''
fig,axes = plt.subplots(4,2, figsize=(15,8*2))
for ax, poly_degree in zip(axes.reshape(-1), (1,2,3,4,5,10,15,20)):
    a=poly_regression(x_norm_train, y_train, poly_degree)
    fitting=sum(a[i]*x_norm_plot**i for i in range(poly_degree+1))
    errors[0].append(quad_error(y_train,sum(a[i]*x_norm_train**i for i in range(poly_degree+1))))
    errors[1].append(quad_error(y_valid,sum(a[i]*x_norm_valid**i for i in range(poly_degree+1))))
    #plot_regression(ax, x_train, y_train, fitting)
    #ax.set_title(f'Степень полинома: {poly_degree}', fontsize=16)
'''


document=Document()
table = document.add_table(rows=len(errors[0]), cols=3)
for i in range(len(errors[0])):
    row=table.rows[i]
    row.cells[0].text = str(round(errors[0][i],3))
    row.cells[1].text = str(round(errors[1][i],3))

poly_degree=20
lambda_values=np.logspace(-9,1,100)
rss_values = np.zeros_like(lambda_values)
for i, lambda_ in enumerate(lambda_values):
    a=poly_regression(x_norm_train, y_train, poly_degree, lambda_)
    rss_values[i]=quad_error(y_valid,sum(a[i]*x_norm_valid**i for i in range(poly_degree+1)))
print(rss_values)
'''
fig, ax = plt.subplots(figsize=(13,6))
ax.semilogx(lambda_values, rss_values, linewidth=2)
ax.set_xlabel(r"$\lambda$", fontsize=20)
ax.set_ylabel(r"$e_{valid}^{(20)}$", fontsize=20)
ax.tick_params(labelsize=20)'''
#document.save('quads.docx')
#plt.ylim(5.5, 7.5)

'''
fig,axes = plt.subplots(2, figsize=(13,6))
ax.tick_params(labelsize=20)
for ax, lambda_ in zip(axes.reshape(-1), (0,lambda_values[np.argmin(rss_values)])):
    a=poly_regression(x_norm_train, y_train, poly_degree, lambda_)
    fitting=sum(a[i]*x_norm_plot**i for i in range(poly_degree+1))
    plot_regression(ax, x_valid, y_valid, fitting)
    ax.set_title(rf'$\lambda=${lambda_}', fontsize=16)'''
x_norm_full = 2*(x_full-np.min(x_full))/(np.max(x_full)-np.min(x_full))-1
a=poly_regression(x_norm_train, y_train, 3)
trend =sum(a[i]*x_norm_full**i for i in range(4))

y=y_full-trend
x=x_full
'''
fig,axes = plt.subplots(2, figsize=(13,8))
for ax in axes:
    ax.set_xlabel(r"$x$", fontsize=20)
    ax.set_ylabel(r"$y$", fontsize=20)
    ax.tick_params(labelsize=20)

axes[0].plot(x,y)
axes[1].plot(x[1:], np.diff(y_full))
axes[0].set_title(f'Вычитание тренда', fontsize=20)
axes[1].set_title(f'Взятие разности', fontsize=20)
'''

y_analyze=y[:128]
a_hat=fft_coef(y_analyze)
a_k = 2*np.real(a_hat)[:64]
b_k = -2*np.imag(a_hat)[:64]
'''
fig, (ax_cos, ax_sin)=plt.subplots(1,2,figsize=(14,6))
ax_cos.plot(range(len(a_k)), a_k, 'o--', linewidth=2)
ax_sin.plot(range(len(b_k)), b_k, 'o--', linewidth=2)
for ax in (ax_cos, ax_sin):
    ax.grid()
    ax.set_xticks(range(len(a_k)))
    ax.set_xlim((0,20))
    ax.set_xlabel(r'$k$', fontsize=20)

ax_cos.set_ylabel(r'$a_k$', fontsize=20)
ax_sin.set_ylabel(r'$b_k$', fontsize=20)
print(a_k[10]/(a_k[11]+a_k[10]), a_k[11]/(a_k[11]+a_k[10]))
print(b_k[10]/(b_k[11]+b_k[10]), b_k[11]/(b_k[11]+b_k[10]))
'''
x=x[:128]
y=y[:128]
x_graph = np.linspace(min(x),max(x),256)
x_norm = np.linspace(-np.pi,np.pi,256)
y_pred = a_k[11] * np.cos(10.6 * x_norm)
plt.plot(x,y,label="Исходные данные")
plt.plot(x_graph,y_pred,"orange",label="$a_5\cdot cos(10.6x)$")
plt.gca().set_xlabel("x",fontsize=11)
plt.gca().set_ylabel("y",fontsize=11,rotation=0,labelpad=8)
plt.legend(loc="upper left")


plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.savefig("img1.png", dpi=300)




