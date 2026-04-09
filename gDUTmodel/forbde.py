#%%#%%
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.multitest as ssm
import scipy as sp
import pickle
import sys
import os
import re
import matplotlib.cm as cm
from matplotlib.pyplot import gcf

# %%
plt.figure(figsize=(8,5))
k_list = [i for i in range(0,13)]
n=100
t=100
y = [(4**k)*k*n*t for k in k_list]
g = sns.lineplot(k_list, y)
g.set_xticks(range(13))
plt.title("Brute Force Algorithm")
plt.xlabel("k")
plt.ylabel("time complexity")
plt.ticklabel_format(style='plain', axis='y',useOffset=False)

plt.show()
# %%



# %%
plt.figure(figsize=(5,8))
k=12
n=100
t_list = [i for i in range(1,300)]
n_list = [i for i in range(1,300)]

y = [(n*k+k*t)*t*(n-k+1) for t in t_list]
g = sns.lineplot(t_list, y)
plt.title("Greedy Algorithm")
plt.ticklabel_format(style='plain', axis='y',useOffset=False)
plt.xlabel("t")
plt.ylabel("time complexity")
plt.show()
# %%
# %%
plt.figure(figsize=(5,8))
k=12
t=100
t_list = [i for i in range(1,300)]
n_list = [i for i in range(1,300)]

y = [(n*k+k*t)*t*(n-k+1) for n in n_list]
g = sns.lineplot(n_list, y)
plt.title("Greedy Algorithm")
plt.ticklabel_format(style='plain', axis='y',useOffset=False)
plt.xlabel("n")
plt.ylabel("time complexity")
plt.show()
# %%
plt.figure(figsize=(5,8))
k=12
t=100
t_list = [i for i in range(1,300)]
n_list = [i for i in range(1,300)]
N=100
y = [N*k*n*t for n in n_list]
g = sns.lineplot(n_list, y)
plt.title("Monte Carlo Algorithm with 100 iterations")
plt.ticklabel_format(style='plain', axis='y',useOffset=False)
plt.xlabel("n")
plt.ylabel("time complexity")
plt.show()
# %%
# %%
plt.figure(figsize=(8,8))
k=12
t=100
t_list = [i for i in range(10,100)]
n_list = [i for i in range(10,100)]
N=50
sns.lineplot(n_list, [10000*k*n*t for n in n_list])
sns.lineplot(n_list, [100*100*k*n*t for n in n_list])
plt.legend(labels=['Monte Carlo Algorithm', 'Gibbs Sampler Algorithm'])
plt.title("Monte Carlo & Gibbs Sampler algorithm with 100 iterations")
plt.ticklabel_format(style='plain', axis='y',useOffset=False)
plt.xlabel("n")
plt.ylabel("time complexity")
plt.show()
# %%
n=100
k=12
t=100
print((n*k+k*t)*t*(n-k+1))
# %%


plt.figure(figsize=(8,5))

n=100
t=100
N=100
k=12
n_list = [i for i in range(1,200)]
k_list = [i for i in range(6,13)]
#sns.lineplot(k_list, [(4**k)*k*n*t for k in k_list], palette=['r'])
sns.lineplot(k_list, [(n*k+k*t)*t*(n-k+1) for n in n_list])
#sns.lineplot(k_list, [1000*k*n*t for k in k_list])
sns.lineplot(k_list, [100*N*k*n*t for n in n_list])
plt.legend(labels=['Greedy Algorithm', 'Gibbs Sampler Algorithm'])
plt.ticklabel_format(style='plain', axis='y',useOffset=False)
plt.xlabel("k")

plt.ylabel("time complexity")
#plt.title("time complexity of 2 algorithms with fixed n & t ")
plt.show()

# %%
