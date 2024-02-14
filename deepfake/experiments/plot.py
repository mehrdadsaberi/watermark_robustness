from math import erf 
import numpy as np 
import json 
import matplotlib
from matplotlib import pyplot as plt 
from six.moves import cPickle as pkl
import seaborn as sns 
sns.set_theme()

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

task = "_task=deepfake"
model = "_vgg"
L = 2
with open("results/result_L={}{}_10{}.pkl".format(L, task, model), "rb") as f:
    auroc7, std_at_alpha7, std7 = pkl.load(f)

task = "_task=faceswap"
with open("results/result_L={}{}_10{}.pkl".format(L, task, model), "rb") as f:
    auroc6, std_at_alpha6, std6 = pkl.load(f)

print("Loaded")
    
plt.figure(dpi=200)
plt.plot(std7, np.mean(std_at_alpha7, 0), label="DeepFakes", color='C0')
plt.fill_between(std7, np.mean(std_at_alpha7, 0) + np.std(std_at_alpha7, 0), np.mean(std_at_alpha7, 0) - np.std(std_at_alpha7, 0), color='C0', alpha=0.3)
plt.plot(std6, np.mean(std_at_alpha6, 0), label="FaceSwap", color='C1')
plt.fill_between(std6, np.mean(std_at_alpha6, 0) + np.std(std_at_alpha6, 0), np.mean(std_at_alpha6, 0) - np.std(std_at_alpha6, 0), color='C1', alpha=0.3)
plt.xlabel(r'Training $\sigma$', fontsize=18)
plt.ylabel(r'Inference $\sigma$ at $\alpha=1\%$', fontsize=18)
plt.ylim([0, 20])
plt.xlim([0, 20])
plt.legend(loc=3)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('sigma_inf_sigma_median{}.pdf'.format(model))


plt.figure(dpi=200)
plt.plot([0, 20], [0.5, 0.5], '--', label='Random detector', color='k')

sa = np.stack(std_at_alpha7).reshape(-1)
a = np.stack(auroc7).reshape(-1)
D = {}
for i in range(len(sa)):
    if sa[i] not in D:
        D[sa[i]] = [a[i]]
    else:
        D[sa[i]].append(a[i])

mean, std = [], []
s = np.sort(list(D.keys()))
for i in s:
    mean.append(np.mean(D[i]))
    std.append(np.std(D[i]))
mean, std = np.stack(mean), np.stack(std)
plt.plot(s, mean, label="DeepFakes", color='C0')
plt.fill_between(s, mean+std, mean-std, color='C0', alpha=0.3)


sa = np.stack(std_at_alpha6).reshape(-1)
a = np.stack(auroc6).reshape(-1)
D = {}
for i in range(len(sa)):
    if sa[i] not in D:
        D[sa[i]] = [a[i]]
    else:
        D[sa[i]].append(a[i])

mean, std = [], []
s = np.sort(list(D.keys()))
for i in s:
    mean.append(np.mean(D[i]))
    std.append(np.std(D[i]))
mean, std = np.stack(mean), np.stack(std)
plt.plot(s, mean, label="FaceSwap", color='C1')
plt.fill_between(s, mean+std, mean-std, color='C1', alpha=0.3)


plt.xlabel(r'$\sigma$ at $\alpha=1\%$', fontsize=18)
plt.ylabel(r'AUROC', fontsize=18)
plt.ylim([0, 1])
plt.xlim([0, 20])
plt.legend(loc=3)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
# plt.title("ResNet-18")
plt.tight_layout()
plt.savefig('tradeoff_median{}.pdf'.format(model))