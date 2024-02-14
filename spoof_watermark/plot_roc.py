import os 
from six.moves import cPickle as pkl 
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt 
import numpy as np
import seaborn as sns 
sns.set_theme()
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 25

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.figure(dpi=200, figsize=(8,6))

for i, method in enumerate(["dwtDct", "dwtDctSvd", "rivaGan", "watermarkDM", "treeRing"]):

    with open("results/{}_spoofed.pkl".format(method), "rb") as f:
        spoofed = np.stack(pkl.load(f))
        
    with open("results/{}_clean.pkl".format(method), "rb") as f:
        clean = np.stack(pkl.load(f))
        
    with open("results/{}_wm.pkl".format(method), "rb") as f:
        wm = np.stack(pkl.load(f))
    

    true_labels = [1]*len(wm) + [0]*len(clean)
    predicted_probs = wm.tolist() + clean.tolist()
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='{} AUROC = {:.4f}'.format(method, roc_auc), color="C{}".format(i))

    true_labels = [1]*len(wm) + [0]*len(spoofed)
    predicted_probs = wm.tolist() + spoofed.tolist()
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, '--', label='{} (spoofed) AUROC = {:.4f}'.format(method, roc_auc), color="C{}".format(i))

method = ""

plt.plot([0, 1], [0, 1], '--', color='k', label='Random detection')

plt.legend(loc='lower right', framealpha=0.5)
plt.xlabel('FPR')   
plt.ylabel('TPR')
plt.tight_layout()
plt.savefig('results/plots/{}_spoof_roc.pdf'.format(method))