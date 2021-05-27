# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:48:01 2021

@author: Adrian
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from analysis import evaluate_window, find_sections, get_radon, radon_fit_ellipses
from training import gen_training_data, slope

### synthetic data example

# parameters for synthetic data
N = 4096
m0 = slope()
dm = 0.75
s0 = np.pi/2
ds = 0.75
R = 0.1
O = 0.33

# generate synthetic data
training_data_params = {'N':N, 'm0':m0, 'dm':dm, 's0':s0, 'ds':ds, 'R':R, 'O':O}
X, Y, oL = gen_training_data(**training_data_params)

# DL segmentation
model = load_model('./DL_denoise_model_4096.h5')
L = evaluate_window(model, Y, 4096)

# partition into sections
sections = find_sections(X, Y, L)

# check slope
m = np.abs([sections[ind]['m'] for ind in sections])
print('Slope in: mu={:.3f}, std={:.3f} -- slope recovered: mu={:.3f}, std={:.3f}'.format(m0, m0*dm/2, np.mean(m), np.std(m)))

plt.figure(figsize=plt.figaspect(.75/(2*1.618)))
plt.scatter(X[oL==0], Y[oL==0], s=1, c='gray', alpha = 0.5)
plt.scatter(X[oL==1], Y[oL==1], s=1, c='r')
plt.scatter(X[oL==-1], Y[oL==-1], s=1, c='b')
plt.xlabel('time (s)', fontsize=16)
plt.ylabel('angle (rad)', fontsize=16)
plt.title('synthetic data (labeled)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

plt.figure(figsize=plt.figaspect(.75/(2*1.618)))
plt.scatter(X[oL==0], Y[oL==0], s=1, c='gray')
plt.scatter(X[oL==1], Y[oL==1], s=1, c='gray')
plt.scatter(X[oL==-1], Y[oL==-1], s=1, c='gray')
plt.xlabel('time (s)', fontsize=16)
plt.ylabel('angle (rad)', fontsize=16)
plt.title('synthetic data (unlabeled)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

plt.figure(figsize=plt.figaspect(.75/(2*1.618)))
plt.scatter(X[L==0], Y[L==0], s=1, c='gray', alpha = 0.5)
plt.scatter(X[L==1], Y[L==1], s=1, c='r')
plt.scatter(X[L==2], Y[L==2], s=1, c='b')
plt.xlabel('time (s)', fontsize=16)
plt.ylabel('angle (rad)', fontsize=16)
plt.title('synthetic data (labels recovered)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

plt.figure(figsize=plt.figaspect(.75/(2*1.618)))
plt.scatter(X[L==0], Y[L==0], s=1, c='gray', alpha = 0.5)
plt.scatter(X[L==1], Y[L==1], s=1, c='r')
plt.scatter(X[L==2], Y[L==2], s=1, c='b')

for i, section in sections.items():
    plt.plot(section['xf'], section['yf'], 'k--')

plt.xlabel('time (s)', fontsize=16)
plt.ylabel('angle (rad)', fontsize=16)
plt.title('synthetic data (with fits)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()


### real data example

Path = './radon_samples'

radon = get_radon(Path)
_, _, Y, _ = radon_fit_ellipses(radon)

X = np.arange(len(Y))

L = evaluate_window(model, Y, 4096)
sections = find_sections(X, Y, L)

plt.figure(figsize=plt.figaspect(.75/(2*1.618)))
plt.scatter(X[L==0], Y[L==0], s=1, c='gray', alpha = 0.5)
plt.scatter(X[L==1], Y[L==1], s=1, c='r')
plt.scatter(X[L==2], Y[L==2], s=1, c='b')
plt.xlabel('time (s)', fontsize=16)
plt.ylabel('angle (rad)', fontsize=16)
plt.title('synthetic data (labels recovered)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

plt.figure(figsize=plt.figaspect(.75/(2*1.618)))
plt.scatter(X[L==0], Y[L==0], s=1, c='gray', alpha = 0.5)
plt.scatter(X[L==1], Y[L==1], s=1, c='r')
plt.scatter(X[L==2], Y[L==2], s=1, c='b')

for i, section in sections.items():
    plt.plot(section['xf'], section['yf'], 'k--')

plt.xlabel('time (s)', fontsize=16)
plt.ylabel('angle (rad)', fontsize=16)
plt.title('synthetic data (with fits)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

m = np.abs([sections[ind]['m'] for ind in sections]) * 3600 / 12 * 180 / np.pi
slen = np.asarray([sections[ind]['nI'] for ind in sections])

np.sum(m*slen)/np.sum(slen)

# Path = 'E:/Seadrive/Adrian F/Meine Bibliotheken/Phasenwellen-Projekt/codes_unsorted/rotation_analysis/SlopeData'
# DataSets = [Set for Set in os.listdir(Path) if '.pkl' in Set]
# Set = np.random.choice(DataSets)

# with open(os.path.join(Path,Set),'rb') as f:
#     Data = pickle.load(f)
#     f.close()
    
# Y = Data['p']
# X = Data['t']
# L = evaluate_window(model, Y, 2048)
# sections = find_sections(X, Y, L)

# plt.figure(figsize=plt.figaspect(.75/(2*1.618)))
# plt.scatter(X[L==0], Y[L==0], s=1, c='gray', alpha = 0.5)
# plt.scatter(X[L==1], Y[L==1], s=1, c='r')
# plt.scatter(X[L==2], Y[L==2], s=1, c='b')
# for i, section in sections.items():
#     plt.plot(section['xf'], section['yf'], 'k--')
# plt.xlabel('time (s)', fontsize=16)
# plt.ylabel('angle (rad)', fontsize=16)
# plt.title('experimental data (labeled, with fits)', fontsize=16)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.tight_layout()