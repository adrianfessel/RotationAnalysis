# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:48:01 2021

@author: Adrian
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.keras.models import load_model

from analysis import evaluate_window, find_sections
from training import gen_training_data

# parameters for synthetic data
N = 4096
m0 = np.pi/3600*12
dm = 0.75
s0 = np.pi/2
ds = 0.75
R = 0.15
O = 0.5

# generate synthetic data
training_data_params = {'N':N, 'm0':m0, 'dm':dm, 's0':s0, 'ds':ds, 'R':R, 'O':O}
X, Y, oL = gen_training_data(**training_data_params)

# DL segmentation
model = load_model('C:/Users/Adrian/Desktop/DL_denoise_model_2048.h5')
L = evaluate_window(model, Y, int(N/2))

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


