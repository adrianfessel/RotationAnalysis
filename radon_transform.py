# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:20:59 2019

@author: Adrian
"""

import os
import numpy as np
import pickle

from skimage.transform import radon
from tqdm import tqdm


def pload(file, binning=None, interp='linear'):
    """
    Function for consistent reading of phase data. Can perform binning.

    Parameters
    ----------
    file : string
        system path
    binning : int, optional
        binning factor; the default is None.
    interp : 'string', optional
        interpolation mode; the default is 'linear', options are 'linear', 
        'nearest', 'cubic'

    Returns
    -------
    I : numpy array
        phase map

    """

    I = cv2.imread(file,cv2.IMREAD_GRAYSCALE)

    Imax = np.iinfo(I.dtype).max
    I = np.float32(I)
    I /= Imax
    
    B = I > 0
    I[B==0] = np.nan
    
    if binning:
        
        size = (np.int(I.shape[1]/binning),np.int(I.shape[0]/binning))
        
        if interp == 'linear':
            I = cv2.resize(I,size,interpolation=cv2.INTER_LINEAR)
        elif interp == 'nearest':
            I = cv2.resize(I,size,interpolation=cv2.INTER_NEAREST)
        elif interp == 'cubic':
            I = cv2.resize(I,size,interpolation=cv2.INTER_CUBIC)
            
    return I


class radon_transform():

    
    def __init__(self, Path, Parameters):
        """
        Class for obtaining the radon transform of a sequence of phase maps.
        Can be used to the direction and other features of phase waves.
        
        In the process, phase maps will be shifted to the center of rotation
        (center of the image).

        Output directory will be automatically created in the parent directory.

        Parameters
        ----------
        Path : string
            system path to input data directory
        Parameters : dict
            transformation parameters
            'binning' : int
                reduce spatial input dimensions by factor
            'n_angles' : int
                number of angles for which the radon transform will be computed

        Returns
        -------
        None.

        """
        
        self.Path = Path
        self.Par = Parameters
        
        self.Par['binning'] = False if 'binning' not in self.Par else self.Par['binning']
        self.Par['n_angles'] = 256 if 'n_angles' not in self.Par else self.Par['n_angles']

        self.T = np.linspace(0,180,self.Par['n_angles'],endpoint=False)

        self.Frames = [Frame for Frame in os.listdir(Path) if '.tif' in Frame or '.png' in Frame or '.jpeg' in Frame or '.jpg' in Frame]
        
        self.Shape = {}
        
        self.Path_Radon = Path + '_Radon'

        if not os.path.isdir(self.Path_Radon):
            os.mkdir(self.Path_Radon)

        
    def run(self):
        """
        Run conversion for the directory specified on instantiation.

        Returns
        -------
        None.

        """
        
        for i, Frame in tqdm(enumerate(self.Frames), total=len(self.Frames)):

            I = shared.pload(os.path.join(self.Path, Frame), binning=self.Par['binning'], interp='linear')
            
            I = I-np.nanmean(I)
            I = I/np.nanstd(I)
            
            I[I!=I] = 0
            
            I = shared.to_center_minimize(I)
            I = shared.pad_to_square(I)
            
            R = radon(I, self.T, circle=False)
            V = np.var(R, axis=0)
            
            self.Shape[Frame] = V
            
            pickle.dump(V, open(os.path.join(self.Path_Radon,Frame.split('.')[0]+'.pkl'), "wb" ) )
        
