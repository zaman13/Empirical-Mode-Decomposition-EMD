#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 12:42:19 2022

@author: Mohammad Asif Zaman

Empirical Mode Decomposition
"""
from __future__ import print_function    

import time
import numpy as np
import pylab as py
from scipy.signal import argrelextrema
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.interpolate import splev, splrep



clr_set = ['#eaeee0','#ce9c9d','#adb5be','#57838d','#80adbc','#b4c9c7','#dadadc','#f3bfb3','#ccadb2','#445a67']


# Load earthquake data set
# my_data = np.genfromtxt('kobe_earthquake.csv', skip_header = 1, delimiter=',')
# t = my_data[:,[0]]
# fs = 1/(t[1]-t[0])
# x = my_data[:,[1]]


# Use test data set
t = np.linspace(0,.22,5001)
sine_wave = lambda f,t: np.sin(2*np.pi*f*t)

y11 = 4+2*sine_wave(50, t)
y12 = -0.4*sine_wave(12, t)
y1 = y11+y12
y2 = sine_wave(250, t)
y = y1*y2+np.exp(15*t) + 10*y12
x = y


# py.plot(t,x)



# =============================================================================
def local_max_min_number(x):
    local_maxima_ind = argrelextrema(x, np.greater)
    local_minima_ind = argrelextrema(x, np.less)
    
    Nmax = np.size(local_maxima_ind)
    Nmin = np.size(local_minima_ind)
    
    return Nmax, Nmin
# =============================================================================




# =============================================================================
def isResidue(x):
    Nmax, Nmin = local_max_min_number(x)
    mx = max(Nmax, Nmin)
    
    flag = True if mx <= 3 else False
    
    return flag
# =============================================================================



# =============================================================================
# Function to check whether a function is a valid IMF or not
# =============================================================================
def isIMF(x,stop_threshold):
    # stop_threshold:  mean value needs to be below this level for us to consider the input as a valid IMF
    
    Nmax, Nmin = local_max_min_number(x)
    Ndiff = np.abs(Nmax-Nmin)
    
    mean_value = np.mean(x)
    
    flag1 = True if Ndiff < 2 else False
    flag2 = True if np.abs(mean_value) < stop_threshold else False
    
    # print('Mean value = %f \n' % (np.abs(mean_value)))
    # print('Threshold = %f \n' % (th))
    # print('Flag 1 = %d \n' % (flag1))
    # print('Flag 2 = %d \n' % (flag2))
   
    return flag1 and flag2
# =============================================================================





# =============================================================================
def low_freq_trend(t,x):
    
    Nx = len(x)
   
    local_maxima_ind = (argrelextrema(x, np.greater))    # Find the indices of local maximas
    local_minima_ind = (argrelextrema(x, np.less))       # Find the indices of local minimas
    
    # Only take the relevant axis data as argrelextrema returns tuple with muultiple axis data. This can happen even if the input array is 1D
    local_maxima_ind = local_maxima_ind[0]
    local_minima_ind = local_minima_ind[0]
    
    
    # Now, we modify the local maxima and minima by appending the first and last elements if needed
    # The goal is to avoid extrapolation when calculating/interpolating the envelope
    # We set the first and last element of these arrays equal to the first and last element of the original series
    # That way, both the envelope start and end at the function value. However, if the first and last element
    # of the function are local maxima/minima themselves, there could be some issues.
    
    local_maxima_ind = np.insert(local_maxima_ind,0,0) if local_maxima_ind[0] != 0 else local_maxima_ind
    local_maxima_ind = np.append(local_maxima_ind,Nx-1) if local_maxima_ind[-1] != Nx - 1 else local_maxima_ind

    local_minima_ind = np.insert(local_minima_ind,0,0) if local_minima_ind[0] != 0 else local_minima_ind
    local_minima_ind = np.append(local_minima_ind,Nx-1) if local_minima_ind[-1] != Nx - 1 else local_minima_ind

    


    # interp_fun_maxima = CubicSpline(t[local_maxima_ind],x[local_maxima_ind],bc_type='natural', extrapolate=bool)    # Create a cubic spline function for the local maxima envelope
    # interp_fun_minima = CubicSpline(t[local_minima_ind],x[local_minima_ind],bc_type='natural', extrapolate=bool)    # Create a cubic spline function for the local minima envelope
    
    # interp_fun_maxima = CubicSpline(t[local_maxima_ind],x[local_maxima_ind])    # Create a cubic spline function for the local maxima envelope
    # interp_fun_minima = CubicSpline(t[local_minima_ind],x[local_minima_ind])    # Create a cubic spline function for the local minima envelope
    
    # interp_fun_maxima = interp1d(t[local_maxima_ind], x[local_maxima_ind], kind = 'cubic', fill_value='extrapolate')
    # interp_fun_minima = interp1d(t[local_minima_ind], x[local_minima_ind], kind = 'cubic', fill_value='extrapolate')
    
   
    # spline_maxima = interp_fun_maxima(t)    # Calculate the upper envelope
    # spline_minima = interp_fun_minima(t)    # Calculate the lower envelope
    
    
    interp_fun_maxima = splrep(t[local_maxima_ind], x[local_maxima_ind])
    interp_fun_minima = splrep(t[local_minima_ind], x[local_minima_ind])
        
      
    spline_maxima = splev(t,interp_fun_maxima)
    spline_minima = splev(t,interp_fun_minima)
    
    spline_avg = 0.5*(spline_maxima + spline_minima)   # Calculate the average of the upper and lower envelope. This is the trend line.
    print('   Envelope mean = %f' %(np.mean(spline_avg)))
    # py.plot(t[local_maxima_ind],x[local_maxima_ind],'r.')
    # py.plot(t[local_minima_ind],x[local_minima_ind],'g.')
    # py.plot(t, spline_maxima,'r')
    # py.plot(t, spline_minima,'g')

    return spline_avg
# =============================================================================




# =============================================================================
def IMF_calc(t,x, stop_threshold, max_iter):
    
    SD_threshold = stop_threshold
    h = x
    counter = 0
    
    flag = False
    
    while flag == False:
        counter = counter + 1
        m = low_freq_trend(t,h)
        h_new = h - m
        SD = np.linalg.norm(h-h_new)/np.linalg.norm(h)
        h = h_new
        # flag = True if counter >= max_iter else isIMF(h, stop_threshold)
        flag1 = SD < SD_threshold
        flag2 = counter > max_iter
        flag = flag1 or flag2
        # print(SD)
      
    
    return h, counter
# =============================================================================




# =============================================================================
# Calculate IMF set
# =============================================================================
def IMF_set(t,x,stop_threshold, max_iter,max_imfs):        
    r =  x    # set residue = original function at start
    imf_set_list = []   # Empty list to store all the IMFs
     
    counter = 0     # Counting the number of calculated IMFs
    
    while not(isResidue(r)):
        counter = counter + 1
        print('Calculating IMF %d' %(counter))
        imf, iter_required = IMF_calc(t,r,stop_threshold, max_iter)
        print('Refinement iterations required = %d ' %(iter_required))
        print('IMF mean = %f ' %(np.mean(imf)))
        
        imf_set_list.append(imf)
        r = r - imf
        print('Residue RMS = %f \n' %(np.sqrt(np.mean(r**2))))
        
        
        # if number of calculated imfs is >= defined maximum limit of imfs, then break        
        if counter >= max_imfs:
            break
        
    return imf_set_list, r
# =============================================================================





# =============================================================================
# Main program
# =============================================================================

max_iter = 100
max_imfs = 8
stop_threshold = 1e-3

imf_set_list,residue = IMF_set(t,x,stop_threshold, max_iter,max_imfs)
N_imf_set = len(imf_set_list)

ind_trunc = np.arange(100,len(t)-100)

py.figure()
for m in range(N_imf_set):
    py.subplot(100*N_imf_set + 111 + m)
    # py.figure()
    py.plot(t[ind_trunc], imf_set_list[m][ind_trunc],clr_set[m],label= 'IMF ' + str(m+1))
    py.legend()


py.subplot(100*N_imf_set + 111 + N_imf_set)    
# py.figure()
py.plot(t, residue, clr_set[m+1], label = 'Residue')
py.legend()

# =============================================================================

