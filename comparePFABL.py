import copy
import sys
import os
import math
import numpy           as np
import multiprocessing as mp
import pandas as pd
import scipy as sp
import re
from scipy.interpolate import Rbf
from scipy             import optimize
from scipy.optimize    import minimize
from scipy.optimize    import Bounds
from statsmodels.graphics.tsaplots import acf


from joblib         import Parallel, delayed
#from scipy.signal   import welch, hanning
from scipy.optimize import curve_fit
from scipy.stats    import pearsonr


import matplotlib
import matplotlib.pyplot as plt

font={'size'   : 16}
matplotlib.rc('font', **font)


def expCurve(t, tau):
    return np.exp(-t/tau)

def readPressure(counter, directory):
    filename = directory + '/probes.' + str(counter).zfill(8) + '.pcd'
    print(str(counter) + ' ' + filename)
    
    data = np.loadtxt(filename, skiprows=1)
            
    return data

def computeAutocorrelation(index, component):
                
    return acf(uPrime[:,index,component], nlags = 1000, fft=True)

colors = {'0.04':'tab:blue'
         ,'0.06':'tab:grey'
         ,'0.08':'tab:red'
         ,'0.10':'tab:cyan'
         ,'0.12':'tab:orange'
         ,'0.14':'tab:purple'
         ,'0.16':'tab:green'}

linestyles = {'52':(0,(1,1))
             ,'57':(0,(1,1))
             ,'62':(0,(5,10))
             ,'67':(0,(5,10))
             ,'72':(0,(5,5))
             ,'77':(0,(5,5))
             ,'82':(0,(3,1,1,1))
             ,'87':(0,(3,1,1,1))
             ,'92':(0,())}
yMax = 1.0

directories = {'Pss/' :[[0.12, 15, 92], 75000,'../../TIGTestMatrixLong']
              ,'PF1/' :[[0.04, 15, 92], 75000,'../../TIGTestMatrixLong']}
              
dt = 0.004
imposed = False
prefixes = ['0p3_','0p6_','0p9_','1p2_','1p5_','1p8_','2p1_','2p4_','2p7_','3p0_','3p3_','3p6_','4p0_','5p0_','6p0_','7p0_','9p0_','11p0_','13p0_']
bound = True

heights = [0.1, 0.2, 0.5]
yOverhMax = 10

cont = 0
my_dpi = 100
plt.figure(figsize=(2000/my_dpi, 550/my_dpi), dpi=my_dpi)


for fold in directories:
    for prefix in prefixes:
            
        lastStep = str(directories[fold][1]).zfill(8)

        h  = directories[fold][0][0]
        u  = directories[fold][0][1]
        nx = directories[fold][0][2]

        c = colors['{0:.2f}'.format(h)]
        l = linestyles['{0:.0f}'.format(nx)]

        avg_u = np.loadtxt(directories[fold][2]+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(nx)+'/'+prefix+'avg_u.'+lastStep+'.collapse_width.dat',skiprows = 3)

        Umag = np.loadtxt(directories[fold][2]+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(nx)+'/'+prefix+'mag_u.'+lastStep+'.collapse_width.dat',skiprows = 3)

        rms_u = np.loadtxt(directories[fold][2]+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(nx)+'/'+prefix+'rms_u.'+lastStep+'.collapse_width.dat',skiprows = 3)
        rms_v = np.loadtxt(directories[fold][2]+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(nx)+'/'+prefix+'rms_v.'+lastStep+'.collapse_width.dat',skiprows = 3)
        rms_w = np.loadtxt(directories[fold][2]+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(nx)+'/'+prefix+'rms_w.'+lastStep+'.collapse_width.dat',skiprows = 3)

        uv = np.loadtxt(directories[fold][2]+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(nx)+'/'+prefix+'uv.'+lastStep+'.collapse_width.dat',skiprows = 3)

        plt.subplot(141)
        plt.plot(avg_u[:,5]/u, avg_u[:,3], color = c, linestyle=l)
        plt.xlabel(r'$U/U_\infty$')
        plt.ylabel('y [m]')
        plt.ylim([0,yMax])
        plt.yticks([0,yMax*0.25,yMax*0.5,yMax*0.75,yMax])
        plt.xlim([0,1.2])
        plt.xticks([0.0,0.6,1.2])

        plt.subplot(142)
        plt.plot(rms_u[:,5]/avg_u[:,5], rms_u[:,3], color = c, linestyle=l)
        plt.xlabel(r'$I_u$')
        plt.ylim([0,yMax])
        plt.yticks([0,yMax*0.25,yMax*0.5,yMax*0.75,yMax])
        plt.xlim([0,0.8])
        plt.xticks([0.0,0.2,0.4,0.6,0.8])
        plt.gca().set_yticklabels([])

        plt.subplot(143)
        plt.plot(rms_v[:,5]/avg_u[:,5], rms_w[:,3], color = c, linestyle=l)
        plt.xlabel(r'$I_v$')
        plt.ylim([0,yMax])
        plt.yticks([0,yMax*0.25,yMax*0.5,yMax*0.75,yMax])
        plt.xlim([0,0.8])
        plt.xticks([0.0,0.2,0.4,0.6,0.8])
        plt.gca().set_yticklabels([])

        plt.subplot(144)
        plt.plot(rms_w[:,5]/avg_u[:,5], rms_w[:,3], color = c, linestyle=l)
        plt.xlabel(r'$I_w$')
        plt.ylim([0,yMax])
        plt.yticks([0,yMax*0.25,yMax*0.5,yMax*0.75,yMax])
        plt.xlim([0,0.8])
        plt.xticks([0.0,0.2,0.4,0.6,0.8])
        plt.gca().set_yticklabels([])
        
#plt.legend(handles, labels, ncol=2, frameon=False)
plt.savefig('ABLfx.png', format="png", bbox_inches='tight')
#plt.show()
plt.close('all')

directories = {'PF1/' :[[0.04, 15, 52], 75000,'../../TIGTestMatrixLong']
              ,'PF2/' :[[0.04, 15, 62], 75000,'../../TIGTestMatrixLong']
              ,'PF3/' :[[0.04, 15, 72], 75000,'../../TIGTestMatrixLong']
              ,'PF4/' :[[0.04, 15, 82], 75000,'../../TIGTestMatrixLong']
              ,'PF5/' :[[0.04, 15, 92], 75000,'../../TIGTestMatrixLong']
              ,'PFj/' :[[0.06, 15, 57], 75000,'../../TIGTestMatrixLong']
              ,'PFk/' :[[0.06, 15, 67], 75000,'../../TIGTestMatrixLong']
              ,'PFl/' :[[0.06, 15, 77], 75000,'../../TIGTestMatrixLong']
              ,'PFm/' :[[0.06, 15, 87], 75000,'../../TIGTestMatrixLong']
              ,'PFn/' :[[0.08, 15, 52], 75000,'../../TIGTestMatrixLong']
              ,'PFo/' :[[0.08, 15, 62], 75000,'../../TIGTestMatrixLong']
              ,'PFp/' :[[0.08, 15, 72], 75000,'../../TIGTestMatrixLong']
              ,'PFq/' :[[0.08, 15, 82], 75000,'../../TIGTestMatrixLong']
              ,'PFr/' :[[0.08, 15, 92], 75000,'../../TIGTestMatrixLong']
              ,'PFg/' :[[0.12, 15, 52], 75000,'../../TIGTestMatrixLong']
              ,'PFR/' :[[0.12, 15, 62], 75000,'../../TIGTestMatrixLong']
              ,'PFV/' :[[0.12, 15, 72], 75000,'../../TIGTestMatrixLong']
              ,'PFh/' :[[0.12, 15, 82], 75000,'../../TIGTestMatrixLong']
              ,'PF./' :[[0.12, 15, 92], 75000,'../../TIGTestMatrixLong']
              ,'PF6/' :[[0.14, 15, 57], 75000,'../../TIGTestMatrixLong']
              ,'PF7/' :[[0.14, 15, 67], 75000,'../../TIGTestMatrixLong']
              ,'PF8/' :[[0.14, 15, 77], 75000,'../../TIGTestMatrixLong']
              ,'PFD/' :[[0.16, 15, 52], 75000,'../../TIGTestMatrixLong']
              ,'PF9/' :[[0.16, 15, 62], 75000,'../../TIGTestMatrixLong']
              ,'PF0/' :[[0.16, 15, 72], 75000,'../../TIGTestMatrixLong']
              ,'Psd/' :[[0.16, 15, 82], 75000,'../../TIGTestMatrixLong']
              ,'Pss/' :[[0.16, 15, 92], 75000,'../../TIGTestMatrixLong']}
prefixes = ['0p3_']
prefix = ''
bound = 0.1
bound = True

heights = [0.1, 0.2, 0.5]
yOverhMax = 10

cont = 0
my_dpi = 100
plt.figure(figsize=(2000/my_dpi, 550/my_dpi), dpi=my_dpi)


for fold in directories:
    for prefix in prefixes:
            
        lastStep = str(directories[fold][1]).zfill(8)

        h  = directories[fold][0][0]
        u  = directories[fold][0][1]
        nx = directories[fold][0][2]

        c = colors['{0:.2f}'.format(h)]
        l = linestyles['{0:.0f}'.format(nx)]

        avg_u = np.loadtxt(directories[fold][2]+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(nx)+'/'+prefix+'avg_u.'+lastStep+'.collapse_width.dat',skiprows = 3)

        Umag = np.loadtxt(directories[fold][2]+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(nx)+'/'+prefix+'mag_u.'+lastStep+'.collapse_width.dat',skiprows = 3)

        rms_u = np.loadtxt(directories[fold][2]+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(nx)+'/'+prefix+'rms_u.'+lastStep+'.collapse_width.dat',skiprows = 3)
        rms_v = np.loadtxt(directories[fold][2]+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(nx)+'/'+prefix+'rms_v.'+lastStep+'.collapse_width.dat',skiprows = 3)
        rms_w = np.loadtxt(directories[fold][2]+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(nx)+'/'+prefix+'rms_w.'+lastStep+'.collapse_width.dat',skiprows = 3)

        uv = np.loadtxt(directories[fold][2]+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(nx)+'/'+prefix+'uv.'+lastStep+'.collapse_width.dat',skiprows = 3)

        plt.subplot(141)
        plt.plot(avg_u[:,5]/u, avg_u[:,3], color = c, linestyle=l)
        plt.xlabel(r'$U/U_\infty$')
        plt.ylabel('y [m]')
        plt.ylim([0,yMax])
        plt.yticks([0,yMax*0.25,yMax*0.5,yMax*0.75,yMax])
        plt.xlim([0,1.2])
        plt.xticks([0.0,0.6,1.2])

        plt.subplot(142)
        plt.plot(rms_u[:,5]/avg_u[:,5], rms_u[:,3], color = c, linestyle=l)
        plt.xlabel(r'$I_u$')
        plt.ylim([0,yMax])
        plt.yticks([0,yMax*0.25,yMax*0.5,yMax*0.75,yMax])
        plt.xlim([0,0.8])
        plt.xticks([0.0,0.2,0.4,0.6,0.8])
        plt.gca().set_yticklabels([])

        plt.subplot(143)
        plt.plot(rms_v[:,5]/avg_u[:,5], rms_w[:,3], color = c, linestyle=l)
        plt.xlabel(r'$I_v$')
        plt.ylim([0,yMax])
        plt.yticks([0,yMax*0.25,yMax*0.5,yMax*0.75,yMax])
        plt.xlim([0,0.8])
        plt.xticks([0.0,0.2,0.4,0.6,0.8])
        plt.gca().set_yticklabels([])

        plt.subplot(144)
        plt.plot(rms_w[:,5]/avg_u[:,5], rms_w[:,3], color = c, linestyle=l)
        plt.xlabel(r'$I_w$')
        plt.ylim([0,yMax])
        plt.yticks([0,yMax*0.25,yMax*0.5,yMax*0.75,yMax])
        plt.xlim([0,0.8])
        plt.xticks([0.0,0.2,0.4,0.6,0.8])
        plt.gca().set_yticklabels([])
        
#plt.legend(handles, labels, ncol=2, frameon=False)
plt.savefig('ABLall.png', format="png", bbox_inches='tight')
#plt.show()
plt.close('all')

f = lambda l,c: plt.plot([],[], color=c, linestyle=l)[0]
#f = lambda l,c: plt.plot([],[], color=c, linestyle=l)[0]

handles = [f("-", colors[i]) for i in colors.keys()]
handles += [f(linestyles[i], "tab:grey") for i in linestyles.keys()]

labels = []
for h in colors.keys():
    labels.append('h='+h+'m')
for nx in linestyles.keys():
    labels.append('r='+nx)
    
plt.figure()
plt.gca().set_yticklabels([])
plt.gca().set_xticklabels([])
plt.legend(handles, labels, ncol=2, frameon=False)
plt.savefig('legend.png', bbox_inches='tight')
plt.show()
plt.close('all')

              
#dt = 0.004
#imposed = False
#prefixes = ['0p3_']
#bound = True
#lw = 0.3
#heights = [0.1, 0.2, 0.5]
#yMax = 1.2
#yOverhMax = 10

#cont = 0
#my_dpi = 100
#plt.figure(figsize=(375/my_dpi, 175/my_dpi), dpi=my_dpi)

#for prefix in prefixes:

    #for fold in directories:
            
        #lastStep = str(directories[fold][1]).zfill(8)

        #h  = directories[fold][0][0]
        #u  = directories[fold][0][1]
        #nx = directories[fold][0][2]

        #c = colors['{0:.2f}'.format(h)]
        #l = linestyles['{0:.0f}'.format(nx)]

        #avg_u = np.loadtxt(directories[fold][2]+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(nx)+'/'+prefix+'avg_u.'+lastStep+'.collapse_width.dat',skiprows = 3)

        #Umag = np.loadtxt(directories[fold][2]+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(nx)+'/'+prefix+'mag_u.'+lastStep+'.collapse_width.dat',skiprows = 3)

        #rms_u = np.loadtxt(directories[fold][2]+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(nx)+'/'+prefix+'rms_u.'+lastStep+'.collapse_width.dat',skiprows = 3)
        #rms_v = np.loadtxt(directories[fold][2]+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(nx)+'/'+prefix+'rms_v.'+lastStep+'.collapse_width.dat',skiprows = 3)
        #rms_w = np.loadtxt(directories[fold][2]+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(nx)+'/'+prefix+'rms_w.'+lastStep+'.collapse_width.dat',skiprows = 3)

        #uv = np.loadtxt(directories[fold][2]+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(nx)+'/'+prefix+'uv.'+lastStep+'.collapse_width.dat',skiprows = 3)

        ##rms_p = np.loadtxt(directories[fold][2]+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(nx)+'/'+prefix+'rms_p.'+lastStep+'.collapse_width.dat',skiprows = 3)

        #plt.subplot(141)
        #plt.plot(avg_u[:,5]/u, avg_u[:,3], color = c, linestyle=l,linewidth=lw)
        #plt.ylim([0,yMax])
        #plt.xlim([0,1.2])
        #plt.gca().set_yticklabels([])
        #plt.gca().set_xticklabels([])
        #plt.axis('off')

        #plt.subplot(142)
        #plt.plot(rms_u[:,5]/avg_u[:,5], rms_u[:,3], color = c, linestyle=l,linewidth=lw)
        #plt.ylim([0,yMax])
        #plt.xlim([0,1.0])
        #plt.gca().set_yticklabels([])
        #plt.gca().set_xticklabels([])
        #plt.axis('off')

        #plt.subplot(143)
        #plt.plot(rms_v[:,5]/avg_u[:,5], rms_v[:,3], color = c, linestyle=l,linewidth=lw)
        #plt.ylim([0,yMax])
        #plt.xlim([0,1.0])
        #plt.gca().set_yticklabels([])
        #plt.gca().set_xticklabels([])
        #plt.axis('off')

        #plt.subplot(144)
        #plt.plot(rms_w[:,5]/avg_u[:,5], rms_w[:,3], color = c, linestyle=l,linewidth=lw)
        #plt.ylim([0,yMax])
        #plt.xlim([0,1.0])
        #plt.gca().set_yticklabels([])
        #plt.gca().set_xticklabels([])
        #plt.axis('off')
        
#plt.savefig('ABLResults.png', bbox_inches='tight')
#plt.savefig('ABLIllustratorGPR.svg', format="svg", bbox_inches='tight')
#plt.show()
#plt.close('all')
