import copy
import sys
import os
import math
import numpy           as np
import multiprocessing as mp
import pandas as pd
import scipy as sp
import scipy.io
import re
from scipy.interpolate import Rbf
from scipy             import optimize
from scipy.optimize    import minimize
from scipy.optimize    import Bounds
from statsmodels.graphics.tsaplots import acf
from scipy.interpolate import interp1d


from joblib         import Parallel, delayed
#from scipy.signal   import welch, hanning
from scipy.optimize import curve_fit
from scipy.stats    import pearsonr
from modelDefinition import *


import matplotlib
import matplotlib.pyplot as plt

font={'size'   : 15}
matplotlib.rc('font', **font)

def RMSE_compute(group):
    
    all_les = np.concatenate(group['LES'].values)
    all_gpr = np.concatenate(group['GPR'].values)
    
    return pd.Series({'RMSE': np.sqrt(np.sum((all_les-all_gpr)**2) / len(all_les))})

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
         ,'0.06':'tab:green'
         ,'0.08':'tab:purple'
         ,'0.12':'tab:orange'
         ,'0.14':'tab:red'
         ,'0.16':'tab:cyan'}

linestyles = {'52':(0,(1,1))
             ,'57':(0,(1,1))
             ,'62':(0,(5,10))
             ,'67':(0,(5,10))
             ,'72':(0,(5,1))
             ,'77':(0,(5,1))
             ,'82':(0,(3,1,1,1))
             ,'87':(0,(3,1,1,1))
             ,'92':(0,())}
        
cat_colors = {'Cat_D':'tab:blue','Cat_C':'tab:green','Cat_B':'tab:orange'}
cat_colors = {'LRB':'tab:blue','MRB':'tab:green','HRB':'tab:orange'}

#figs = ['ABLenvelope','ABLfx','ABLfh','downstreamRMSEGPR','downstreamTestGPR','upstreamTestGPR','UFenvelope','codesOptimized']
figs = ['codesOptimized']

for fig in figs:
    
    if fig == 'ABLenvelope':
        
        yMax = 1.5

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

        prefixes = ['0p6_','0p9_','1p2_','1p5_','1p8_','2p1_','2p4_','2p7_','3p0_','3p3_','3p6_','4p0_','5p0_','6p0_','7p0_','9p0_','11p0_','13p0_']
        prefix = ''
        bound = 0.1
        bound = True

        heights = [0.1, 0.2, 0.5]
        yOverhMax = 10

        cont = 0

        y = np.linspace(0.01,3.0,300)

        umin = np.ones((len(y),))*100
        uMax = np.ones((len(y),))*0

        Iumin = np.ones((len(y),))*100
        IuMax = np.ones((len(y),))*0

        Ivmin = np.ones((len(y),))*100
        IvMax = np.ones((len(y),))*0

        Iwmin = np.ones((len(y),))*100
        IwMax = np.ones((len(y),))*0


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
                
                uInterp  = (interp1d(avg_u[:,3], avg_u[:,5])(y))
                IuInterp = (interp1d(rms_u[:,3], rms_u[:,5]/avg_u[:,5])(y))
                IvInterp = (interp1d(rms_v[:,3], rms_v[:,5]/avg_u[:,5])(y))
                IwInterp = (interp1d(rms_w[:,3], rms_w[:,5]/avg_u[:,5])(y))
                
                umin = np.min([umin,uInterp/15], axis=0)
                uMax = np.max([uMax,uInterp/15], axis=0)
                
                Iumin = np.min([Iumin,IuInterp], axis=0)
                IuMax = np.max([IuMax,IuInterp], axis=0)
                
                Ivmin = np.min([Ivmin,IvInterp], axis=0)
                IvMax = np.max([IvMax,IvInterp], axis=0)
                
                Iwmin = np.min([Iwmin,IwInterp], axis=0)
                IwMax = np.max([IwMax,IwInterp], axis=0)

        plt.subplot(1,4,1)
        plt.fill_betweenx(y, umin, uMax, color='tab:grey',alpha = 0.4)
        plt.xlabel(r'$U/U_\infty$')
        plt.ylabel('y [m]')
        plt.ylim([0,yMax])
        plt.yticks([0,0.5,1.0,1.5])
        plt.xlim([0,1.2])
        plt.xticks([0.0,0.6,1.2])

        plt.subplot(1,4,2)
        plt.fill_betweenx(y, Iumin, IuMax, color='tab:grey',alpha = 0.4)
        plt.xlabel(r'$I_u$')
        plt.ylim([0,yMax])
        plt.yticks([0,0.5,1.0,1.5])
        plt.xlim([0,0.8])
        plt.xticks([0.0,0.2,0.4,0.6,0.8])
        plt.gca().set_yticklabels([])

        plt.subplot(1,4,3)
        plt.fill_betweenx(y, Ivmin, IvMax, color='tab:grey',alpha = 0.4)
        plt.xlabel(r'$I_v$')
        plt.ylim([0,yMax])
        plt.yticks([0,0.5,1.0,1.5])
        plt.xlim([0,0.8])
        plt.xticks([0.0,0.2,0.4,0.6,0.8])
        plt.gca().set_yticklabels([])

        plt.subplot(1,4,4)
        plt.fill_betweenx(y, Iwmin, IwMax, color='tab:grey',alpha = 0.4,label='Database envelope')
        plt.xlabel(r'$I_w$')
        plt.ylim([0,yMax])
        plt.yticks([0,0.5,1.0,1.5])
        plt.xlim([0,0.8])
        plt.xticks([0.0,0.2,0.4,0.6,0.8])
        plt.gca().set_yticklabels([])
                
        plt.legend(frameon=False)
        plt.savefig('../PaperPlots/ABLrange.png', format="png", bbox_inches='tight')
        plt.show()
        plt.close('all')
        
    elif fig == 'ABLfx':

        directories = {'Pss/' :[[0.12, 15, 92], 75000,'../../TIGTestMatrixLong']
                    ,'PF1/' :[[0.04, 15, 92], 75000,'../../TIGTestMatrixLong']}
                    
        dt = 0.004
        imposed = False
        prefixes = ['0p6_','0p9_','1p2_','1p5_','1p8_','2p1_','2p4_','2p7_','3p0_','3p3_','3p6_','4p0_','5p0_','6p0_','7p0_','9p0_','11p0_','13p0_']
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
                plt.yticks([0,0.5,1.0,1.5])
                plt.xlim([0,1.2])
                plt.xticks([0.0,0.6,1.2])

                plt.subplot(142)
                plt.plot(rms_u[:,5]/avg_u[:,5], rms_u[:,3], color = c, linestyle=l)
                plt.xlabel(r'$I_u$')
                plt.ylim([0,yMax])
                plt.yticks([0,0.5,1.0,1.5])
                plt.xlim([0,0.8])
                plt.xticks([0.0,0.2,0.4,0.6,0.8])
                plt.gca().set_yticklabels([])

                plt.subplot(143)
                plt.plot(rms_v[:,5]/avg_u[:,5], rms_w[:,3], color = c, linestyle=l)
                plt.xlabel(r'$I_v$')
                plt.ylim([0,yMax])
                plt.yticks([0,0.5,1.0,1.5])
                plt.xlim([0,0.8])
                plt.xticks([0.0,0.2,0.4,0.6,0.8])
                plt.gca().set_yticklabels([])

                plt.subplot(144)
                plt.plot(rms_w[:,5]/avg_u[:,5], rms_w[:,3], color = c, linestyle=l)
                plt.xlabel(r'$I_w$')
                plt.ylim([0,yMax])
                plt.yticks([0,0.5,1.0,1.5])
                plt.xlim([0,0.8])
                plt.xticks([0.0,0.2,0.4,0.6,0.8])
                plt.gca().set_yticklabels([])
                
        #plt.legend(handles, labels, ncol=2, frameon=False)
        plt.savefig('../PaperPlots/ABLfx.png', format="png", bbox_inches='tight')
        plt.show()
        plt.close('all')
        
    elif fig == 'ABLfh':

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
        prefixes = ['0p6_']
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
                plt.yticks([0,0.5,1.0,1.5])
                plt.xlim([0,1.2])
                plt.xticks([0.0,0.6,1.2])

                plt.subplot(142)
                plt.plot(rms_u[:,5]/avg_u[:,5], rms_u[:,3], color = c, linestyle=l)
                plt.xlabel(r'$I_u$')
                plt.ylim([0,yMax])
                plt.yticks([0,0.5,1.0,1.5])
                plt.xlim([0,0.8])
                plt.xticks([0.0,0.2,0.4,0.6,0.8])
                plt.gca().set_yticklabels([])

                plt.subplot(143)
                plt.plot(rms_v[:,5]/avg_u[:,5], rms_w[:,3], color = c, linestyle=l)
                plt.xlabel(r'$I_v$')
                plt.ylim([0,yMax])
                plt.yticks([0,0.5,1.0,1.5])
                plt.xlim([0,0.8])
                plt.xticks([0.0,0.2,0.4,0.6,0.8])
                plt.gca().set_yticklabels([])

                plt.subplot(144)
                plt.plot(rms_w[:,5]/avg_u[:,5], rms_w[:,3], color = c, linestyle=l)
                plt.xlabel(r'$I_w$')
                plt.ylim([0,yMax])
                plt.yticks([0,0.5,1.0,1.5])
                plt.xlim([0,0.8])
                plt.xticks([0.0,0.2,0.4,0.6,0.8])
                plt.gca().set_yticklabels([])

        f = lambda l,c: plt.plot([],[], color=c, linestyle=l)[0]

        plt.subplot(143)
        handles = [f("-", colors[i]) for i in colors.keys()]
        labels = []
        for h in colors.keys():
            labels.append('h='+h+'m')
        plt.legend(handles, labels, ncol=1, frameon=False)
            
        plt.subplot(144)
        handles = [f(linestyles[i], "tab:grey") for i in linestyles.keys()]
        labels = []
        for nx in linestyles.keys():
            labels.append('r='+nx)
        plt.legend(handles, labels, ncol=1, frameon=False)
        plt.savefig('../PaperPlots/ABLfhr.png', format="png", bbox_inches='tight')
        plt.show()
        plt.close('all')
        
    elif fig == 'downstreamRMSEGPR':

        PFDatabase = './GPRDatabase/'

        # stresses_normalized
        hTrain = [0.04,0.08,0.12,0.16]
        rTrain = [52,62,72,82,92]

        setsToPlot = np.array([[0.06,67],[0.06,77],[0.14,57],[0.14,87]])

        QoIs = ['u','Iu','Iv','Iw']
        testID = 'intensities_high'

        xList = [0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3.0,3.3,3.6,4.0,5.0,6.0,7.0,9.0,11.0,13.0]
        #xList = [0.3,0.6,0.9,1.2,1.5]

        trainPairs = np.zeros((len(hTrain)*len(rTrain),2))
        cont=0
        for h in hTrain:
            for r in rTrain:
                trainPairs[cont,:] = [h, r]
                cont+=1

        features = ['y','h','r']

        my_dpi = 100
        plt.figure(figsize=(2000/my_dpi, 1000/my_dpi), dpi=my_dpi)
        cont=411

        for QoI in QoIs:
            
            RMSEPlot = pd.DataFrame(columns=['h','r','x','LES','GPR'])

            for x in xList:
                            
                prefix = str(str(x)+'_').replace('.','p')
                
                DF = pd.read_csv('Predictions/'+prefix+testID+'.csv')
                            
                for h, r in setsToPlot:
                    
                    predictions = DF[(DF['h'] == h) & (DF['r'] == r)]
                    
                    RMSEPlot = RMSEPlot.append({'h':h,'r':r,'x':x,'LES':predictions[QoI + ' LES'].to_numpy(),'GPR':predictions[QoI + ' GPR'].to_numpy()},ignore_index=True)
                    
                    #plt.plot(withheldData[QoI],withheldData['y'],color='tab:grey')
                    #plt.plot(y_mean_dev['y_model'],withheldData['y'],label = 'x='+str(x)+'m, h='+str(h)+'m, r='+str(r))
                    #plt.show()
                    
                result = RMSEPlot.groupby(['h', 'x']).apply(RMSE_compute).reset_index()
                
            unique_h = result['h'].unique()
            
            plt.subplot(cont)
            for idx, h in enumerate(sorted(unique_h)):
                subset = result[result['h'] == h]
                plt.plot(subset['x'], subset['RMSE'],marker = 'o', color = colors['{0:.2f}'.format(h)],label='h='+str(h)+'m')
                
            if 'I' in QoI:
                plt.ylim([0,0.01])
                plt.yticks([0,0.005,0.01])
                QoI = r'$'+QoI[0]+'_'+QoI[1]+''
            else:
                plt.ylim([0,0.08])
                plt.yticks([0,0.04,0.08])
                QoI = r'$'+QoI[0]+''
                
            plt.xticks([0.6,2.1,4.0,7.0,13])
            
            if not(cont == 414):
                plt.gca().set_xticklabels([])
            plt.ylabel(QoI+' \ RMSE$')
                
            if cont == 414:
                plt.legend(frameon=False)
            cont+=1
        plt.xlabel('x [m]')
        plt.savefig('../PaperPlots/RMSE_GPR.png', format="png", bbox_inches='tight')
        plt.show()
        
    elif fig == 'downstreamTestGPR':

        PFDatabase = './GPRDatabase/'

        # stresses_normalized
        hTrain = [0.04,0.08,0.12,0.16]
        rTrain = [52,62,72,82,92]

        setsToPlot = np.array([[0.06,67],[0.06,77],[0.14,57],[0.14,87]])

        QoIs = ['u','Iu','Iv','Iw']
        testID = 'intensities_high'

        xList = [0.6,2.1,4.0]

        trainPairs = np.zeros((len(hTrain)*len(rTrain),2))
        cont=0
        for h in hTrain:
            for r in rTrain:
                trainPairs[cont,:] = [h, r]
                cont+=1

        features = ['y','h','r']

        for x in xList:

            my_dpi = 100
            plt.figure(figsize=(2000/my_dpi, 550/my_dpi), dpi=my_dpi)
            cont=1
                
            prefix = str(str(x)+'_').replace('.','p')
                            
            for QoI in QoIs:
            
                RMSEPlot = pd.DataFrame(columns=['h','r','x','LES','GPR'])
                
                DF = pd.read_csv('Predictions/'+prefix+testID+'.csv')
                plt.subplot(1,4,cont)
                
                saw = False
                for h, r in setsToPlot:
                    
                    predictions = DF[(DF['h'] == h) & (DF['r'] == r)]
                    plt.plot(predictions[QoI+' LES'],predictions['y'],color='tab:grey')
                    
                    if saw == False:
                        plt.fill_betweenx(predictions['y'], predictions[QoI+' LES']*0.9, predictions[QoI+' LES']*1.1, color='tab:grey', alpha=0.2,label=r'Reference $\pm$10%')
                        saw = True
                    else:
                        plt.fill_betweenx(predictions['y'], predictions[QoI+' LES']*0.9, predictions[QoI+' LES']*1.1, color='tab:grey', alpha=0.2)
                    plt.plot(predictions[QoI+' GPR'],predictions['y'],label = 'h='+str(h)+'m, r='+str(int(r)), linestyle = linestyles['{0:.0f}'.format(r)], color = colors['{0:.2f}'.format(h)])
                    if 'I' in QoI:
                        plt.xlabel(r'$'+QoI[0]+'_'+QoI[1]+'$')
                    else:
                        plt.xlabel(r'$'+QoI[0]+'[m/s]$')
                    plt.ylim([0,1.5])
                    plt.yticks([0,0.5,1.0,1.5])
                
                if QoI == 'u':
                    plt.xlim([0,18])
                else:
                    plt.xlim([0,0.5])
                
                if cont%4 == 1:
                    plt.ylabel('y[m]')
                else:
                    plt.gca().set_yticklabels([])
                    
                #if cont in [1,2,3,4,5,6,7,8]:
                    #plt.gca().set_xticklabels([])
                    
                if QoI == 'Iw' and x==0.6:
                    plt.legend(frameon=False)
                    
                cont+=1
                
            plt.savefig('../PaperPlots/'+prefix+'Test_GPR.png', format="png", bbox_inches='tight')
            plt.show()
            
        plt.show()
        
    elif fig == 'upstreamTestGPR':

        PFDatabase = './GPRDatabase/'

        # stresses_normalized
        hTrain = [0.04,0.08,0.12,0.16]
        rTrain = [52,62,72,82,92]

        setsToPlot = np.array([[0.06,67],[0.06,77],[0.14,57],[0.14,87]])

        QoIs = ['u','uu','vv','ww','uv']
        testID = 'inflow_stresses'

        xList = [-4.95]

        trainPairs = np.zeros((len(hTrain)*len(rTrain),2))
        cont=0
        for h in hTrain:
            for r in rTrain:
                trainPairs[cont,:] = [h, r]
                cont+=1

        features = ['y','h','r']

        my_dpi = 100
        plt.figure(figsize=(2000/my_dpi, 700/my_dpi), dpi=my_dpi)
        cont=151

        for x in xList:
                
            prefix = str(str(x)+'_').replace('.','p')
                            
            for QoI in QoIs:
            
                RMSEPlot = pd.DataFrame(columns=['h','r','x','LES','GPR'])
                
                DF = pd.read_csv('Predictions/'+prefix+testID+'.csv')
                plt.subplot(cont)
                
                saw = False
                for h, r in setsToPlot:
                    
                    predictions = DF[(DF['h'] == h) & (DF['r'] == r)]
                    plt.plot(predictions[QoI+' LES'],predictions['y'],color='tab:grey')
                    
                    if saw == False:
                        plt.fill_betweenx(predictions['y'], predictions[QoI+' LES']*0.9, predictions[QoI+' LES']*1.1, color='tab:grey', alpha=0.2,label=r'Reference $\pm$10%')
                        saw = True
                    else:
                        plt.fill_betweenx(predictions['y'], predictions[QoI+' LES']*0.9, predictions[QoI+' LES']*1.1, color='tab:grey', alpha=0.2)
                    plt.plot(predictions[QoI+' GPR'],predictions['y'],label = 'h='+str(h)+'m, r='+str(int(r)), linestyle = linestyles['{0:.0f}'.format(r)], color = colors['{0:.2f}'.format(h)])
                    if QoI == 'u':
                        plt.xlabel(r'$'+QoI[0]+'[m/s]$')
                    else:
                        plt.xlabel(r'$\overline{'+QoI[0]+"'"+QoI[1]+"'"+'}[m^2/s^2]$')
                    plt.ylim([0,1.5])
                    plt.yticks([0,0.5,1.0,1.5])
                
                if QoI == 'u':
                    plt.xlim([0,18])
                    plt.xticks([0,6,12,18])
                elif QoI == 'uu':
                    plt.xlim([0,0.03])
                    plt.xticks([0,0.01,0.02,0.03])
                else:
                    plt.xlim([0,0.016])
                    plt.xticks([0,0.005,0.010,0.015])
                
                if cont == 151:
                    plt.ylabel('y[m]')
                else:
                    plt.gca().set_yticklabels([])
                    
                #if cont in [1,2,3,4,5,6,7,8]:
                    #plt.gca().set_xticklabels([])
                    
                if QoI == 'uv':
                    plt.legend(frameon=False)
                    
                cont+=1
                
            plt.savefig('../PaperPlots/'+prefix+'Test_GPR.png', format="png", bbox_inches='tight')
        plt.show()
        plt.close('all')
        
    elif fig == 'UFenvelope':
        
        mat_data = scipy.io.loadmat('2017-10-14 - Terraformer Homogeneous - 1050RPM - No Spires - Unfiltered.mat')
        print(mat_data.keys())
        nPoints = 183


        my_dpi = 100
        plt.figure(figsize=(2000/my_dpi, 550/my_dpi), dpi=my_dpi)
        
        uminUF = np.ones((183,))*100
        uMaxUF = np.zeros((183,))
       
        IuminUF = np.ones((183,))*100
        IuMaxUF = np.zeros((183,))
        
        IvminUF = np.ones((183,))*100
        IvMaxUF = np.zeros((183,))
        
        IwminUF = np.ones((183,))*100
        IwMaxUF = np.zeros((183,))
        
        yUF = np.linspace(1,183,183)/100
        
        #plt.subplot(141)
        for i in range(17):
            uminUF = np.min([uminUF,mat_data['SpatialStatistics'][0][0][0][i][0][:,0],mat_data['SpatialStatistics'][0][0][0][i][1][:,0]], axis=0)
            IuminUF = np.min([IuminUF,mat_data['SpatialStatistics'][0][0][7][i][0][:,0],mat_data['SpatialStatistics'][0][0][7][i][1][:,0]], axis=0)
            IvminUF = np.min([IvminUF,mat_data['SpatialStatistics'][0][0][9][i][0][:,0],mat_data['SpatialStatistics'][0][0][9][i][1][:,0]], axis=0)
            IwminUF = np.min([IwminUF,mat_data['SpatialStatistics'][0][0][8][i][0][:,0],mat_data['SpatialStatistics'][0][0][8][i][1][:,0]], axis=0)
            
            uMaxUF = np.max([uMaxUF,mat_data['SpatialStatistics'][0][0][0][i][0][:,0],mat_data['SpatialStatistics'][0][0][0][i][1][:,0]], axis=0)
            IuMaxUF = np.max([IuMaxUF,mat_data['SpatialStatistics'][0][0][7][i][0][:,0],mat_data['SpatialStatistics'][0][0][7][i][1][:,0]], axis=0)
            IvMaxUF = np.max([IvMaxUF,mat_data['SpatialStatistics'][0][0][9][i][0][:,0],mat_data['SpatialStatistics'][0][0][9][i][1][:,0]], axis=0)
            IwMaxUF = np.max([IwMaxUF,mat_data['SpatialStatistics'][0][0][8][i][0][:,0],mat_data['SpatialStatistics'][0][0][8][i][1][:,0]], axis=0)
            
        #plt.subplot(141)
        #for i in range(17):
            #plt.plot(mat_data['SpatialStatistics'][0][0][0][i][0],np.linspace(1,183,183)/100)
            #plt.plot(mat_data['SpatialStatistics'][0][0][0][i][1],np.linspace(1,183,183)/100)
            
        #plt.subplot(142)
        #for i in range(17):
            #plt.plot(mat_data['SpatialStatistics'][0][0][7][i][0],np.linspace(1,183,183)/100)
            #plt.plot(mat_data['SpatialStatistics'][0][0][7][i][1],np.linspace(1,183,183)/100)
            
        #plt.subplot(143)
        #for i in range(17):
            #plt.plot(mat_data['SpatialStatistics'][0][0][9][i][0],np.linspace(1,183,183)/100)
            #plt.plot(mat_data['SpatialStatistics'][0][0][9][i][1],np.linspace(1,183,183)/100)
            
        #plt.subplot(144)
        #for i in range(17):
            #plt.plot(mat_data['SpatialStatistics'][0][0][7][i][0],np.linspace(1,183,183)/100)
            #plt.plot(mat_data['SpatialStatistics'][0][0][8][i][1],np.linspace(1,183,183)/100)
        
        yMax = 1.83

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

        prefixes = ['0p6_','0p9_','1p2_','1p5_','1p8_','2p1_','2p4_','2p7_','3p0_','3p3_','3p6_','4p0_','5p0_','6p0_','7p0_','9p0_','11p0_','13p0_']
        prefix = ''
        bound = 0.1
        bound = True

        heights = [0.1, 0.2, 0.5]
        yOverhMax = 10

        cont = 0

        y = np.linspace(0.01,3.0,300)

        umin = np.ones((len(y),))*100
        uMax = np.ones((len(y),))*0

        Iumin = np.ones((len(y),))*100
        IuMax = np.ones((len(y),))*0

        Ivmin = np.ones((len(y),))*100
        IvMax = np.ones((len(y),))*0

        Iwmin = np.ones((len(y),))*100
        IwMax = np.ones((len(y),))*0

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
                
                uInterp  = (interp1d(avg_u[:,3], avg_u[:,5])(y))
                IuInterp = (interp1d(rms_u[:,3], rms_u[:,5]/avg_u[:,5])(y))
                IvInterp = (interp1d(rms_v[:,3], rms_v[:,5]/avg_u[:,5])(y))
                IwInterp = (interp1d(rms_w[:,3], rms_w[:,5]/avg_u[:,5])(y))
                
                umin = np.min([umin,uInterp/15], axis=0)
                uMax = np.max([uMax,uInterp/15], axis=0)
                
                Iumin = np.min([Iumin,IuInterp], axis=0)
                IuMax = np.max([IuMax,IuInterp], axis=0)
                
                Ivmin = np.min([Ivmin,IvInterp], axis=0)
                IvMax = np.max([IvMax,IvInterp], axis=0)
                
                Iwmin = np.min([Iwmin,IwInterp], axis=0)
                IwMax = np.max([IwMax,IwInterp], axis=0)

        plt.subplot(1,4,1)
        plt.fill_betweenx(y, umin*15, uMax*15, color='tab:grey',alpha = 0.4)
        plt.fill_betweenx(yUF, uminUF, uMaxUF, color='tab:green',alpha = 0.4)
        plt.xlabel(r'$U/U_\infty$')
        plt.ylabel('y [m]')
        plt.ylim([0,yMax])
        plt.yticks([0,0.5,1.0,1.5])
        #plt.xlim([0,1.2])
        #plt.xticks([0.0,0.6,1.2])

        plt.subplot(1,4,2)
        plt.fill_betweenx(y, Iumin, IuMax, color='tab:grey',alpha = 0.4)
        plt.fill_betweenx(yUF, IuminUF, IuMaxUF, color='tab:green',alpha = 0.4)
        plt.xlabel(r'$I_u$')
        plt.ylim([0,yMax])
        plt.yticks([0,0.5,1.0,1.5])
        plt.xlim([0,0.8])
        plt.xticks([0.0,0.2,0.4,0.6,0.8])
        plt.gca().set_yticklabels([])

        plt.subplot(1,4,3)
        plt.fill_betweenx(y, Ivmin, IvMax, color='tab:grey',alpha = 0.4)
        plt.fill_betweenx(yUF, IvminUF, IvMaxUF, color='tab:green',alpha = 0.4)
        plt.xlabel(r'$I_v$')
        plt.ylim([0,yMax])
        plt.yticks([0,0.5,1.0,1.5])
        plt.xlim([0,0.8])
        plt.xticks([0.0,0.2,0.4,0.6,0.8])
        plt.gca().set_yticklabels([])

        plt.subplot(1,4,4)
        plt.fill_betweenx(y, Iwmin, IwMax, color='tab:grey',alpha = 0.4,label='Database envelope')
        plt.fill_betweenx(yUF, IwminUF, IwMaxUF, color='tab:green',alpha = 0.4,label='UF envelope')
        plt.xlabel(r'$I_w$')
        plt.ylim([0,yMax])
        plt.yticks([0,0.5,1.0,1.5])
        plt.xlim([0,0.8])
        plt.xticks([0.0,0.2,0.4,0.6,0.8])
        plt.gca().set_yticklabels([])
                
        plt.legend(frameon=False)
        plt.savefig('../PaperPlots/UFrange.png', format="png", bbox_inches='tight')
        plt.show()
        plt.close('all')
        
        plt.show()
        
    elif fig == 'codesOptimized':

        PFDatabase = './GPRDatabase/'
        
        fNames = {'HRB_Cat_B':76, 'HRB_Cat_C':112,'HRB_Cat_D':115
                 ,'MRB_Cat_D':116,'MRB_Cat_C':101,'MRB_Cat_B':78
                 ,'LRB_Cat_B':73, 'LRB_Cat_C':94, 'LRB_Cat_D':109}
        
        xList = [0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3.0,3.3,3.6,4.0,5.0,6.0,7.0,9.0,11.0,13.0]
        
        testID = 'intensities_high'
        targetQoIs = ['u','Iu','Iv','Iw']
        
        everyQoI = ['u','Iu','Iv','Iw']
        features = ['y','h','r']
        varNames = [r'$h$',r'$r$',r'$\alpha$',r'$k$',r'$x$']

        yMax= 1.5

        hTrain = [0.04,0.08,0.12,0.16]
        rTrain = [52,62,72,82,92]

        trainPairs = np.zeros((len(hTrain)*len(rTrain),2))
        cont=0
        for h in hTrain:
            for r in rTrain:
                trainPairs[cont,:] = [h, r]
                cont+=1

        devPairs = np.array([[0.06,57],[0.06,87],[0.14,67],[0.14,77]])
        testPairs = np.array([[0.06,67],[0.06,77],[0.14,57],[0.14,87]])
        
        my_dpi = 100
        #plt.figure(figsize=(2000/my_dpi, 1650/my_dpi), dpi=my_dpi)
        plt.figure(figsize=(1650/my_dpi, 2000/my_dpi), dpi=my_dpi)
        case = 0

        try: 
            mode = sys.argv[1]
        except:
            mode = []

        for fName in fNames:
            
            ref_abl = pd.read_csv('TestCases/'+fName+'.dat',sep=',')
            header = list(ref_abl.columns)
            idx = np.argmax(ref_abl['y'].to_numpy())

            Uref = 1.0*ref_abl['u'].iloc[idx]
            yref = ref_abl['y'].iloc[idx]*1.0

            ref_abl['y'] = ref_abl['y']/yref
            ref_abl['u'] = ref_abl['u']/Uref
            for rsc in ['uu','vv','ww','uv','uw','vw']:
                if rsc in header:
                    ref_abl[rsc] = ref_abl[rsc]/(Uref**2)
                    
            effectiveQoIs = []

            for QoI in everyQoI:
                if (QoI in header) and (QoI in targetQoIs):
                    effectiveQoIs.append(QoI)
            
            RMSEValues    = pd.read_csv('TestCases/'+fName+"_pareto_front.csv")
            effectiveQoIs = RMSEValues.columns.tolist()
            decisionVars  = pd.read_csv('TestCases/'+fName+"_decision_variables.csv")
            
            NSGA_results = pd.concat([RMSEValues, decisionVars], axis=1).sort_values(by='u', ascending=False)
            NSGA_results['$h$'] = ((NSGA_results['$h$']*100).round().astype(int)/100).to_numpy()
            NSGA_results['$r$'] = NSGA_results['$r$'].round().astype(int).to_numpy()
            NSGA_results[r'$\alpha$'] = NSGA_results[r'$\alpha$'].round(2).astype(float).to_numpy()
            NSGA_results[r'$k$'] = NSGA_results[r'$k$'].round(2).astype(float).to_numpy()
            NSGA_results['$x$'] = [xList[x] for x in NSGA_results['$x$'].round().astype(int).to_numpy()]
            NSGA_results = NSGA_results.drop_duplicates(subset=varNames, keep='first').reset_index(drop=True)
            
            parameters = NSGA_results.loc[[fNames[fName]], [r'$h$', r'$r$', r'$\alpha$', r'$k$', r'$x$']].to_dict(orient='list')
            parameters['idx'] = [fNames[fName]]
            
            hTemp = parameters[r'$h$'][0]
            rTemp = parameters[r'$r$'][0]
            alphaTemp = parameters[r'$\alpha$'][0]
            kTemp = parameters[r'$k$'][0]
            xTemp = parameters[r'$x$'][0]

            fit_features = pd.DataFrame()
            fit_features['y'] = np.linspace(0.01,1.0,2000)
            fit_features['h'] = hTemp
            fit_features['r'] = rTemp
            fit_features['alpha'] = alphaTemp
            fit_features['k'] = kTemp
            fit_features['x'] = xTemp
            
            #parameters = {r'$h$':parameters[r'$h$'][0],r'$r$':parameters[r'$r$'][0],r'$\alpha$':parameters[r'$\alpha$'][0],r'$k$':parameters[r'$k$'][0],r'$x$':parameters[r'$x$'][0],'idx':fNames[fName]}
            
            #if 'LRB' in fName:
                #case = 1
            #if 'MRB' in fName:
                #case = 5
            #if 'HRB' in fName:
                #case = 9
            #cat_part = '_'.join(fName.split('_')[1:])
                
            if 'Cat_B' in fName:
                case = 1
            if 'Cat_C' in fName:
                case = 2
            if 'Cat_D' in fName:
                case = 3
            cat_part = fName.split('_')[0]
            
            cc = cat_colors.get(cat_part, 'black')
            cont = 0
            
            for QoI in ['u','Iu','Iv','Iw']:
        
                trainPoints = {'h':trainPairs[:,0],'r':trainPairs[:,1],'x':[xTemp]}
                devPoints = {'h':devPairs[:,0],'r':devPairs[:,1],'x':[xTemp]}
                testPoints = {'h':testPairs[:,0],'r':testPairs[:,1],'x':[xTemp]}

                gp = gaussianProcess(trainPoints, devPoints, testPoints, yMax, PFDatabase)
                
                prefix = str(str(xTemp)+'_').replace('.','p')
                directory = prefix+testID
                model = '../GPRModels/'+directory+'_'+QoI+'.pkl'
                
                y_mean = gp.predict(model,fit_features,features,QoI)
                y_mean = y_mean.loc[y_mean['y']<=alphaTemp*np.max(y_mean['y'])]
                y_mean['y'] = y_mean['y']/(y_mean['y'].max())
                
                if QoI == 'u':
                    y_mean['y_model'] = y_mean['y_model']*fit_features['k']
                    y_mean['y_std'] = y_mean['y_std']*fit_features['k']
                    
                plt.subplot(4,3,cont+case)
            
                if QoI in header:
                    plt.plot(ref_abl[QoI],ref_abl['y'],color=cc,label='Target',linewidth=3)
                    plt.fill_betweenx(ref_abl['y'], ref_abl[QoI]*0.9, ref_abl[QoI]*1.1, color=cc, alpha=0.2,label=r'Reference $\pm$10%')
                    
                line = plt.plot(y_mean['y_model'],y_mean['y'],linestyle='--',linewidth=3, color=cc
                        ,label=str(fNames[fName])+' h='+'{0:.2f}'.format(hTemp)+'m, r='+'{0:.0f}'.format(rTemp)+',\n'+r'$\alpha$='+'{0:.2f}'.format(alphaTemp)+r', $k$='+'{0:.2f}'.format(kTemp)+', x='+'{0:.2f}'.format(xTemp)+'m')
                
                if QoI in header:
                    max_x = np.ceil((1.2*max([np.max(ref_abl[QoI]),np.max(y_mean['y_model'])])*10000).astype(int))/10000
                else:
                    max_x = np.ceil(1.2*np.max(y_mean['y_model'])*10000).astype(int)/10000
                
                if QoI in header:
                    max_x = np.ceil((1.2*max([np.max(ref_abl[QoI]),np.max(y_mean['y_model'])])*10000).astype(int))/10000
                else:
                    max_x = np.ceil(1.2*np.max(y_mean['y_model'])*10000).astype(int)/10000
                
                if QoI == 'u':
                    plt.xlim(0,1.2)
                else:
                    plt.xlim(0,1.2*max_x)
                
                plt.xlabel(QoI)
                
                plt.ylim(0,1.0)
                plt.yticks([0.0,0.5,1.0])
                plt.ylabel('y/H')
                
                cont +=3
                
        plt.savefig('../PaperPlots/ASCE.png', format="png", bbox_inches='tight')
        plt.show()
        plt.close('all')
                
        
        
        
        
        
        
        
        