from datetime import datetime
import joblib
import sys
import scipy as sp
from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, WhiteKernel
from scipy.interpolate import interp1d
from joblib         import Parallel, delayed
from modelDefinition import *
from statsmodels.graphics.tsaplots import acf
from scipy.interpolate import interp1d

def expCurve(t, tau):
    return np.exp(-t/tau)

def readPressure(counter, directory):
    filename = directory + '/probes.' + str(counter).zfill(8) + '.pcd'
    print(str(counter) + ' ' + filename)
    
    data = np.loadtxt(filename, skiprows=1)
            
    return data

def computeAutocorrelation(index, component):
                
    return acf(uPrime[:,index,component], nlags = 1000, fft=True)

def vonKarmanSpectrum(component,freq,timeScale):
    
    n = freq*timeScale
    
    if component == 0:
        S_adim = 4*n/((1.0+70.8*n*n)**(5.0/6.0))
    
    if component == 1 or component == 2:
        S_adim = 4*n*(1+755.2*n*n)/((1.0+283.2*n*n)**(11.0/6.0))

    return S_adim

PFDatabase = './GPRDatabase'

cases = {'LRB_Cat1_scale1to50':{'directory':{'../../CatherineABLs/LRB_Cat1_scale1to50/':[r'$U_{\infty}=45m/s$',[50000,150000,0.002],'tab:blue']}
                              ,'reference':{'h':0.12,'r':79,'alpha':0.18,'x':11.0,'hMatch':0.4}
                              ,'yMax':0.18}
        ,'LRB_Cat1_scale1to25':{'directory':{'../../CatherineABLs/LRB_Cat1_scale1to25/':[r'$U_{\infty}=45m/s$',[50000,150000,0.002],'tab:blue']}
                              ,'reference':{'h':0.04,'r':75,'alpha':0.36,'x':3.3,'hMatch':0.4}
                              ,'yMax':0.36}
        ,'LRB_Cat1_scale1to10':{'directory':{'../../CatherineABLs/LRB_Cat1_scale1to10/':[r'$U_{\infty}=45m/s$',[50000,150000,0.002],'tab:blue']}
                              ,'reference':{'h':0.16,'r':64,'alpha':0.90,'x':9.0,'hMatch':0.666}
                              ,'yMax':0.9}}

#cases = {'themisABL':{'directory':{'../../InflowValidation/RefSmall/':[r'$U_{\infty}=15m/s$',[20000,75000,0.004],'tab:blue']}
                              #,'reference':{'h':0.06,'r':87,'alpha':0.4,'k':1.5,'x':1.5,'hMatch':0.714}
                              #,'yMax':0.40}}


###########################################################

plt.close('all')

intensitiesModelID = 'intensities'
uncertainty = False

nPlots = len(cases)

features = ['y','h','r']

xList = [0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3.0,3.3,3.6,4.0,5.0,6.0,7.0,9.0,11.0,13.0]

hTrain = [0.04,0.08,0.12,0.16]
rTrain = [52,62,72,82,92]

devPairs = np.array([[0.06,57],[0.06,87],[0.14,67],[0.14,77]])
testPairs = np.array([[0.06,67],[0.06,77],[0.14,57],[0.14,87]])

trainPairs = np.zeros((len(hTrain)*len(rTrain),2))
cont=0
for hTemp in hTrain:
    for rTemp in rTrain:
        trainPairs[cont,:] = [hTemp, rTemp]
        cont+=1

#my_dpi = 100
#plt.figure(figsize=(2260/my_dpi, 1300/my_dpi), dpi=my_dpi)

#outerCont = 0
#for fName in cases:
        
    #resultsDF = pd.DataFrame()
    #fit_features = pd.DataFrame()
    
    #reference = cases[fName]['reference']
        
    #fit_features['y'] = np.linspace(0.01,1.0,2000)
    #fit_features['x'] = reference['x']
    #fit_features['h'] = reference['h']
    #fit_features['r'] = reference['r']
    #fit_features['alpha'] = reference['alpha']

    #trainPoints = {'h':trainPairs[:,0],'r':trainPairs[:,1],'x':[reference['x']]}
    #devPoints = {'h':devPairs[:,0],'r':devPairs[:,1],'x':[reference['x']]}
    #testPoints = {'h':testPairs[:,0],'r':testPairs[:,1],'x':[reference['x']]}

    #gp = gaussianProcess(trainPoints, devPoints, testPoints, cases[fName]['yMax'], PFDatabase)

    #prefix = str(str(reference['x'])+'_').replace('.','p')
    #directory = prefix+intensitiesModelID
        
    #ref_abl = pd.read_csv('TestCases/'+fName+'.dat',sep=',')
    #header = list(ref_abl.columns)
    #idx = np.argmax(ref_abl['y'].to_numpy())

    #yref = ref_abl['y'].iloc[idx]*1.0

    #ref_abl['y'] = ref_abl['y']/yref
    #ref_abl['u'] = ref_abl['u']
    
    #model = '../GPRModels/'+directory+'_u.pkl'
    #y_mean = gp.predict(model,fit_features,features,'u')
    #y_mean = y_mean.loc[y_mean['y']<=reference['alpha']*np.max(y_mean['y'])]
    #y_mean['y'] = y_mean['y']/(y_mean['y'].max())
    ##y_mean['y_model'] = y_mean['y_model']

    #U_ABL_dim = (interp1d(ref_abl['y'], ref_abl['u'])(reference['hMatch'])).item()
    #U_TIG_dim = (interp1d(y_mean['y'], y_mean['y_model'])(reference['hMatch'])).item()

    #Uscaling = U_ABL_dim/(U_TIG_dim)
        
    #cont=1+outerCont
    
    #fold = list(cases[fName]['directory'].keys())[0]
    #directoryDict = cases[fName]['directory']

    #avg_u = np.loadtxt(fold+prefix+'avg_u.'+str(directoryDict[fold][1][1]).zfill(8)+'.collapse_width.dat',skiprows = 3)

    #Umag = np.loadtxt(fold+prefix+'mag_u.'+str(directoryDict[fold][1][1]).zfill(8)+'.collapse_width.dat',skiprows = 3)

    #rms_u = np.loadtxt(fold+prefix+'rms_u.'+str(directoryDict[fold][1][1]).zfill(8)+'.collapse_width.dat',skiprows = 3)
    #rms_v = np.loadtxt(fold+prefix+'rms_v.'+str(directoryDict[fold][1][1]).zfill(8)+'.collapse_width.dat',skiprows = 3)
    #rms_w = np.loadtxt(fold+prefix+'rms_w.'+str(directoryDict[fold][1][1]).zfill(8)+'.collapse_width.dat',skiprows = 3)

    #uv = np.loadtxt(fold+prefix+'uv.'+str(directoryDict[fold][1][1]).zfill(8)+'.collapse_width.dat',skiprows = 3)
    
    #resultsDF['y'] = avg_u[:,3]/yref
    #resultsDF['u'] = avg_u[:,5]
    #resultsDF['Iu'] = rms_u[:,5]/Umag[:,5]
    #resultsDF['Iv'] = rms_v[:,5]/Umag[:,5]
    #resultsDF['Iw'] = rms_w[:,5]/Umag[:,5]
    
    #for QoI in ['u','Iu','Iv','Iw']:
        
        #plt.subplot(nPlots,4,cont)
            
        #plt.plot(resultsDF[QoI],resultsDF['y'], label = directoryDict[fold][0], color=directoryDict[fold][2],linewidth=2)
        #plt.ylim([0,1.0])

        #cont += 1
        
    #cont = 1+outerCont
    #for QoI in ['u','Iu','Iv','Iw']:
        #plt.subplot(nPlots,4,cont)
        
        #model = '../GPRModels/'+directory+'_'+QoI+'.pkl'
        
        #y_mean = gp.predict(model,fit_features,features,QoI)
        #y_mean = y_mean.loc[y_mean['y']<=reference['alpha']*np.max(y_mean['y'])]
        #y_mean['y'] = y_mean['y']/(y_mean['y'].max())
        
        #if QoI == 'u':
            #y_mean['y_model'] = y_mean['y_model']*Uscaling
            #y_mean['y_std'] = y_mean['y_std']*Uscaling
        
        #plt.plot(y_mean['y_model'],y_mean['y'],label=r'Optimization prediction, $U_{\infty}=15m/s$',linewidth=2,color='tab:green')
                        
        #if (QoI in header):
            #plt.plot(ref_abl[QoI],ref_abl['y'],color='tab:red',label=fName,linewidth=2)
            #plt.fill_betweenx(ref_abl['y'], ref_abl[QoI]*0.9, ref_abl[QoI]*1.1, color='tab:red', alpha=0.2,label=r'Reference $\pm$10%')
        #cont +=1
    
    #outerCont+=4
    #plt.legend()
    
#plt.legend()
#plt.savefig('./ASCE/'+fName+'.png', bbox_inches='tight')
#plt.show()
#plt.close('all')

my_dpi = 100
plt.figure(figsize=(2260/my_dpi, 1300/my_dpi), dpi=my_dpi)

outerCont = 0
for fName in cases:
    
    fold = list(cases[fName]['directory'].keys())[0]
    directoryDict = cases[fName]['directory']

    data_list = Parallel(n_jobs=12)(delayed(readPressure)(counter, fold+'Probes')
                    for counter in range(directoryDict[fold][1][0],directoryDict[fold][1][1]))

    coords = np.loadtxt(fold+'Probes/probes.pxyz', skiprows=1, usecols=(1,2,3))

    data_array = np.array(data_list)
    meanVelocities = np.mean(data_array,axis = 0)

    uPrime = data_array - meanVelocities
    
    heights = np.unique(coords[:,1])
       
    cont = 1+outerCont
    for comp in [0,1,2,3]:

        for y in heights:

            idx = np.where(np.abs(coords[:,1]-y)<1e-4)[0]

            f = 0
            pxx = 0
            data = 0

            for i in idx:
                
                temp_f, temp_pxx = sp.signal.welch(uPrime[:,i,comp],fs=1/directoryDict[fold][1][2])
                f += temp_f[1:]/len(idx)
                pxx += temp_pxx[1:]/len(idx)
                
                data += computeAutocorrelation(i, comp)/len(idx)
            
            meanU  = np.mean(meanVelocities[idx,0])
            sigma2 = np.mean((np.var(uPrime[:,:,comp],axis=0, ddof=1))[idx])
            #print(meanU)
            #input()
            
            steps = np.arange(0,len(data),1)*directoryDict[fold][1][2]            
            tScale = interp1d(steps,data)(np.exp(-1))
            
            #print(vks)
            #print(f)
            #input()

            #print(f,pxx)
            plt.subplot(nPlots,4,cont)
            #plt.loglog(f,pxx,label = 'y='+str(y)+'m'+',T='+str(tScale)+'s')
            line1, = plt.loglog(f,pxx,label = 'y='+str(y)+'m')
            if cont%4!=0:
                vks = vonKarmanSpectrum(comp,f,tScale)*sigma2/f
                plt.loglog(f,vks,color = line1.get_color(), linestyle='--')
            if cont%4 == 1:
                plt.xlabel(r'$E_u$')
            elif cont%4 == 2:
                plt.gca().set_yticklabels([])
                plt.xlabel(r'$E_v$')
            elif cont%4 == 3:
                plt.gca().set_yticklabels([])
                plt.xlabel(r'$E_w$')
            elif cont%4 == 0:
                plt.gca().set_yticklabels([])
                plt.xlabel(r'$E_p$')
        cont+=1
        #plt.show()
        
    
    plt.legend()
    outerCont+=4

plt.savefig('ABLSpectra.png', bbox_inches='tight')
plt.show()
plt.close('all')
