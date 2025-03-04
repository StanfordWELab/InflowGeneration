import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from modelDefinition import *

###########################################################

PFDatabase = './GPRDatabase'

#fName = 'themisABL'
#directories = {'../../InflowValidation/ThemisShortPFh0.06u15r87/':[r'$U_{\infty}=26m/s$',[25000,75000],'tab:blue']}
#reference = {'h':0.06,'r':87,'alpha':0.4,'x':0.9,'hMatch':0.714}
#prefix = '0p9_'
#yMax = 0.4

#fName = 'LRB_Cat1_scale1to50'
#directories = {'../../CatherineABLs/LRB_Cat1_scale1to50/':[r'$U_{\infty}=45m/s$',[25000,75000],'tab:blue']}
#reference = {'h':0.12,'r':79,'alpha':0.18,'x':11.0,'hMatch':0.4}
#yMax = 0.18

#fName = 'LRB_Cat1_scale1to25'
#directories = {'../../CatherineABLs/LRB_Cat1_scale1to25/':[r'$U_{\infty}=45m/s$',[25000,75000],'tab:blue']}
#reference = {'h':0.04,'r':75,'alpha':0.36,'x':3.3,'hMatch':0.4}
#yMax = 0.36

fName = 'LRB_Cat1_scale1to10'
directories = {'../../CatherineABLs/LRB_Cat1_scale1to10/':[r'$U_{\infty}=45m/s$',[25000,75000],'tab:blue']}
reference = {'h':0.16,'r':64,'alpha':0.90,'x':9.0,'hMatch':0.666}
yMax = 0.9

###########################################################

intensitiesModelID = 'intensities'
uncertainty = False

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
        
fit_features = pd.DataFrame()
fit_features['y'] = np.linspace(0.01,1.0,2000)
fit_features['x'] = reference['x']
fit_features['h'] = reference['h']
fit_features['r'] = reference['r']
fit_features['alpha'] = reference['alpha']

trainPoints = {'h':trainPairs[:,0],'r':trainPairs[:,1],'x':[reference['x']]}
devPoints = {'h':devPairs[:,0],'r':devPairs[:,1],'x':[reference['x']]}
testPoints = {'h':testPairs[:,0],'r':testPairs[:,1],'x':[reference['x']]}

gp = gaussianProcess(trainPoints, devPoints, testPoints, yMax, PFDatabase)

prefix = str(str(reference['x'])+'_').replace('.','p')
directory = prefix+intensitiesModelID
    
ref_abl = pd.read_csv('TestCases/'+fName+'.dat',sep=',')
header = list(ref_abl.columns)
idx = np.argmax(ref_abl['y'].to_numpy())

yref = ref_abl['y'].iloc[idx]*1.0

ref_abl['y'] = ref_abl['y']/yref
ref_abl['u'] = ref_abl['u']
  
model = '../GPRModels/'+directory+'_u.pkl'
y_mean = gp.predict(model,fit_features,features,'u')
y_mean = y_mean.loc[y_mean['y']<=reference['alpha']*np.max(y_mean['y'])]
y_mean['y'] = y_mean['y']/(y_mean['y'].max())
#y_mean['y_model'] = y_mean['y_model']

U_ABL_dim = (interp1d(ref_abl['y'], ref_abl['u'])(reference['hMatch'])).item()
U_TIG_dim = (interp1d(y_mean['y'], y_mean['y_model'])(reference['hMatch'])).item()

Uscaling = U_ABL_dim/(U_TIG_dim)

my_dpi = 100
plt.figure(figsize=(2260/my_dpi, 1300/my_dpi), dpi=my_dpi)


for fold in directories:
    
    cont=1
        
    resultsDF = pd.DataFrame()

    avg_u = np.loadtxt(fold+prefix+'avg_u.'+str(directories[fold][1][1]).zfill(8)+'.collapse_width.dat',skiprows = 3)

    Umag = np.loadtxt(fold+prefix+'mag_u.'+str(directories[fold][1][1]).zfill(8)+'.collapse_width.dat',skiprows = 3)

    rms_u = np.loadtxt(fold+prefix+'rms_u.'+str(directories[fold][1][1]).zfill(8)+'.collapse_width.dat',skiprows = 3)
    rms_v = np.loadtxt(fold+prefix+'rms_v.'+str(directories[fold][1][1]).zfill(8)+'.collapse_width.dat',skiprows = 3)
    rms_w = np.loadtxt(fold+prefix+'rms_w.'+str(directories[fold][1][1]).zfill(8)+'.collapse_width.dat',skiprows = 3)

    uv = np.loadtxt(fold+prefix+'uv.'+str(directories[fold][1][1]).zfill(8)+'.collapse_width.dat',skiprows = 3)
    
    resultsDF['y'] = avg_u[:,3]/yref
    resultsDF['u'] = avg_u[:,5]
    resultsDF['Iu'] = rms_u[:,5]/Umag[:,5]
    resultsDF['Iv'] = rms_v[:,5]/Umag[:,5]
    resultsDF['Iw'] = rms_w[:,5]/Umag[:,5]
    
    for QoI in ['u','Iu','Iv','Iw']:
        
        plt.subplot(1,4,cont)
            
        plt.plot(resultsDF[QoI],resultsDF['y'], label = directories[fold][0], color=directories[fold][2],linewidth=2)
        plt.ylim([0,1.0])

        cont += 1
cont = 1
for QoI in ['u','Iu','Iv','Iw']:
    plt.subplot(1,4,cont)
    
    model = '../GPRModels/'+directory+'_'+QoI+'.pkl'
    
    y_mean = gp.predict(model,fit_features,features,QoI)
    y_mean = y_mean.loc[y_mean['y']<=reference['alpha']*np.max(y_mean['y'])]
    y_mean['y'] = y_mean['y']/(y_mean['y'].max())
    
    if QoI == 'u':
        y_mean['y_model'] = y_mean['y_model']*Uscaling
        y_mean['y_std'] = y_mean['y_std']*Uscaling
    
    plt.plot(y_mean['y_model'],y_mean['y'],label=r'Optimization prediction, $U_{\infty}=15m/s$',linewidth=2,color='tab:green')
                    
    if (QoI in header):
        plt.plot(ref_abl[QoI],ref_abl['y'],color='tab:red',label='Target',linewidth=2)
        plt.fill_betweenx(ref_abl['y'], ref_abl[QoI]*0.9, ref_abl[QoI]*1.1, color='tab:red', alpha=0.2,label=r'Reference $\pm$10%')
    cont +=1
    
plt.legend()
plt.show()