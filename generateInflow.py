import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from modelDefinition import *

###########################################################

PFDatabase = './GPRDatabase'
fName = 'themisABL'

# stresses_normalized
hTrain = [0.04,0.08,0.12,0.16]
rTrain = [52,62,72,82,92]
testID = 'intensities'

devPairs = np.array([[0.06,57],[0.06,87],[0.14,67],[0.14,77]])
testPairs = np.array([[0.06,67],[0.06,77],[0.14,57],[0.14,87]])

uncertainty = True

###########################################################

xList = [0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3.0,3.3,3.6,4.0,5.0,6.0,7.0,9.0,11.0,13.0]

QoIs = ['u','Iu','Iv','Iw']

trainPairs = np.zeros((len(hTrain)*len(rTrain),2))
cont=0
for h in hTrain:
    for r in rTrain:
        trainPairs[cont,:] = [h, r]
        cont+=1

features = ['y','h','r']

yMax= 1.0
    
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
        
escape = 't'
while not(escape == 'y'):
    more = 'a'
    
    listSolutions = input('Which h, r, alpha, and x values do you wanna use to generate the inflow?\n')
    try:
        listSolutions = list(map(float, listSolutions.split()))
    except:
        listSolutions = [None]
        
    if len(listSolutions) == 4:
        h     = listSolutions[0]
        r     = listSolutions[1]
        alpha = listSolutions[2]
        x     = listSolutions[3]
        
        if (h<0.04) or (h>0.16) or (r<52) or (r>92) or (alpha<0.1) or alpha>1.0 or not(x in xList):
            print('Bad choice buddy...')
            if (h<0.04) or (h>0.16):
                print('h should be in [0.04,0.16]!')
            if (r<52) or (r>92):
                print('r should be in [52,92]!')
            if (alpha<0.1) or (alpha>1.0):
                print('alpha should be in ]0.1,1.0],!')
            if not(x in xList):
                print('x should be in '+str(xList)+'!')
            print('Try again\n')
        else:
            escape = 'y'

        
my_dpi = 100
plt.figure(figsize=(2260/my_dpi, 1300/my_dpi), dpi=my_dpi)
cont=1
for QoI in ['u','Iu','Iv','Iw']:

    fit_features = pd.DataFrame()
    fit_features['y'] = np.linspace(0.01,1.0,2000)
    fit_features['x'] = x
    fit_features['h'] = h
    fit_features['r'] = r
    fit_features['alpha'] = alpha

    trainPoints = {'h':trainPairs[:,0],'r':trainPairs[:,1],'x':[x]}
    devPoints = {'h':devPairs[:,0],'r':devPairs[:,1],'x':[x]}
    testPoints = {'h':testPairs[:,0],'r':testPairs[:,1],'x':[x]}

    gp = gaussianProcess(trainPoints, devPoints, testPoints, yMax, PFDatabase)
    
    prefix = str(str(x)+'_').replace('.','p')
    directory = prefix+testID
    model = '../GPRModels/'+directory+'_'+QoI+'.pkl'
    
    y_mean = gp.predict(model,fit_features,features)
    y_mean = y_mean.loc[y_mean['y']<=alpha*np.max(y_mean['y'])]
    y_mean['y'] = y_mean['y']/(y_mean['y'].max())
    
    if QoI == 'u':
        y_mean['y_model'] = y_mean['y_model']/(y_mean['y_model'].iloc[-1])
        y_mean['y_std'] = y_mean['y_std']/(y_mean['y_model'].iloc[-1])
        
    plt.subplot(2,2,cont)
    
    if (QoI in header):
        plt.plot(ref_abl[QoI],ref_abl['y'],color='tab:red',label='Target',linewidth=3)
        plt.fill_betweenx(ref_abl['y'], ref_abl[QoI]*0.9, ref_abl[QoI]*1.1, color='tab:red', alpha=0.2,label=r'Reference $\pm$10%')
        
    line = plt.plot(y_mean['y_model'],y_mean['y'],linestyle='--',linewidth=3
            ,label=r'x='+'{0:.2f}'.format(x)+'m,h='+'{0:.2f}'.format(h)+'m'+r'm,$\alpha$='+'{0:.2f}'.format(alpha)+r',r='+'{0:.2f}'.format(r))
    
    if QoI in header:
        max_x = np.ceil((1.2*max([np.max(ref_abl[QoI]),np.max(y_mean['y_model'])])*10000).astype(int))/10000
    else:
        max_x = np.ceil(1.2*np.max(y_mean['y_model'])*10000).astype(int)/10000
    
    plt.xlim(0,1.2*max_x)
    plt.xlabel(QoI)
    
    plt.ylim(0,1)
    plt.yticks([0.5,1.0])
    
    if QoI=='u'or QoI == 'Iv':
        plt.ylabel('y/H')
    else:
        plt.gca().set_yticklabels([])
    
    if uncertainty == True:
        plt.fill_betweenx(y_mean['y'], y_mean['y_model']-2*y_mean['y_std'], y_mean['y_model']+2*y_mean['y_std'], color=line[0].get_color(), alpha=0.2)
    
    cont +=1

plt.legend(frameon=False)
plt.show()
plt.close('all')



            
            
            
            
            
            
            
            
            
            