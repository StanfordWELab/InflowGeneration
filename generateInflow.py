import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from modelDefinition import *

###########################################################

PFDatabase = './GPRDatabase'

#### themisABL setup ####
#reference = {'fName':'themisABL'
            #,'h':0.06,'r':87,'alpha':0.4,'k':1.5,'x':0.9,'hMatch':0.714}

#### themisABL_alpha setup ####
#reference = {'fName':'themisABL_alpha'
            #,'h':0.04,'r':92,'alpha':0.20,'k':1.5,'x':0.3,'hMatch':0.714}

#### TPU_highrise_14_middle_dimensional setup ####
reference = {'fName':'TPU_highrise_14_middle_dimensional'
             ,'h':0.04,'r':92,'alpha':0.7,'k':1.11,'x':2.7,'hMatch':0.714}

##### LRB Cat1 scale 1:50 ####
#reference = {'fName':LRB_Cat1_scale1to50
             #,'h':0.12,'r':79,'alpha':0.18,'k':1.62,'x':11.0,'hMatch':0.4}

##### LRB Cat1 scale 1:25 ####
#fName = 'LRB_Cat1_scale1to25'
#reference = {'fName':'LRB_Cat1_scale1to25'
             #,'h':0.04,'r':75,'alpha':0.36,'k':1.36,'x':3.3,'hMatch':0.666}

##### LRB Cat1 scale 1:10 ####
#reference = {'fName':'LRB_Cat1_scale1to10'
             #,'h':0.16,'r':64,'alpha':0.90,'k':1.25,'x':9.0,'hMatch':0.666}

##### MRB Cat2 scale 1:100 ####
#reference = {'fName':'MRB_Cat2_scale1to100'
             #,'h':0.04,'r':92,'alpha':0.45,'k':1.39,'x':3.30,'hMatch':0.666}

#### MRB Cat3 scale 1:100 ####
reference = {'fName':'MRB_Cat3_scale1to100'
             ,'h':0.08,'r':92,'alpha':0.45,'k':1.56,'x':1.50,'hMatch':0.666}

########################################################

intensitiesModelID = 'intensities'
inflowModelID = 'inflow_stresses'
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

yMax= 1.0

fit_features = pd.DataFrame()
fit_features['y'] = np.linspace(0.01,1.0,2000)
fit_features['x'] = reference['x']
fit_features['h'] = reference['h']
fit_features['alpha'] = reference['alpha']
fit_features['r'] = reference['r']
fit_features['k'] = reference['k']

trainPoints = {'h':trainPairs[:,0],'r':trainPairs[:,1],'x':[reference['x']]}
devPoints = {'h':devPairs[:,0],'r':devPairs[:,1],'x':[reference['x']]}
testPoints = {'h':testPairs[:,0],'r':testPairs[:,1],'x':[reference['x']]}

gp = gaussianProcess(trainPoints, devPoints, testPoints, yMax, PFDatabase)

prefix = str(str(reference['x'])+'_').replace('.','p')
directory = prefix+intensitiesModelID
    
ref_abl = pd.read_csv('TestCases/'+reference['fName']+'.dat',sep=',')
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

Uscaling = U_ABL_dim/(U_TIG_dim)*yref/reference['alpha']

print('===========================================================================')
print('Scaling velocity from GPR to reference ABL: '+str(np.round(Uscaling,3))+'m/s')
print('Reference velocity at '+str(np.round(reference['hMatch']*yref,3))+'m reference ABL height: '+str(np.round(U_ABL_dim,3))+'m/s')
print('===========================================================================')


my_dpi = 100
plt.figure(figsize=(2260/my_dpi, 1300/my_dpi), dpi=my_dpi)
cont=1

for QoI in ['u','Iu','Iv','Iw']:
    model = '../GPRModels/'+directory+'_'+QoI+'.pkl'
    
    y_mean = gp.predict(model,fit_features,features,QoI)
    y_mean = y_mean.loc[y_mean['y']<=reference['alpha']*np.max(y_mean['y'])]
    y_mean['y'] = y_mean['y']/(y_mean['y'].max())
    
    if QoI == 'u':
        y_mean['y_model'] = y_mean['y_model']*Uscaling
        y_mean['y_std'] = y_mean['y_std']*Uscaling
        
    plt.subplot(2,4,cont)
    
    if (QoI in header):
        plt.plot(ref_abl[QoI],ref_abl['y'],color='tab:red',label='Target',linewidth=3)
        plt.fill_betweenx(ref_abl['y'], ref_abl[QoI]*0.9, ref_abl[QoI]*1.1, color='tab:red', alpha=0.2,label=r'Reference $\pm$10%')
        
    line = plt.plot(y_mean['y_model'],y_mean['y'],linestyle='--',linewidth=3
            ,label=r'x='+'{0:.2f}'.format(reference['x'])+'m,h='+'{0:.2f}'.format(reference['h'])+'m'+r'm,$\alpha$='+'{0:.2f}'.format(reference['alpha'])+r',r='+'{0:.2f}'.format(reference['r']))
    
    if QoI in header:
        max_x = np.ceil((1.2*max([np.max(ref_abl[QoI]),np.max(y_mean['y_model'])])*10000).astype(int))/10000
    else:
        max_x = np.ceil(1.2*np.max(y_mean['y_model'])*10000).astype(int)/10000
        
    if uncertainty == True:
        plt.fill_betweenx(y_mean['y'], y_mean['y_model']-2*y_mean['y_std'], y_mean['y_model']+2*y_mean['y_std'], color=line[0].get_color(), alpha=0.2)
    
    
    plt.xlim(0,1.1*max_x)
    plt.xlabel(QoI)
    
    plt.ylim(0,1)
    plt.yticks([0.25,0.5,0.75,1.0])
    
    if QoI=='u'or QoI == 'Iv':
        plt.ylabel('y/H')
    else:
        plt.gca().set_yticklabels([])
        
    plt.subplot(2,4,cont+4)
    
    y_mean = gp.predict(model,fit_features,features,QoI)
    y_mean['y'] = y_mean['y']/(y_mean['y'].max())
    
    if QoI == 'u':
        y_mean['y_model'] = y_mean['y_model']*Uscaling
        y_mean['y_std'] = y_mean['y_std']*Uscaling
    
    if (QoI in header):
        plt.plot(ref_abl[QoI],ref_abl['y']*yref,color='tab:red',label='Target',linewidth=3)
        plt.fill_betweenx(ref_abl['y']*yref, ref_abl[QoI]*0.9, ref_abl[QoI]*1.1, color='tab:red', alpha=0.2,label=r'Reference $\pm$10%')
        
    line = plt.plot(y_mean['y_model'],y_mean['y'],linestyle='--',linewidth=3
            ,label=r'x='+'{0:.2f}'.format(reference['x'])+'m,h='+'{0:.2f}'.format(reference['h'])+'m'+r'm,$\alpha$='+'{0:.2f}'.format(reference['alpha'])+r',r='+'{0:.2f}'.format(reference['r']))
    
    if QoI in header:
        max_x = np.ceil((1.2*max([np.max(ref_abl[QoI]),np.max(y_mean['y_model'])])*10000).astype(int))/10000
    else:
        max_x = np.ceil(1.2*np.max(y_mean['y_model'])*10000).astype(int)/10000
        
    if uncertainty == True:
        plt.fill_betweenx(y_mean['y'], y_mean['y_model']-2*y_mean['y_std'], y_mean['y_model']+2*y_mean['y_std'], color=line[0].get_color(), alpha=0.2)
    
    
    plt.xlim(0,1.1*max_x)
    plt.xlabel(QoI)
    
    if QoI=='u'or QoI == 'Iv':
        plt.ylabel('y[m]')
    else:
        plt.gca().set_yticklabels([])
    
    cont +=1

plt.suptitle('Chosen setup, dimension vs adimensional y')

plt.legend(frameon=False)
plt.savefig('TestCases/'+reference['fName']+'.png', bbox_inches='tight')
plt.show()
plt.close('all')

yMax = 1.5
        
my_dpi = 100
plt.figure(figsize=(2260/my_dpi, 1300/my_dpi), dpi=my_dpi)

for x in [-4.95,-2.85]:
    cont=1
                
    trainPoints = {'h':trainPairs[:,0],'r':trainPairs[:,1],'x':[x]}
    devPoints = {'h':devPairs[:,0],'r':devPairs[:,1],'x':[x]}
    testPoints = {'h':testPairs[:,0],'r':testPairs[:,1],'x':[x]}
        
    gp = gaussianProcess(trainPoints, devPoints, testPoints, yMax, PFDatabase, np.linspace(0.01,1.0,100))

    outputDF = pd.DataFrame()
    fit_features = pd.DataFrame()

    fit_features['y'] = np.linspace(0.01/yMax,1.0,1501)
    fit_features['x'] = x
    fit_features['h'] = reference['h']
    fit_features['r'] = reference['r']
        
    outputDF['x'] = np.ones((len(fit_features['y'].to_numpy()),))*x
    outputDF['y'] = fit_features['y'].to_numpy()*yMax*reference['alpha']/yref
    outputDF['z'] = np.zeros((len(fit_features['y'].to_numpy()),))
    outputDF['y-velocity'] = np.zeros((len(fit_features['y'].to_numpy()),))
    outputDF['z-velocity'] = np.zeros((len(fit_features['y'].to_numpy()),))
    outputDF['uw-reynolds-stress'] = np.zeros((len(fit_features['y'].to_numpy()),))
    outputDF['vw-reynolds-stress'] = np.zeros((len(fit_features['y'].to_numpy()),))

    for QoI in ['u','uu','vv','ww','uv']:
        
        prefix = str(str(x)+'_').replace('.','p')
        directory = prefix+inflowModelID
        model = '../GPRModels/'+directory+'_'+QoI+'.pkl'
        
        y_mean = gp.predict(model,fit_features,features,QoI)
        y_mean['y'] = y_mean['y']/(y_mean['y'].max())
        
        if QoI == 'u':
            y_mean['y_model'] = y_mean['y_model']/(y_mean['y_model'].iloc[-1])*Uscaling
            y_mean['y_std'] = y_mean['y_std']/(y_mean['y_model'].iloc[-1])*Uscaling
        else:
            y_mean['y_model'] = y_mean['y_model']*Uscaling*Uscaling
            y_mean['y_std'] = y_mean['y_std']*Uscaling*Uscaling
            
        plt.subplot(2,3,cont)
        
        if x == -4.95:
            lab = 'inflow_input'
        elif x == -2.85:
            lab = 'ALF_input'
            
        if QoI == 'u':
            outputDF['x-velocity'] = y_mean['y_model'].to_numpy()
            line = plt.plot(outputDF['x-velocity'],outputDF['y'],linewidth=2,label=lab)
        elif QoI == 'uu':
            outputDF['uu-reynolds-stress'] = np.abs(y_mean['y_model'].to_numpy())
            line = plt.plot(outputDF['uu-reynolds-stress'],outputDF['y'],linewidth=2,label=lab)
        elif QoI == 'vv':
            outputDF['vv-reynolds-stress'] = np.abs(y_mean['y_model'].to_numpy())
            line = plt.plot(outputDF['vv-reynolds-stress'],outputDF['y'],linewidth=2,label=lab)
        elif QoI == 'ww':
            outputDF['ww-reynolds-stress'] = np.abs(y_mean['y_model'].to_numpy())
            line = plt.plot(outputDF['ww-reynolds-stress'],outputDF['y'],linewidth=2,label=lab)
        elif QoI == 'uv':
            outputDF['uv-reynolds-stress'] = -np.abs(y_mean['y_model'].to_numpy())
            line = plt.plot(-outputDF['uv-reynolds-stress'],outputDF['y'],linewidth=2,label=lab)
        
        max_x = np.ceil(1.2*np.max(y_mean['y_model'])*10000).astype(int)/10000
        
        plt.xlim(0,1.1*max_x)
        
        if QoI=='u':
            plt.ylabel('y[m]')
            plt.xlabel('u[m/s]')
        else:
            plt.ylabel('y[m]')
            plt.xlabel(QoI+r'$[m^2/s^2]$')
        
        #if uncertainty == True:
            #plt.fill_betweenx(y_mean['y'], y_mean['y_model']-2*y_mean['y_std'], y_mean['y_model']+2*y_mean['y_std'], color=line[0].get_color(), alpha=0.2)
        
        cont +=1
    
    outputDF['y'] = outputDF['y']
    outputDF[['x','y','z','x-velocity','y-velocity','z-velocity','uu-reynolds-stress','vv-reynolds-stress','ww-reynolds-stress','uv-reynolds-stress','uw-reynolds-stress','vw-reynolds-stress']].to_csv('TestCases/'+reference['fName']+'_'+lab+'.txt',sep='\t',index=False)

plt.suptitle('Chosen setup')
plt.legend(frameon=False)
plt.show()
plt.close('all')
            
            
            
            
            
            
            
            
            