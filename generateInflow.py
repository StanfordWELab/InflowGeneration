# Script to generate inflow profiles using Gaussian Process Regression (GPR)
# and prepare corresponding case geometry and boundary‐condition files.

# ===== Imports & Library Setup =====
import os
import sys
import joblib
# Import data processing and plotting libraries
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
# Import application‐specific GPR and mesh modules
from modelDefinition import *
from stl import mesh
from hyperparametersGPR import features, xList, hTrain, rTrain, devPairs, testPairs, trainPairs

# Path to the precomputed GPR feature database
PFDatabase = './GPRDatabase'

# ===== Reference Case Configuration =====
# Reference case setups (multiple options commented out; select one below)
#### themisABL setup ####
#reference = {'fName':'themisABL'
            #,'h':0.06,'r':87,'alpha':0.4,'k':1.5,'x':0.9,'hMatch':0.714}

#### themisABL_alpha setup ####
#reference = {'fName':'themisABL_alpha'
            #,'h':0.04,'r':92,'alpha':0.20,'k':1.5,'x':0.3,'hMatch':0.714}

##### TPU_highrise_14_middle_dimensional setup ####
#reference = {'fName':'TPU_highrise_14_middle_dimensional'
             #,'h':0.04,'r':92,'alpha':0.7,'k':1.11,'x':2.7,'hMatch':0.714}

##### LRB Cat1 scale 1:50 ####
#reference = {'fName':LRB_Cat1_scale1to50
             #,'h':0.12,'r':79,'alpha':0.18,'k':1.62,'x':11.0,'hMatch':0.4}

##### LRB Cat1 scale 1:25 ####
#reference = {'fName':'LRB_Cat1_scale1to25'
             #,'h':0.04,'r':75,'alpha':0.36,'k':1.36,'x':3.3,'hMatch':0.666}

###### LRB Cat1 scale 1:10 ####
##reference = {'fName':'LRB_Cat1_scale1to10'
             ##,'h':0.16,'r':64,'alpha':0.90,'k':1.25,'x':9.0,'hMatch':0.666}

###### MRB Cat2 scale 1:100 ####
##reference = {'fName':'MRB_Cat2_scale1to100'
             ##,'h':0.04,'r':92,'alpha':0.45,'k':1.39,'x':3.30,'hMatch':0.666}

###### MRB Cat3 scale 1:100 ####
##reference = {'fName':'MRB_Cat3_scale1to100'
             ##,'h':0.08,'r':92,'alpha':0.45,'k':1.56,'x':1.50,'hMatch':0.666}

##### LRB Cat1 ####
#fName = 'LRB_Cat1'
#reference = {'fName':'LRB_Cat1'
             #,'h':0.06,'r':54,'alpha':0.42,'k':1.35,'x':3.3,'hMatch':0.666}

##### LRB Cat2 ####
#fName = 'LRB_Cat2'
#reference = {'fName':'LRB_Cat2'
             #,'h':0.04,'r':89,'alpha':0.23,'k':1.75,'x':0.6,'hMatch':0.666}

##### MRB Cat2 ####
#fName = 'MRB_Cat2'
#reference = {'fName':'MRB_Cat2'
             #,'h':0.04,'r':92,'alpha':0.42,'k':1.47,'x':3.0,'hMatch':0.666}


##### HRB Cat2 ####
#fName = 'HRB_Cat4'
#reference = {'fName':'HRB_Cat4'
             #,'h':0.05,'r':52,'alpha':0.25,'k':1.52,'x':0.6,'hMatch':0.666}

#### Frank Cat_4 ####
# fName = 'BL-1_0_alpha'
# reference = {'fName':'BL-1_0_alpha'
#              ,'h':0.13,'r':91,'alpha':0.61,'k':1.48,'x':4.0,'hMatch':0.666}

##### MRB CatB ####
# fName = 'MRB_Cat_B'
# reference = {'fName':fName
#             ,'h':0.04,'r':92,'alpha':0.42,'k':1.47,'x':3.0,'hMatch':0.666}

##### LRB CatB ####
fName = 'LRB_Cat_B'
reference = {'fName':fName
            ,'h':0.07,'r':74,'alpha':0.34,'k':1.62,'x':0.6,'hMatch':0.666}

# ===== Scaling Factor Computation =====
# Compute geometric scaling factors from HABL and reference α
scale = 1.0/100.0
#scale = 1.0/21.42857142857111
#HABL = 240.0
H_build = 6 #Full Scale Building Height
HABL = H_build * 1.5
Uscaling = 15 # target velocity at building height, default 15 m/s (was used in GPR fitting)

scaling = HABL*scale/reference['alpha']
caseDirectory = './'+reference['fName']+'_geometric_1to'+str(np.round(1.0/scale).astype(int))

generateCase(scaling, reference['h'], reference['x'], caseDirectory, reference['fName'])
# Toggle plotting of ABL profiles
plotABL = True

# Define model identifiers for intensity and inflow stress fields
intensitiesModelID = 'intensities'
inflowModelID = 'inflow_stresses'
uncertainty = False

# Set maximum non‐dimensional y for normalization in GPR
yMax= 1.0

# Prepare DataFrame of features at the reference case for prediction
fit_features = pd.DataFrame()
fit_features['y'] = np.linspace(0.01,1.0,2000)
fit_features['x'] = reference['x']
fit_features['h'] = reference['h']
fit_features['alpha'] = reference['alpha']
fit_features['r'] = reference['r']
fit_features['k'] = reference['k']

# ===== Assemble Training, Development, and Test Datasets =====
# Build parameter pairs and pack into dicts for GPR
# Assemble dictionaries of training, development, and test points
devPoints = {'h':devPairs[:,0],'r':devPairs[:,1],'x':[reference['x']]}
testPoints = {'h':testPairs[:,0],'r':testPairs[:,1],'x':[reference['x']]}
trainPoints = {'h':trainPairs[:,0],'r':trainPairs[:,1],'x':[reference['x']]}

# Initialize the GaussianProcess object with our data sets
gp = gaussianProcess(trainPoints, devPoints, testPoints, yMax, PFDatabase)

# ===== Plot ABL Profiles (if enabled) =====
if plotABL:
    # -- Load & normalize reference ABL data (plot only) --
    prefix   = str(reference['x']).replace('.', 'p') + '_'
    directory = prefix + intensitiesModelID
    ref_abl  = pd.read_csv('TestCases/'+reference['fName']+'.dat', sep=',')
    header   = list(ref_abl.columns)
    idx      = np.argmax(ref_abl['y'].to_numpy())
    yref     = ref_abl['y'].iloc[idx]
    ref_abl['y'] = ref_abl['y'] / yref

    # -- Initial GPR 'u' prediction to compute Uscaling (plot only) --
    model    = '../GPRModels/'+directory+'_u.pkl'
    y_mean   = gp.predict(model, fit_features, features, 'u')
    y_mean   = y_mean.loc[y_mean['y'] <= reference['alpha'] * y_mean['y'].max()]
    y_mean['y'] = y_mean['y'] / y_mean['y'].max()

    U_ABL_dim = interp1d(ref_abl['y'], ref_abl['u'])(reference['hMatch']).item()
    U_TIG_dim = interp1d(y_mean['y'],  y_mean['y_model'])(reference['hMatch']).item()
    # Uscaling  = U_ABL_dim / U_TIG_dim * yref / reference['alpha']

    # -- Print scaling summary for the plot --
    # print('Scaling velocity from GPR to reference ABL:', np.round(Uscaling,3), 'm/s')
    print('Reference velocity at', np.round(reference['hMatch']*yref,3),
          'm:', np.round(U_ABL_dim,3), 'm/s')

    # -- Configure figure and plot profiles --
    my_dpi = 100
    plt.figure(figsize=(2260/my_dpi, 1300/my_dpi), dpi=my_dpi)
    cont = 1
    for QoI in ['u','Iu','Iv','Iw']:
        # Predict each quantity of interest (QoI) with GPR
        model = '../GPRModels/'+directory+'_'+QoI+'.pkl'
        
        y_mean = gp.predict(model,fit_features,features,QoI)
        y_mean = y_mean.loc[y_mean['y']<=reference['alpha']*np.max(y_mean['y'])]
        y_mean['y'] = y_mean['y']/(y_mean['y'].max())
        
        # Apply appropriate scaling to the predicted profiles
        if QoI == 'u':
            y_mean['y_model'] = y_mean['y_model']*reference['k']
            y_mean['y_std'] = y_mean['y_std']*reference['k']
            
        plt.subplot(1,4,cont)
        
        if (QoI in header):
            
            if QoI == 'u':
                plt.plot(ref_abl[QoI]/ref_abl['u'].iloc[idx],ref_abl['y'],color='tab:red',label='Target',linewidth=3)
                plt.fill_betweenx(ref_abl['y'], ref_abl[QoI]/ref_abl['u'].iloc[idx]*0.9, ref_abl[QoI]/ref_abl['u'].iloc[idx]*1.1, color='tab:red', alpha=0.2,label=r'Reference $\pm$10%')
            else:
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
        
        
        #plt.xlim(0,1.1*max_x)
        plt.xlabel(QoI)
        
        plt.ylim(0,1)
        plt.yticks([0.25,0.5,0.75,1.0])
        
        if QoI=='u'or QoI == 'Iv':
            plt.ylabel('y/H')
        else:
            plt.gca().set_yticklabels([])

        cont += 1

    plt.suptitle('Chosen setup, dimension vs adimensional y')

    plt.legend(frameon=False)
    plt.savefig('TestCases/'+reference['fName']+'.png', bbox_inches='tight')

# ===== Generate Geometric Case Files & Inflow Profiles =====
yMax = 1.5 # [m] Height of the GPR downstream fit (not yMax in the paper)
# redefine yMax to extend the vertical normalization range for inflow generation

# override previously computed Uscaling to a fixed inlet‐velocity scale for case export
        
# ===== Inflow Profile Generation & Export =====
plt.figure(figsize=(2260/my_dpi, 1300/my_dpi), dpi=my_dpi)
for x in [-4.95, -2.85]:
    # Reinitialize GPR for a new inflow plane at x
    trainPoints = {'h': trainPairs[:,0], 'r': trainPairs[:,1], 'x': [x]}
    devPoints   = {'h': devPairs[:,0],   'r': devPairs[:,1],   'x': [x]}
    testPoints  = {'h': testPairs[:,0],  'r': testPairs[:,1],  'x': [x]}
    # redefine train/dev/test points dictionaries for the current x-location

    gp = gaussianProcess(trainPoints, devPoints, testPoints, yMax, PFDatabase, np.linspace(0.01,1.0,100))
    # reinitialize the GaussianProcess object for each inflow plane

    outputDF = pd.DataFrame()  
    fit_features = pd.DataFrame()
    # reset DataFrames inside the loop to accumulate new predictions per x

    fit_features['y'] = np.linspace(0.01/yMax, 1.0, 1501)
    fit_features['x'] = x
    fit_features['h'] = reference['h']
    fit_features['r'] = reference['r']
        
    outputDF['x'] = np.ones((len(fit_features['y'].to_numpy()),))*x
    outputDF['y'] = fit_features['y'].to_numpy()*yMax*scaling
    #outputDF['y'] = fit_features['y'].to_numpy()*yMax*reference['alpha']/yref
    outputDF['z'] = np.zeros((len(fit_features['y'].to_numpy()),))
    outputDF['y-velocity'] = np.zeros((len(fit_features['y'].to_numpy()),))
    outputDF['z-velocity'] = np.zeros((len(fit_features['y'].to_numpy()),))
    outputDF['uw-reynolds-stress'] = np.zeros((len(fit_features['y'].to_numpy()),))
    outputDF['vw-reynolds-stress'] = np.zeros((len(fit_features['y'].to_numpy()),))

    cont = 1
    for QoI in ['u','uu','vv','ww','uv']:
        prefix = str(str(x)+'_').replace('.','p')
        directory = prefix + inflowModelID
        model = '../GPRModels/'+directory+'_'+QoI+'.pkl'  
        # redefine 'model' to load the QoI-specific GPR file each iteration

        y_mean = gp.predict(model, fit_features, features, QoI)
        y_mean['y'] = y_mean['y']/(y_mean['y'].max())
        
        if QoI == 'u':
            y_mean['y_model'] = y_mean['y_model']*Uscaling * reference['k']
            y_mean['y_std'] = y_mean['y_std']*Uscaling * reference['k']
        else:
            y_mean['y_model'] = y_mean['y_model']*(Uscaling*reference['k'])**2
            y_mean['y_std'] = y_mean['y_std']*(Uscaling*reference['k'])**2
            
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
    outputDF[['x','y','z','x-velocity','y-velocity','z-velocity','uu-reynolds-stress','vv-reynolds-stress','ww-reynolds-stress','uv-reynolds-stress','uw-reynolds-stress','vw-reynolds-stress']].to_csv(caseDirectory+'/'+reference['fName']+'_'+lab+'.txt',sep='\t',index=False)

plt.suptitle('Chosen setup')
plt.legend(frameon=False)
plt.show()
plt.close('all')