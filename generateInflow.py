import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from modelDefinition import *

###########################################################

PFDatabase = './GPRDatabase'
#fName = 'TPU_highrise_14_middle_dimensional'
fName = 'themisABL'

# stresses_normalized
intensitiesModelID = 'intensities'
inflowModelID = 'inflow_stresses'
uncertainty = False

hMatch_adimensional = 0.714

########################################################

##### TPU_highrise_14_middle_dimensional setup ####
#h = 0.04
#r = 92
#alpha = 0.70
#k = 1.11
#x = 2.7
#hMatch_adimensional = 0.714

##### themisABL setup ####
h = 0.06
r = 87
alpha = 0.4
k = 1.5
x = 0.9
#hMatch_adimensional = 0.5

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

#QoIs = ['u','Iu','Iv','Iw']

yMax= 1.0

#escape = 't'
#while not(escape == 'y'):
    #more = 'a'
    
    #listSolutions = input('Which h, r, alpha, and x values do you wanna use to generate the inflow?\n')
    #try:
        #listSolutions = list(map(float, listSolutions.split()))
    #except:
        #listSolutions = [None]
        
    #if len(listSolutions) == 4:
        #h     = listSolutions[0]
        #r     = listSolutions[1]
        #alpha = listSolutions[2]
        #x     = listSolutions[3]
        
        #if (h<0.04) or (h>0.16) or (r<52) or (r>92) or (alpha<0.1) or alpha>1.0 or not(x in xList):
            #print('Bad choice buddy...')
            #if (h<0.04) or (h>0.16):
                #print('h should be in [0.04,0.16]!')
            #if (r<52) or (r>92):
                #print('r should be in [52,92]!')
            #if (alpha<0.1) or (alpha>1.0):
                #print('alpha should be in ]0.1,1.0],!')
            #if not(x in xList):
                #print('x should be in '+str(xList)+'!')
            #print('Try again\n')
        #else:
            #escape = 'y'

fit_features = pd.DataFrame()
fit_features['y'] = np.linspace(0.01,1.0,2000)
fit_features['x'] = x
fit_features['h'] = h
fit_features['r'] = r
fit_features['k'] = k
fit_features['alpha'] = alpha

trainPoints = {'h':trainPairs[:,0],'r':trainPairs[:,1],'x':[x]}
devPoints = {'h':devPairs[:,0],'r':devPairs[:,1],'x':[x]}
testPoints = {'h':testPairs[:,0],'r':testPairs[:,1],'x':[x]}

gp = gaussianProcess(trainPoints, devPoints, testPoints, yMax, PFDatabase)

prefix = str(str(x)+'_').replace('.','p')
directory = prefix+intensitiesModelID
    
ref_abl = pd.read_csv('TestCases/'+fName+'.dat',sep=',')
header = list(ref_abl.columns)
idx = np.argmax(ref_abl['y'].to_numpy())

yref = ref_abl['y'].iloc[idx]*1.0

ref_abl['y'] = ref_abl['y']/yref
ref_abl['u'] = ref_abl['u']
  
model = '../GPRModels/'+directory+'_u.pkl'
y_mean = gp.predict(model,fit_features,features,'u')
y_mean = y_mean.loc[y_mean['y']<=alpha*np.max(y_mean['y'])]
y_mean['y'] = y_mean['y']/(y_mean['y'].max())
#y_mean['y_model'] = y_mean['y_model']

U_ABL_dim = (interp1d(ref_abl['y'], ref_abl['u'])(hMatch_adimensional)).item()
U_TIG_dim = (interp1d(y_mean['y'], y_mean['y_model'])(hMatch_adimensional)).item()

Uscaling = U_ABL_dim/(U_TIG_dim)

#print(U_ABL_dim)
#print(U_TIG_dim)
#print(fit_features['k'][0])
print('===========================================================================')
print('Scaling velocity from GPR to reference ABL: '+str(np.round(Uscaling,3))+'m/s')
print('Reference velocity at '+str(np.round(hMatch_adimensional*yref,3))+'m reference ABL height: '+str(np.round(U_ABL_dim,3))+'m/s')
print('===========================================================================')


my_dpi = 100
plt.figure(figsize=(2260/my_dpi, 1300/my_dpi), dpi=my_dpi)
cont=1

for QoI in ['u','Iu','Iv','Iw']:
    model = '../GPRModels/'+directory+'_'+QoI+'.pkl'
    
    y_mean = gp.predict(model,fit_features,features,QoI)
    y_mean = y_mean.loc[y_mean['y']<=alpha*np.max(y_mean['y'])]
    y_mean['y'] = y_mean['y']/(y_mean['y'].max())
    
    if QoI == 'u':
        y_mean['y_model'] = y_mean['y_model']*Uscaling
        y_mean['y_std'] = y_mean['y_std']*Uscaling
        
    plt.subplot(2,4,cont)
    
    if (QoI in header):
        plt.plot(ref_abl[QoI],ref_abl['y'],color='tab:red',label='Target',linewidth=3)
        plt.fill_betweenx(ref_abl['y'], ref_abl[QoI]*0.9, ref_abl[QoI]*1.1, color='tab:red', alpha=0.2,label=r'Reference $\pm$10%')
        
    line = plt.plot(y_mean['y_model'],y_mean['y'],linestyle='--',linewidth=3
            ,label=r'x='+'{0:.2f}'.format(x)+'m,h='+'{0:.2f}'.format(h)+'m'+r'm,$\alpha$='+'{0:.2f}'.format(alpha)+r',r='+'{0:.2f}'.format(r))
    
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
            ,label=r'x='+'{0:.2f}'.format(x)+'m,h='+'{0:.2f}'.format(h)+'m'+r'm,$\alpha$='+'{0:.2f}'.format(alpha)+r',r='+'{0:.2f}'.format(r))
    
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
plt.savefig('TestCases/'+fName+'.png', bbox_inches='tight')
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
    fit_features['h'] = h
    fit_features['r'] = r
        
    outputDF['x'] = np.ones((len(fit_features['y'].to_numpy()),))*x
    outputDF['y'] = fit_features['y'].to_numpy()*yMax
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
    outputDF[['x','y','z','x-velocity','y-velocity','z-velocity','uu-reynolds-stress','vv-reynolds-stress','ww-reynolds-stress','uv-reynolds-stress','uw-reynolds-stress','vw-reynolds-stress']].to_csv('TestCases/'+fName+'_'+lab+'.txt',sep='\t',index=False)

plt.suptitle('Chosen setup')
plt.legend(frameon=False)
plt.show()
plt.close('all')
            


##h = 0.06
##r = 87
##alpha = 0.40
##x = 0.9

##yMax = 1.5
        
##my_dpi = 100
##plt.figure(figsize=(2260/my_dpi, 1300/my_dpi), dpi=my_dpi)

##for x in [-4.95,-2.85]:
    ##cont=1
                
    ##trainPoints = {'h':trainPairs[:,0],'r':trainPairs[:,1],'x':[x]}
    ##devPoints = {'h':devPairs[:,0],'r':devPairs[:,1],'x':[x]}
    ##testPoints = {'h':testPairs[:,0],'r':testPairs[:,1],'x':[x]}
        
    ##gp = gaussianProcess(trainPoints, devPoints, testPoints, yMax, PFDatabase, np.linspace(0.01,1.0,100))

    ##outputDF = pd.DataFrame()
    ##fit_features = pd.DataFrame()

    ##trainData = loadData([h], [x], [r], yMax, PFDatabase, np.linspace(0.01,1.0,100))
        
    ##outputDF['x'] = np.ones((len(trainData['y'].to_numpy()),))*x
    ##outputDF['y'] = trainData['y'].to_numpy()*yMax
    ##outputDF['z'] = np.zeros((len(trainData['y'].to_numpy()),))
    ##outputDF['y-velocity'] = np.zeros((len(trainData['y'].to_numpy()),))
    ##outputDF['z-velocity'] = np.zeros((len(trainData['y'].to_numpy()),))
    ##outputDF['uw-reynolds-stress'] = np.zeros((len(trainData['y'].to_numpy()),))
    ##outputDF['vw-reynolds-stress'] = np.zeros((len(trainData['y'].to_numpy()),))

    ##for QoI in ['u','uu','vv','ww','uv']:
        
        ##prefix = str(str(x)+'_').replace('.','p')
        ##directory = prefix+inflowModelID
        ##model = '../GPRModels/'+directory+'_'+QoI+'.pkl'
        
        ##y_mean = gp.predict(model,trainData,features,QoI)
        ##y_mean['y'] = y_mean['y']/(y_mean['y'].max())
        
        ###if QoI == 'u':
            ###y_mean['y_model'] = y_mean['y_model']/(y_mean['y_model'].iloc[-1])*Uscaling*15.17/17.10
            ###y_mean['y_std'] = y_mean['y_std']/(y_mean['y_model'].iloc[-1])*Uscaling*15.17/17.10
        ###else:
            ###y_mean['y_model'] = y_mean['y_model']*((Uscaling*15.17/17.10)**2)
            ###y_mean['y_std'] = y_mean['y_std']*((Uscaling*15.17/17.10)**2)
        
        ##if QoI == 'u':
            ##y_mean['y_model'] = y_mean['y_model']/(y_mean['y_model'].iloc[-1])*Uscaling
            ##y_mean['y_std'] = y_mean['y_std']/(y_mean['y_model'].iloc[-1])*Uscaling
        ##else:
            ##y_mean['y_model'] = y_mean['y_model']*Uscaling*Uscaling
            ##y_mean['y_std'] = y_mean['y_std']*Uscaling*Uscaling
            
        ##plt.subplot(2,3,cont)
        
        ##if x == -4.95:
            ##lab = 'inflow_input'
        ##elif x == -2.85:
            ##lab = 'ALF_input'
            
        ##if QoI == 'u':
            ##outputDF['x-velocity'] = y_mean['y_model'].to_numpy()
            ##line = plt.plot(outputDF['x-velocity'],outputDF['y'],linewidth=2,label=lab)
        ##elif QoI == 'uu':
            ##outputDF['uu-reynolds-stress'] = np.abs(y_mean['y_model'].to_numpy())
            ##line = plt.plot(outputDF['uu-reynolds-stress'],outputDF['y'],linewidth=2,label=lab)
        ##elif QoI == 'vv':
            ##outputDF['vv-reynolds-stress'] = np.abs(y_mean['y_model'].to_numpy())
            ##line = plt.plot(outputDF['vv-reynolds-stress'],outputDF['y'],linewidth=2,label=lab)
        ##elif QoI == 'ww':
            ##outputDF['ww-reynolds-stress'] = np.abs(y_mean['y_model'].to_numpy())
            ##line = plt.plot(outputDF['ww-reynolds-stress'],outputDF['y'],linewidth=2,label=lab)
        ##elif QoI == 'uv':
            ##outputDF['uv-reynolds-stress'] = -np.abs(y_mean['y_model'].to_numpy())
            ##line = plt.plot(outputDF['uv-reynolds-stress'],outputDF['y'],linewidth=2,label=lab)
        
        ##max_x = np.ceil(1.2*np.max(y_mean['y_model'])*10000).astype(int)/10000
        
        ##plt.xlim(0,1.1*max_x)
        
        ##if QoI=='u':
            ##plt.ylabel('y[m]')
            ##plt.xlabel('u[m/s]')
        ##else:
            ##plt.ylabel('y[m]')
            ##plt.xlabel(QoI+r'$[m^2/s^2]$')
        
        ###if uncertainty == True:
            ###plt.fill_betweenx(y_mean['y'], y_mean['y_model']-2*y_mean['y_std'], y_mean['y_model']+2*y_mean['y_std'], color=line[0].get_color(), alpha=0.2)
        
        ##cont +=1

    ##outputDF[['x','y','z','x-velocity','y-velocity','z-velocity','uu-reynolds-stress','vv-reynolds-stress','ww-reynolds-stress','uv-reynolds-stress','uw-reynolds-stress','vw-reynolds-stress']].to_csv('TestCases/'+fName+'_'+lab+'.txt',sep='\t',index=False)

##plt.suptitle('Chosen setup')
##plt.legend(frameon=False)
##plt.show()
##plt.close('all')
            
            
            
            
            
            
            
            
            