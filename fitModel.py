from datetime import datetime
import joblib
import sys
from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, WhiteKernel
from scipy.interpolate import interp1d
from modelDefinition import *

colors  = {'0.03':['tab:cyan',(0,(3,5,3,5))]
          ,'0.04':['tab:cyan',(0,(3,1,1,1,1,1))]
          ,'0.05':['tab:cyan',(0,(5,1))]
          ,'0.06':['tab:cyan',(0,())]
          ,'0.07':['tab:blue',(0,(3,5,3,5))]
          ,'0.08':['tab:blue',(0,(3,1,1,1,1,1))]
          ,'0.09':['tab:blue',(0,(5,1))]
          ,'0.10':['tab:blue',(0,())]
          ,'0.11':['tab:orange',(0,(3,5,3,5))]
          ,'0.12':['tab:orange',(0,(3,1,1,1,1,1))]
          ,'0.13':['tab:orange',(0,(5,1))]
          ,'0.14':['tab:orange',(0,())]
          ,'0.15':['tab:red',(0,(3,5,3,5))]
          ,'0.16':['tab:red',(0,(3,1,1,1,1,1))]
          ,'0.17':['tab:red',(0,(5,1))]
          ,'0.18':['tab:red',(0,())]}

###########################################################

hTrain = [0.04,0.08,0.12,0.16]
rTrain = [32,42,52,62]
#hTrain = [0.04,0.08]
#rTrain = [32,42]

nCpu = 12

devPairs = np.array([[0.06,37],[0.10,47]])
testPairs = np.array([[0.06,47],[0.10,57]])

PFDatabase = '../../PFTestMatrix'

testID = 'hf'

###########################################################

xModels = eval(sys.argv[1])
mode    = sys.argv[2]

trainPairs = np.zeros((len(hTrain)*len(rTrain),2))
cont=0
for h in hTrain:
    for r in rTrain:
        trainPairs[cont,:] = [h, r]
        cont+=1

features = ['y','h','r']

yMax= 1.0

if mode == 'Gridsearch':

    for x in xModels:
        
        trainPoints = {'h':trainPairs[:,0],'r':trainPairs[:,1],'x':[x]}
        devPoints = {'h':devPairs[:,0],'r':devPairs[:,1],'x':[x]}
        testPoints = {'h':testPairs[:,0],'r':testPairs[:,1],'x':[x]}
        
        prefix = str(str(x)+'_').replace('.','p')
        directory = prefix+testID

        for QoI in ['u','Iu','Iv','Iw']:

            gp = gaussianProcess(trainPoints, devPoints, testPoints, yMax, PFDatabase)
            
            try:
                os.mkdir('../GPRModels/'+directory+'_'+QoI)
            except:
                print('Directory already exist')
            
                    
            with open('../GPRModels/'+directory+'_'+QoI+'.dat', 'w+') as out:
                now = datetime.now()
                out.write('\n'*10+'Gridsearch performed on ' + str(now.strftime("%d %m %Y, %H:%M:%S"))+ '\n'*10)
                
            _ = joblib.Parallel(n_jobs=nCpu)(joblib.delayed(gp.gridsearch)(seed, features, directory, QoI)
                                    for seed in range(0,2))

#############################################################


elif mode == 'Plot':

    setToPlot = 'Test'

    for x in xModels:

        my_dpi = 100
        plt.figure(figsize=(2260/my_dpi, 1300/my_dpi), dpi=my_dpi)
        cont=1
                
        trainPoints = {'h':trainPairs[:,0],'r':trainPairs[:,1],'x':[x]}
        devPoints = {'h':devPairs[:,0],'r':devPairs[:,1],'x':[x]}
        testPoints = {'h':testPairs[:,0],'r':testPairs[:,1],'x':[x]}
        
        gp = gaussianProcess(trainPoints, devPoints, testPoints, yMax, PFDatabase)

        for QoI in ['u','Iu','Iv','Iw']:
            
            #refTrainData = trainPairs
            #refDevData = devPairs
        
            for i in range(trainPairs.shape[0]):
                
                h=trainPairs[i,0]
                r=trainPairs[i,1]
                
                trainData = loadData([h], [x], [r], yMax, plot=False)
                
                prefix = str(str(x)+'_').replace('.','p')
                directory = prefix+'test'+'_'+QoI
                
                model = '../GPRModels/'+prefix+'test_'+QoI+'.pkl'
                
                y_mean_train = gp.predict(model,trainData,features)
                
                plt.subplot(2,4,cont)
                plt.plot(trainData[QoI],trainData['y'],color='tab:grey')
                plt.plot(y_mean_train['y_model'],trainData['y'],label = 'x='+str(x)+'m, h='+str(h)+'m, r='+str(r))
                plt.ylabel('y/H')
                plt.xlabel(QoI +' Train')
                
            if setToPlot == 'Dev':
                withheld = devPairs
            elif setToPlot=='Test':
                withheld = testPairs
                        
            for i in range(withheld.shape[0]):
                
                h=withheld[i,0]
                r=withheld[i,1]
                
                withheldData = loadData([h], [x], [r], yMax, plot=False)
                
                prefix = str(str(x)+'_').replace('.','p')
                directory = prefix+'test'+'_'+QoI
                
                model = '../GPRModels/'+prefix+'test_'+QoI+'.pkl'
                
                y_mean_dev = gp.predict(model,withheldData,features)
                
                plt.subplot(2,4,cont+4)
                plt.plot(withheldData[QoI],withheldData['y'],color='tab:grey')
                if i == 0:
                    plt.fill_betweenx(withheldData['y'], withheldData[QoI]*0.9, withheldData[QoI]*1.1, color='tab:grey', alpha=0.3,label=r'Reference $\pm$10%')
                else:
                    plt.fill_betweenx(withheldData['y'], withheldData[QoI]*0.9, withheldData[QoI]*1.1, color='tab:grey', alpha=0.3)
                plt.plot(y_mean_dev['y_model'],withheldData['y'],label = 'x='+str(x)+'m, h='+str(h)+'m, r='+str(r))
                plt.ylabel('y/H')
                plt.xlabel(QoI +' '+setToPlot)
                
            cont +=1

        plt.legend()
        plt.savefig('../RegressionPlots/'+prefix+setToPlot+'.png', bbox_inches='tight')
        plt.show()
        plt.close('all')