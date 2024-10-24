import numpy as np
import pandas as pd
import re
import os
from sklearn.linear_model import LinearRegression

import matplotlib
#matplotlib.use( 'Agg' )
import matplotlib.pyplot as plt

#################################################################

testID = 'hf'
xList = [0.3,0.6]

#################################################################

cont = 1
quantities = ['u','Iu','Iv','Iw']

#if plot==True:
    #my_dpi = 100
    #plt.figure(figsize=(2260/my_dpi, 1300/my_dpi), dpi=my_dpi)
    
for x in xList:
    
    for QoI in quantities:
        
        prefix = str(str(x)+'_').replace('.','p')
        fName = prefix+testID+'_'+QoI 

        seed          = []
        alpha         = []
        RMSE_relative = []
        RMSE          = []
        kernel        = []
        nKernels      = []
        trainPoints   = []
        devPoints     = []
        
        fileName = str('../GPRModels/' + fName + '.dat')

        ###################################
        ### Read results into dataframe ###
        ###################################
        
        with open(fileName,'r') as infile:
            for line in infile:
                if '=' in line:
                    seedFound = False
                if 'Seed' in line:
                    seed.append([int(x) for x in re.findall('[0-9]+', line)][0])
                    seedFound = True
                if 'Alpha' in line:
                    alpha.append([float(x) for x in re.findall('[0-9].+', line)][0])
                if 'Kernels' in line:
                    idx = line.index(':')
                    kernel.append(re.findall(r'[\+\s](.*?)(?=\()', line[line.index(':')+1:]))
                    nKernels.append(line.count(')'))
                if 'Dev  RMSE Relative' in line:
                    RMSE_relative.append([float(x) for x in re.findall('[0-9].+', line)][0])
                if 'Dev  RMSE' in line:
                    RMSE.append([float(x) for x in re.findall('[0-9].+', line)][0])
                if 'Train points' in line:
                    idx = line.index(':')
                    trainPoints.append(line[idx+2:-1])
                if 'Dev points' in line:
                    idx = line.index(':')
                    devPoints.append(line[idx+2:-1])
                    
                df = pd.DataFrame(list(zip(seed, alpha, nKernels, kernel, RMSE_relative, RMSE, trainPoints, devPoints)), 
                                columns = ['Seed', 'Alpha', '# of Kernels', 'Kernels', 'Dev RMSE Relative', 'Dev RMSE', 'Train points','Dev points'])
                
        df = df.sort_values(by = 'Dev RMSE Relative', ascending = True)

        print("=================================================================================================================")
        print('Considering file ' + fName)
        temp = df.head(n=5)
        print(temp.to_string(index=False))
        print("\n===============================================================================================================")
        print('WHICH SEED WOULD YOU LIKE TO SAVE?')
        
        seed = df['Seed'].iloc[0]
        os.system('cp ../GPRModels/' + fName + '/'+str(seed)+'.pkl ' +' ../GPRModels/'+fName+'.pkl')
        
        #if plot == True:
            #plt.subplot(3,4,cont)
            #plt.scatter(df['# of Kernels']+np.random.uniform(-0.3,0.3,len(df['# of Kernels'])),np.log10(df['Alpha']),c = df['Dev RMSE Relative'],vmin=0,vmax=3)
            #plt.colorbar()
            #plt.title(QoI +' at x='+str(x))
            #cont+=1
#plt.show()
















