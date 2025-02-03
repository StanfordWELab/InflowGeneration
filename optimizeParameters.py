import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy          import optimize
from scipy.optimize import minimize
from scipy.optimize import Bounds
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, WhiteKernel
from scipy.interpolate import interp1d
from modelDefinition import *

font={'size'   : 20}
matplotlib.rc('font', **font)
    
def scale_predictions(model_profile, target_profile, alpha, QoI):
        
    ##Predicted output is scaled
    model_profile = model_profile.loc[model_profile['y']<=alpha*np.max(model_profile['y'])]
    
    ##Predicted scaled output vertical scale is normalized between something and 1
    model_profile['y'] = model_profile['y']/model_profile['y'].max()
    
    ##Find the largest minimum and smallest maximum between model output and target data
    ##This way, the smallest and largest y values are excluded
    minVal = max([np.min(model_profile['y']),np.min(target_profile['y'])])
    maxVal = min([np.max(model_profile['y']),np.max(target_profile['y'])])
    
    nModel  = len(model_profile.loc[(model_profile['y']>=minVal) & (model_profile['y']<=maxVal)])
    nTarget = len(target_profile.loc[(target_profile['y']>=minVal) & (target_profile['y']<=maxVal)])
    
    if nModel>=nTarget:
        yQuery = target_profile.loc[(target_profile['y']>=minVal) & (target_profile['y']<=maxVal),'y'].to_numpy()
        
        yData   = model_profile.loc[(model_profile['y']>=minVal) & (model_profile['y']<=maxVal),'y'].to_numpy()
        QoIData = model_profile.loc[(model_profile['y']>=minVal) & (model_profile['y']<=maxVal),'y_model'].to_numpy()
        
    else:
        raise Exception('You need to increase the # of points at which the model is evaluated')
    
    QoIQuery = interp1d(model_profile['y'].to_numpy(),model_profile['y_model'].to_numpy())(yQuery)
                
    if QoI == 'u':
        QoIQuery = QoIQuery/interp1d(yData,QoIData)(1.0).item()
    
    return  yQuery, QoIQuery
    
def evaluate_setup(hTr, hD, hT, yMax, parameters, refAbl, features, QoIs, home, testName):
    
    h     = parameters[0]
    x     = parameters[1]
    r     = parameters[2]
    alpha = parameters[3]
    
    print(h,r,x,alpha)
    
    fit_features = pd.DataFrame()
    
    trainPoints = {'h':hTr[:,0],'r':hTr[:,1],'x':[x]}
    devPoints = {'h':hD[:,0],'r':hD[:,1],'x':[x]}
    testPoints = {'h':hT[:,0],'r':hT[:,1],'x':[x]}
    gp = gaussianProcess(trainPoints, devPoints, testPoints, yMax, home)
    
    prefix = str(str(x)+'_').replace('.','p')
    directory = prefix+testName
    
        
    fit_features['x'] = x
    #fit_features['y'] = np.linspace(0.01,1.0,1000)
    fit_features['y'] = np.concatenate((np.linspace(0.01,0.3,1000),np.linspace(0.3,1.0,1000)),axis=0)
    fit_features['h'] = h
    fit_features['r'] = r
    dictionary = {}
    
    dictionary['h'] = h
    dictionary['x'] = x
    dictionary['r'] = r
    dictionary['alpha'] = alpha
    
    for QoI in QoIs:
        
        model = '../GPRModels/'+directory+'_'+QoI+'.pkl'
        y_mean = gp.predict(model,fit_features,features)
        
        y_query, model_predictions = scale_predictions(y_mean, refAbl, alpha, QoI)
        
        target = refAbl.loc[(refAbl['y']>=np.min(y_query)) & (refAbl['y']<=np.max(y_query)),QoI].to_numpy()
        
        #RMSE_relative  = np.linalg.norm((y_mean['y_model']-refAbl[QoI]).to_numpy()/refAbl[QoI].to_numpy())
        #RMSE_relative  = np.linalg.norm((model_predictions-target)/target)
        dictionary['RMSE relative '+QoI] = np.linalg.norm((model_predictions-target)/target)
        dictionary['RMSE '+QoI] = np.linalg.norm(model_predictions-target)

    return dictionary
    
###############################################################################################

PFDatabase = '../../TIGTestMatrixLong/'

#fNames = ['testABL']
#fNames = ['TPU_ABL']
#fNames = ['themisABL']
#fNames = ['inflowProfile_U10_Cat4_1uu1vv1ww']
#fNames = ['inflowProfile_U10_Cat1_1uu1vv1ww','inflowProfile_U10_Cat2_1uu1vv1ww','inflowProfile_U10_Cat3_1uu1vv1ww','inflowProfile_U10_Cat4_1uu1vv1ww']
#fNames = ['inflowProfile_Cat2_1uu1vv1ww','inflowProfile_Cat3_1uu1vv1ww']
fNames = ['TPU_highrise_14_middle']
fNames = ['TPU_highrise_14']

testID = 'intensities'

#hList = [0.08,0.10,0.12,0.14,0.16]
#xList = [0.3,0.6,1.2,1.8,2.7,3.6,5.0,7.0,9.0,11.0,13.0]
#alphaList = [0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#rList = [52,62,72,82,92]

hList = [0.08,0.10,0.14]
xList = [0.6,1.5,3.0,11.0]
#xList = [0.3,1.8,3.3,7.0]
alphaList = [1.0]
rList = [52,62,72,82,92]

hSearch = [0.14]
xSearch = [1.5]
alphaSearch = alphaList
#alphaSearch = [0.7]
rSearch = rList

nCpu = 12

nResults = 3

#targetQoIs = ['u','uu','vv','ww']
#weights={'u':0.25,'uu':0.25,'ww':0.25,'vv':0.25}
targetQoIs = ['u','Iu']
weights={'u':0.25,'Iu':1.0}

everyQoI = ['u','Iu','Iv','Iw']
#everyQoI = ['u','uu','vv','ww']

#### metric is 'RMSE' or 'RMSE relative'
metric = 'RMSE'
uncertainty = True

###############################################################################################

mode = sys.argv[1]

yMax= 1.0
features = ['y','h','r']

hTrain = [0.08,0.12,0.16]
rTrain = [52,62,72,82,92]
trainPairs = np.zeros((len(hTrain)*len(rTrain),2))
cont=0
for h in hTrain:
    for r in rTrain:
        trainPairs[cont,:] = [h, r]
        cont+=1

devPairs = np.array([[0.14,67],[0.14,77]])
testPairs = np.array([[0.14,57],[0.14,87]])

#xABL = 0.9
#hABL = 0.08
#rABL = 47
##ref_abl = loadData([hABL], [xABL], [rABL], yMax, plot=False, home='../../PFTestMatrix')



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
            
    #ref_abl['uu'] = ref_abl['uu']/(Uref**2)
    #ref_abl['vv'] = ref_abl['vv']/(Uref**2)
    #ref_abl['ww'] = ref_abl['ww']/(Uref**2)
    #ref_abl['uv'] = ref_abl['uv']/(Uref**2)
    #ref_abl['uw'] = ref_abl['uw']/(Uref**2)
    #ref_abl['vw'] = ref_abl['vw']/(Uref**2)
    if 'u min' in header:
        ref_abl['u min'] = ref_abl['u min']/Uref
    if 'u max' in header:
        ref_abl['u max'] = ref_abl['u max']/Uref
    QoIs = []

    for QoI in everyQoI:
        if (QoI in header) and (QoI in targetQoIs):
            QoIs.append(QoI)

    #opt = optimizer(ref_abl, yMax, hTrain, features)
    #hABL, xABL = opt.findParameters()

    if mode == 'Gridsearch':

        nTests = len(hList)*len(xList)*len(rList)*len(alphaList)
        exploration_matrix = np.zeros((nTests,4))
        print(nTests)

        cont = 0
        for h in hList:
            for x in xList:
                for r in rList:
                    for alpha in alphaList:
                        exploration_matrix[cont,:] = [h, x, r, alpha]
                        cont+=1
                        
        temp = joblib.Parallel(n_jobs=nCpu)(joblib.delayed(evaluate_setup)(trainPairs,devPairs,testPairs,
            yMax,exploration_matrix[i,:],ref_abl,features,QoIs,PFDatabase,testID) for i in range(nTests))

        df = pd.DataFrame()
        for dct in temp:
            if df.empty:
                df = pd.DataFrame([dct])
            else:
                df = pd.concat([df, pd.DataFrame([dct])], ignore_index=True)

        df.to_csv('TestCases/'+fName+'_optimal.csv',index=False,index_label=False)

    elif mode == 'Plot':

        df = pd.read_csv('TestCases/'+fName+'_optimal.csv')
        df['cost'] = 0
        for QoI in weights:
            df['cost'] += weights[QoI]*df[metric+' '+QoI]
            
        optimum_setup = df.sort_values(by = 'cost', ascending = True)
        optimum_setup = (optimum_setup[optimum_setup['h'].isin(hSearch) & optimum_setup['x'].isin(xSearch) & optimum_setup['r'].isin(rSearch) & optimum_setup['alpha'].isin(alphaSearch)]).sort_values(by = 'cost', ascending = True)
        
            
        print(optimum_setup.sort_values(by = 'cost', ascending = True).iloc[:10])
        
        my_dpi = 100
        plt.figure(figsize=(2260/my_dpi, 1300/my_dpi), dpi=my_dpi)

        cont=1

        for QoI in weights:
            
            for i in range(nResults):

                xTemp = optimum_setup['x'].iloc[i]
                hTemp = optimum_setup['h'].iloc[i]
                rTemp = optimum_setup['r'].iloc[i]
                alphaTemp = optimum_setup['alpha'].iloc[i]

                fit_features = pd.DataFrame()
                fit_features['y'] = np.linspace(0.01,1.0,2000)
                fit_features['x'] = xTemp
                fit_features['h'] = hTemp
                fit_features['r'] = rTemp
                fit_features['alpha'] = alphaTemp
            
                trainPoints = {'h':trainPairs[:,0],'r':trainPairs[:,1],'x':[xTemp]}
                devPoints = {'h':devPairs[:,0],'r':devPairs[:,1],'x':[xTemp]}
                testPoints = {'h':testPairs[:,0],'r':testPairs[:,1],'x':[xTemp]}

                gp = gaussianProcess(trainPoints, devPoints, testPoints, yMax, PFDatabase)
                
                prefix = str(str(xTemp)+'_').replace('.','p')
                directory = prefix+testID
                model = '../GPRModels/'+directory+'_'+QoI+'.pkl'
                
                y_mean = gp.predict(model,fit_features,features)
                y_mean = y_mean.loc[y_mean['y']<=alphaTemp*np.max(y_mean['y'])]
                y_mean['y'] = y_mean['y']/(y_mean['y'].max())
                
                if QoI == 'u':
                    y_mean['y_model'] = y_mean['y_model']/(y_mean['y_model'].iloc[-1])
                    y_mean['y_std'] = y_mean['y_std']/(y_mean['y_model'].iloc[-1])
                if len(weights) == 1:
                    plt.subplot(1,1,cont)
                elif len(weights) == 2:
                    plt.subplot(1,2,cont)
                else:
                    plt.subplot(2,2,cont)
                    
                if i==0:
                    plt.plot(ref_abl[QoI],ref_abl['y'],color='tab:red',label='Target',linewidth=3)
                    #if (str(QoI+' min') in header) and (str(QoI+' max') in header):
                        #plt.fill_betweenx(ref_abl['y'], ref_abl[QoI+' min'], ref_abl[QoI+' max'], color='tab:red', alpha=0.3,label=r'Reference and range')
                    #else:
                    plt.fill_betweenx(ref_abl['y'], ref_abl[QoI]*0.9, ref_abl[QoI]*1.1, color='tab:red', alpha=0.2,label=r'Reference $\pm$10%')
                    
                line = plt.plot(y_mean['y_model'],y_mean['y'],linestyle='--',linewidth=3
                        ,label=str(i+1)+r': x='+'{0:.2f}'.format(xTemp)+'m,h='+'{0:.2f}'.format(hTemp)+'m'+r'm,$\alpha$='+'{0:.2f}'.format(alphaTemp)+r',r='+'{0:.2f}'.format(rTemp))
                
                max_x = np.ceil((1.2*max([np.max(ref_abl[QoI]),np.max(y_mean['y_model'])])*10000).astype(int))/10000
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
                #plt.title(QoI +', Target vs model')
                
            cont +=1

        plt.legend(frameon=False)
            
        print('Best '+str(nResults)+' results:')
        print(optimum_setup.iloc[0:nResults])

        #plt.savefig('../RegressionPlots/'+fName+'_optimal.png', bbox_inches='tight')
        plt.show()
        plt.close('all')


