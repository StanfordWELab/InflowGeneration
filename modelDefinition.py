import os
import random
import joblib
import warnings
import numpy  as np
import pandas as pd
import copy   as cp

from matplotlib import pyplot as plt
from datetime import datetime
from scipy.interpolate import interp1d
from stl               import mesh

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, WhiteKernel

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.result import Result
from pymoo.core.problem import Problem

def loadData(heights, location, roughness, yMax, home='./GPRDatabase', reference_velocity = False, yInterp=False):
    ymin = 0.005
    u=15
    data = pd.DataFrame()
    
    for i in range(len(heights)):
            
        h=heights[i]
        r=roughness[i]
        
        database_df = (pd.read_csv(home+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(r)+'.txt',sep='\t'))
            
        for x in location:
            
            prefix = str(str(x)+'_').replace('.','p')
            
            temp         = pd.DataFrame()
            interpolated = pd.DataFrame()
            
            interp_df   = database_df[(abs(database_df['x'] - x) < 1e-6)]
            filtered_df = database_df[(abs(database_df['x'] - x) < 1e-6) & (database_df['y'] >= ymin) & (database_df['y'] <= yMax)]
            
            #print(filtered_df)
            #input()
            
            temp['x'] = filtered_df['x'].round(1).astype(float)
            temp['y'] = filtered_df['y']/yMax
            temp['u'] = filtered_df['x-velocity'].copy(deep=True)
            
            temp['uu'] = filtered_df['uu-reynolds-stress'].copy(deep=True)
            temp['vv'] = filtered_df['vv-reynolds-stress'].copy(deep=True)
            temp['ww'] = filtered_df['ww-reynolds-stress'].copy(deep=True)
            temp['uv'] = abs(filtered_df['uv-reynolds-stress'].copy(deep=True))
            
            temp['Iu']  = np.sqrt(temp['uu'])/filtered_df['x-velocity-magnitude']
            temp['Iv']  = np.sqrt(temp['vv'])/filtered_df['x-velocity-magnitude']
            temp['Iw']  = np.sqrt(temp['ww'])/filtered_df['x-velocity-magnitude']
            temp['Iuv'] = np.sqrt(abs(temp['uv']))/filtered_df['x-velocity-magnitude']
            
            temp['h'] = filtered_df['h'].round(2).astype(float)
            temp['r'] = filtered_df['r'].astype(int)
            
            if reference_velocity == False:
                U_yMax_interp = (interp1d(interp_df['y'], interp_df['x-velocity'])(yMax)).item()
            else:
                U_yMax_interp = reference_velocity*1.0
            
            temp['u'] = temp['u']/U_yMax_interp
            temp['uu'] = temp['uu']/(U_yMax_interp**2)
            temp['uv'] = temp['uv']/(U_yMax_interp**2)
            temp['vv'] = temp['vv']/(U_yMax_interp**2)
            temp['ww'] = temp['ww']/(U_yMax_interp**2)
            
            lastRow={'x':filtered_df['x'].iloc[0], 'y':yMax/yMax, 'u':(interp1d(interp_df['y'], interp_df['x-velocity'])(yMax)).item()/U_yMax_interp, 'h':filtered_df['h'].iloc[0],'r':filtered_df['r'].iloc[0]
                   , 'Iu':(interp1d(interp_df['y'], np.sqrt(interp_df['uu-reynolds-stress'])/interp_df['x-velocity-magnitude']))(yMax).item()
                   , 'Iv':(interp1d(interp_df['y'], np.sqrt(interp_df['vv-reynolds-stress'])/interp_df['x-velocity-magnitude']))(yMax).item()
                   , 'Iw':(interp1d(interp_df['y'], np.sqrt(interp_df['ww-reynolds-stress'])/interp_df['x-velocity-magnitude']))(yMax).item()
                   ,'Iuv':(interp1d(interp_df['y'], np.sqrt(abs(interp_df['uv-reynolds-stress']))/interp_df['x-velocity-magnitude']))(yMax).item()
                   , 'uu':((interp1d(interp_df['y'], interp_df['uu-reynolds-stress']))(yMax).item())/(U_yMax_interp**2)
                   , 'vv':((interp1d(interp_df['y'], interp_df['vv-reynolds-stress']))(yMax).item())/(U_yMax_interp**2)
                   , 'ww':((interp1d(interp_df['y'], interp_df['ww-reynolds-stress']))(yMax).item())/(U_yMax_interp**2)
                   , 'uv':abs((interp1d(interp_df['y'], interp_df['uv-reynolds-stress']))(yMax).item())/(U_yMax_interp**2)}
            
            
            temp = pd.concat([pd.DataFrame([lastRow]),temp], ignore_index=True)
            #print(temp)
            
            if isinstance(yInterp, (np.ndarray, np.generic)):
            
                cols = temp.columns.tolist()
                
                cols.remove('x')
                interpolated['x'] = x*np.ones(yInterp.shape)
                
                cols.remove('y')
                interpolated['y'] = 1.0*yInterp
                
                cols.remove('h')
                interpolated['h'] = h
                
                cols.remove('r')
                interpolated['r'] = r
                
                for col in cols:
                    
                    f = interp1d(temp['y'], temp[col],kind='cubic')

                    interpolated[col] = 1.0*f(yInterp)
                
                if data.empty:
                    data = interpolated
                else:
                    data = data.merge(interpolated, how='outer')
                
            else:
                
                if data.empty:
                    data = temp
                else:
                    data = data.merge(temp, how='outer')

    return data

def scale_predictions(model_profile, target_profile, alpha, QoI):
        
    ##Predicted output is scaled
    model_profile = model_profile.loc[model_profile['y']<=alpha*np.max(model_profile['y'])]
    
    ##Predicted scaled output vertical scale is normalized between something and 1
    model_profile['y'] = model_profile['y']/model_profile['y'].max()
    
    ##Find the largest minimum and smallest maximum between model output and target data
    ##This way, the smallest and largest y values are excluded
    minVal = max([np.min(model_profile['y']),np.min(target_profile['y'])])
    maxVal = min([np.max(model_profile['y']),np.max(target_profile['y'])])
    
    #nModel  = len(model_profile.loc[(model_profile['y']>=minVal) & (model_profile['y']<=maxVal)])
    #nTarget = len(target_profile.loc[(target_profile['y']>=minVal) & (target_profile['y']<=maxVal)])
    
    #if nModel>=nTarget:
    yQuery = target_profile.loc[(target_profile['y']>=minVal) & (target_profile['y']<=maxVal),'y'].to_numpy()
    
    #yData   = model_profile.loc[(model_profile['y']>=minVal) & (model_profile['y']<=maxVal),'y'].to_numpy()
    QoIData = model_profile.loc[(model_profile['y']>=minVal) & (model_profile['y']<=maxVal),'y_model'].to_numpy()
        
    #else:
        #raise Exception('You need to increase the # of points at which the model is evaluated')
    
    QoIQuery = interp1d(model_profile['y'].to_numpy(),model_profile['y_model'].to_numpy())(yQuery)
                
    #if QoI == 'u':
        #QoIQuery = QoIQuery/interp1d(yData,QoIData)(1.0).item()
    
    return  yQuery, QoIQuery
        
def parallelCoordinatesPlot(pyMooResults, xValues, decisionVars, QoIs, indices, case):

    xPlot = np.linspace(0,1,len(QoIs))

    y0Plot,y1Plot = [(pyMooResults.F.min(axis=0)).min()*0.95,(pyMooResults.F.max(axis=0)).max()*1.05]

        
    if len(pyMooResults.X) == 2:
        cont0 = 211
    else:
        cont0 = 231
        
    paramNames =[r'$h$',r'$r$',r'$\alpha$',r'$k$',r'$x$']

    my_dpi = 100
    plt.figure(figsize=(2200/my_dpi, 1200/my_dpi), dpi=my_dpi)
            
    cont = 0
    for param in (pyMooResults.X).T:
        
        
        vmin,vmax=[param.min(), param.max()]
        
        if paramNames[cont] == r'$h$':
            vmin, vmax = [-0.005, 0.195]
            cm = plt.cm.tab20
        if paramNames[cont] == r'$r$':
            vmin, vmax = [np.round(np.min(param))-5, np.round(np.max(param))+5]
            cm = plt.cm.hsv
        if paramNames[cont] == r'$\alpha$':
            vmin, vmax = [0.05, 1.05]
            cm = plt.cm.tab20
        if paramNames[cont] == r'$k$':
            vmin, vmax = [np.round(vmin)-0.1, np.round(vmax)+0.1]
            cm = plt.cm.hsv
        if paramNames[cont] == r'$x$':
            param = [xValues.index(p) for p in param]
            vmin, vmax = [-0.5,19.5]
            cm = plt.cm.tab20
        
        norm = plt.Normalize(vmin, vmax)
        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        
        plt.subplot(cont+cont0)
        
        idx=0
        for yVal in pyMooResults.F:
            
            plt.plot(xPlot,yVal,color=cm(norm(param[idx])))
            
            if idx%5 == 0:
                plt.text(-0.01, yVal[0], str(indices[idx]), fontsize=10, ha='right', va='center')
            
            idx+=1
        
        plt.ylabel('RMSE')
            
        cbar=plt.colorbar(sm)
        #cbar.set_label(decisionVars[cont])
        
        for x in xPlot:
            plt.plot([x]*10,np.linspace(y0Plot,y1Plot,10),color='black',alpha=0.5,linewidth=2)
            plt.xticks(xPlot,labels=QoIs)
        
        if paramNames[cont] == r'$h$':
            cbar.set_ticks(np.linspace(0.04,0.18,15))
        if paramNames[cont] == r'$\alpha$':
            cbar.set_ticks(np.linspace(0.1,1.0,10))
        if paramNames[cont] == r'$x$':
            cbar.set_ticks(np.linspace(0,len(xValues)-1,len(xValues)))
            cbar.set_ticklabels(xValues)
            
        plt.title(paramNames[cont])
            
        cont+=1
        
    plt.axis('tight')
    plt.savefig('TestCases/'+case+'_PCP.png', bbox_inches='tight')
    plt.show()
    
def plotSetup(trainPairs, devPairs, testPairs, yMax, features, testID, PFDatabase, parameters, ref_abl, QoIs, uncertainty,case):
    
    header = list(ref_abl.columns)
    
    my_dpi = 100
    plt.figure(figsize=(2260/my_dpi, 1300/my_dpi), dpi=my_dpi)

    cont=1

    for QoI in ['u','Iu','Iv','Iw']:
        
        for i in range(len(parameters[r'$h$'])):

            hTemp = parameters[r'$h$'][i]
            rTemp = parameters[r'$r$'][i]
            alphaTemp = parameters[r'$\alpha$'][i]
            kTemp = parameters[r'$k$'][i]
            xTemp = parameters[r'$x$'][i]

            fit_features = pd.DataFrame()
            fit_features['y'] = np.linspace(0.01,1.0,2000)
            fit_features['h'] = hTemp
            fit_features['r'] = rTemp
            fit_features['alpha'] = alphaTemp
            fit_features['k'] = kTemp
            fit_features['x'] = xTemp
        
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
                
            plt.subplot(2,2,cont)
            
            if i==0 and (QoI in header):
                plt.plot(ref_abl[QoI],ref_abl['y'],color='tab:red',label='Target',linewidth=3)
                plt.fill_betweenx(ref_abl['y'], ref_abl[QoI]*0.9, ref_abl[QoI]*1.1, color='tab:red', alpha=0.2,label=r'Reference $\pm$10%')
                
            line = plt.plot(y_mean['y_model'],y_mean['y'],linestyle='--',linewidth=3
                    ,label=str(parameters['idx'][i])+' h='+'{0:.2f}'.format(hTemp)+'m, r='+'{0:.0f}'.format(rTemp)+',\n'+r'$\alpha$='+'{0:.2f}'.format(alphaTemp)+r', $k$='+'{0:.2f}'.format(kTemp)+', x='+'{0:.2f}'.format(xTemp)+'m')
            
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

    #plt.savefig('../RegressionPlots/'+fName+'_best.png', bbox_inches='tight')
    plt.savefig('TestCases/'+case+'_solutions.png', bbox_inches='tight')
    plt.show()
    plt.close('all')
    
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
        y_mean = gp.predict(model,fit_features,features,QoI)
        
        y_query, model_predictions = scale_predictions(y_mean, refAbl, alpha, QoI)
        
        target = refAbl.loc[(refAbl['y']>=np.min(y_query)) & (refAbl['y']<=np.max(y_query)),QoI].to_numpy()
        
        #RMSE_relative  = np.linalg.norm((y_mean['y_model']-refAbl[QoI]).to_numpy()/refAbl[QoI].to_numpy())
        #RMSE_relative  = np.linalg.norm((model_predictions-target)/target)
        dictionary['RMSE relative '+QoI] = np.linalg.norm((model_predictions-target)/target)
        dictionary['RMSE '+QoI] = np.linalg.norm(model_predictions-target)

    return dictionary


class gaussianProcess:
    
    normalization = 'normal'
    
    model_loaded = False
    
    def __init__(self, trainPoints, devPoints, testPoints, yMax, homeDirectory, yInterp=False):
        
        self.yMax = yMax
        
        self.trainData = loadData(trainPoints['h'], trainPoints['x'], trainPoints['r'], self.yMax, homeDirectory, 15.0, yInterp)
        self.devData  = loadData(devPoints['h'], devPoints['x'], devPoints['r'], self.yMax, homeDirectory, 15.0, yInterp)
        self.testData = loadData(testPoints['h'], testPoints['x'], testPoints['r'], self.yMax, homeDirectory, 15.0, yInterp)
        
        self.meanVal = self.trainData.mean()
        self.stdVal  = self.trainData.std()
        
        return
        
    
    def loadModel(self,model):
        
        self.predictive_model = joblib.load(model)
        self.model_loaded = True
        #print('Preloading model '+model)
    
    def predict(self, model, cDF, features,QoI):
        
        if self.model_loaded == False:
            self.predictive_model = joblib.load(model)
            #print('Loading model from scratch')
            
        if self.normalization == 'log':
            warnings.filterwarnings("ignore")
            cDF['y_model'], cDF['y_std'] = self.predictive_model.predict(np.log(cDF[features]), return_std=True)
        elif self.normalization == 'normal':
            warnings.filterwarnings("ignore")
            cDF['y_model'], cDF['y_std'] = self.predictive_model.predict((cDF[features]-self.meanVal[features])/self.stdVal[features], return_std=True)
        
        return cDF
        

    def gridsearch(self, seed, features, directory, QoI):
        
        kernels = ['RBF()', 'Matern()', 'RationalQuadratic()', 'DotProduct()', 'WhiteKernel()']
        
        np.random.seed(seed)
        random.seed(seed)
        
        alpha  = 10**np.random.uniform(-7.0,-2.0)
        
        nKernels = np.random.randint(1,5)
        
        expression=''
        for i in range(nKernels):
            k = random.choice(kernels)
            expression = expression+k+'+'
            kernels.remove(k)
        expression=expression[:-1]
        
        print('Running seed '+str(seed)+' with ' +expression)
        
        from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, WhiteKernel
        
        gpr = GaussianProcessRegressor(kernel=eval(expression),alpha = alpha, random_state=seed, normalize_y=True)
        
        if self.normalization == 'log':
            gpr.fit(np.log(self.trainData[features].to_numpy()), self.trainData[[QoI]].to_numpy())
            y_dev, _  = gpr.predict(np.log(self.devData[features]), return_std=True)
            y_test, _ = gpr.predict(np.log(self.testData[features]), return_std=True)
        
        elif self.normalization == 'normal':
            gpr.fit(((self.trainData[features]-self.meanVal[features])/self.stdVal[features]).to_numpy(), self.trainData[[QoI]].to_numpy())
            y_dev, _  = gpr.predict(((self.devData[features]-self.meanVal[features])/self.stdVal[features]).to_numpy(), return_std=True)
            y_test, _ = gpr.predict(((self.testData[features]-self.meanVal[features])/self.stdVal[features]).to_numpy(), return_std=True)
        
        dev_RMSE_relative  = np.linalg.norm((y_dev-self.devData[QoI].to_numpy())/self.devData[QoI].to_numpy())
        dev_RMSE  = np.linalg.norm(y_dev-self.devData[QoI].to_numpy())
        
        test_RMSE_relative = np.linalg.norm((y_test-self.testData[QoI].to_numpy())/self.testData[QoI].to_numpy())
        test_RMSE = np.linalg.norm(y_test-self.testData[QoI].to_numpy())
        
        with open('../GPRModels/'+directory+'_'+QoI+'.dat','a+') as out:
            out.write('\n==============================================')
            out.write('\nSeed               : ' +str(seed))
            out.write('\nAlpha              : ' +str(alpha))
            out.write('\n# of kernels       : ' +str(nKernels))
            out.write('\nKernels            : ' +expression)
            out.write('\nDev  RMSE Relative : ' +str(dev_RMSE_relative))
            out.write('\nTest RMSE Relative : ' +str(test_RMSE_relative))
            out.write('\nDev  RMSE          : ' +str(dev_RMSE))
            out.write('\nTest RMSE          : ' +str(test_RMSE))
            out.write('\nNormalization      : ' +str(self.normalization))
            out.write('\nTrain points       : ' +'x='+str(np.unique(self.trainData['x']))+'; h='+str(np.unique(self.trainData['h']))+'; r='+str(np.unique(self.trainData['r'])))
            #out.write('\nTest points        : ' +'x='+str(np.unique(self.devData['x']))+'; h='+str(np.unique(self.devData['h'])))
            out.write('\nDev points         : ' +'x='+str(np.unique(self.testData['x']))+'; h='+str(np.unique(self.testData['h']))+'; r='+str(np.unique(self.devData['r'])))
            out.write('\n==============================================')
            
            
        joblib.dump(gpr, '../GPRModels/'+directory+'_'+QoI+'/'+str(seed)+'.pkl')
        
        return

class MyProblem(Problem):

    def __init__(self, varDict, hTr, hD, hT, yMax, xList, refAbl, features, QoIs, home, testName,nCpu):
    
        self.trainPoints = hTr
        self.devPoints = hD
        self.testPoints = hT
        self.yMax = yMax
        self.xList = xList
        
        self.targetABL  = refAbl
        
        self.targetVars = QoIs
        self.features = features
        self.datasetLocation = home
        self.testPrefix = testName
        
        self.nGenerations = 0
        
        self.nCpu = nCpu
        
        self.modelDict = {}
        
        print(self.nCpu)
        print('Preloading models...')
        
        for x in self.xList:
            prefix = str(str(x)+'_').replace('.','p')
            self.modelDict[prefix]= {}
            for QoI in QoIs:
                self.modelDict[prefix][QoI]= {}
        
        for x in self.xList:
            trainPoints = {'h':self.trainPoints[:,0],'r':self.trainPoints[:,1],'x':[x]}
            devPoints = {'h':self.devPoints[:,0],'r':self.devPoints[:,1],'x':[x]}
            testPoints = {'h':self.testPoints[:,0],'r':self.testPoints[:,1],'x':[x]}
            gp_model = gaussianProcess(trainPoints, devPoints, testPoints, self.yMax, self.datasetLocation)

            prefix = str(str(x)+'_').replace('.','p')
            
            for QoI in QoIs:
        
                model = '../GPRModels/'+prefix+self.testPrefix+'_'+QoI+'.pkl'
                gp_model.loadModel(model)
                
                self.modelDict[prefix][QoI] = cp.deepcopy(gp_model)
        
        print('Models loaded!')
        
        super().__init__(n_var=5,
                         n_obj=len(QoIs),
                         xl=[varDict[r'$h$'][0], varDict[r'$r$'][0],varDict[r'$\alpha$'][0], varDict[r'$k$'][0], varDict[r'$x$'][0]],
                         xu=[varDict[r'$h$'][1], varDict[r'$r$'][1],varDict[r'$\alpha$'][1], varDict[r'$k$'][1], varDict[r'$x$'][1]])
        
        self.var_names = QoIs
    
    def eval_model_delayed(self, params):
        
        hDiscrete = np.round(params[0]*100)/100
        rDiscrete = np.round(params[1])
        #alphaDiscrete = np.round(params[2],2)
        alphaDiscrete = params[2]
        #kDiscrete = np.round(params[3],2)
        kDiscrete = params[3]
        xDiscrete = self.xList[int(np.round(params[4]))]
        #print(hDiscrete,rDiscrete,alphaDiscrete,kDiscrete,xDiscrete)
        
        fit_features = pd.DataFrame()
        fit_features['y'] = np.concatenate((np.linspace(0.01,0.3,200),np.linspace(0.3,1.0,200)),axis=0)
        fit_features['x'] = xDiscrete
        fit_features['r'] = rDiscrete
        fit_features['k'] = kDiscrete
        fit_features['h'] = hDiscrete

        prefix = str(str(xDiscrete)+'_').replace('.','p')
        
        scores = [None]*len(self.targetVars)
        
        cont = 0
            
        for QoI in self.targetVars:
            
            y_mean = self.modelDict[prefix][QoI].predict(None,fit_features,self.features,QoI)
            
            y_query, model_predictions = scale_predictions(y_mean, self.targetABL, alphaDiscrete, QoI)
            
            target = self.targetABL.loc[(self.targetABL['y']>=np.min(y_query)) & (self.targetABL['y']<=np.max(y_query)),QoI].to_numpy()
            
            if QoI == 'u':
                model_predictions = model_predictions*kDiscrete
            
                #if self.nGenerations >= 3:
                    #plt.plot(model_predictions,y_query)
                    #plt.plot(target,y_query)
                    #print(kDiscrete,alphaDiscrete)
                    #plt.show()
                
            scores[cont] = np.linalg.norm(model_predictions-target)
            cont+=1
            
        return scores

    def _evaluate(self, params, out, *args, **kwargs):
        
        nIndividuals = params.shape[0]
        
        temp = joblib.Parallel(n_jobs=self.nCpu)(joblib.delayed(self.eval_model_delayed)(params[i,:]) for i in range(nIndividuals))
        
        self.nGenerations += 1
            
        out["F"] = np.array(temp)

class generateCase:
    
    lSponge = 0.8
    lBox    = 2.1
    lFetch  = 2.7
    spacing = 0.3
    hDomain = 3.0
    wDomain = 3.0
    
    NSmooth = 100
    
    upstreamLength = lBox + 0.5*spacing+lFetch
    
    def __init__(self, scaling, hRough, xABL, directory, fName):
        
        self.scaling   = scaling
        self.hRough    = hRough
        self.directory = directory
        self.xABL = xABL
        
        try:
            os.mkdir(self.directory)
            os.mkdir(self.directory+'/Domain')
        except:
            print('Directory already exists')
        
        self.generateVolumes()
        self.generateGeometry()
        self.writeSurfer()
        self.writeStitch()
        self.writeCharlesInput(fName, prerun = False)
        self.writeCharlesInput(fName, prerun = True)
        os.system('cd '+str(directory)+r' && /home/mattiafc/cascade-inflow/bin/surfer.exe -i surferDomain.in')

    def writeRect(self, rmin, rMAX, name):

        xmin = rmin[0]
        ymin = rmin[1]
        zmin = rmin[2]

        xMAX = rMAX[0]
        yMAX = rMAX[1]
        zMAX = rMAX[2]
    
        if (xmin > xMAX)or(ymin > yMAX)or(zmin > zMAX):
            raise ValueError('Sugheiscion definiscion')
    
        if xmin == xMAX:

            vertices = np.array([\
                [xmin, ymin, zmin],
                [xmin, ymin, zMAX],
                [xMAX, yMAX, zMAX],
                [xMAX, yMAX, zmin]])

        elif ymin == yMAX:

            vertices = np.array([\
                [xmin, ymin, zmin],
                [xmin, ymin, zMAX],
                [xMAX, yMAX, zMAX],
                [xMAX, yMAX, zmin]])

        elif zmin == zMAX:

            vertices = np.array([\
                [xmin, ymin, zmin],
                [xMAX, ymin, zMAX],
                [xMAX, yMAX, zMAX],
                [xmin, yMAX, zmin]])

        else:
            print('=================================== WARNING ' +name+' ==================================')
            print('Sughellino, occhio che nel file ' +name+' ce una patch orientata in maniera non standard')

            vertices = np.array([\
                [xmin, ymin, zmin],
                [xmin, ymin, zMAX],
                [xMAX, yMAX, zMAX],
                [xMAX, yMAX, zmin]])


        # Define the 12 triangles composing the cube
        faces = np.array([\
            [0,1,2],
            [0,2,3]])

        # Create the mesh
        stlFace = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                stlFace.vectors[i][j] = vertices[f[j],:]
                
        stlFace.save(name)
        
        return

    def writeTriangle(self, vert1, vert2, vert3, name):

        vertices = np.array([vert1, vert2, vert3])


        # Define the 12 triangles composing the cube
        faces = np.array([[0,1,2]])

        # Create the mesh
        stlFace = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                stlFace.vectors[i][j] = vertices[f[j],:]
                
        stlFace.save(name)
        
        return 0

    def generateVolumes(self):

        x0 = (self.lBox+0.5*self.spacing)*self.scaling
        x1 = x0+self.lFetch*self.scaling
        z0 = -0.5*5*self.wDomain*self.scaling
        z1 =  0.5*5*self.wDomain*self.scaling

        Cd = 100.0
        spacing = [self.spacing*self.scaling, self.spacing*self.scaling]
        L = [0.051*self.scaling, self.hRough*self.scaling, 0.102*self.scaling]

        roughElement = np.array([-L[0]*0.5,L[0]*0.5,0,L[1],-L[2]*0.5,L[2]*0.5,Cd])

        xElem = np.linspace(x0,x1, int(np.round((x1-x0)/spacing[0])+1))

        zElemOdd = np.linspace(z0, z1, int(np.round((z1-z0)/spacing[1])+1))
        zElemEven = np.linspace(z0+spacing[1]/2, z1-spacing[1]/2, int(np.round((z1-z0)/spacing[1])))

        print('=================================================')
        print('Elements at x = '+str(xElem))
        print('Streamwise roughness rows: '+str(len(xElem)))
        print('=================================================')
        print('Odd rows elements at z =  '+str(zElemOdd))
        print('Spanwise roughness elements, odd rows:  '+str(len(zElemOdd)))
        print('Even rows elements at z = '+str(zElemEven))
        print('Spanwise roughness elements, even rows: '+str(len(zElemEven)))
        print('=================================================')

        nRows = len(xElem)
        nEven = len(zElemEven)
        nOdd  = len(zElemOdd)

        if nRows%2 == 0:
            nElem = int(nEven*(nRows/2) +  nOdd*(nRows/2))
        else:    
            nElem = nEven*np.floor(nRows/2).astype(int) +  nOdd*np.ceil(nRows/2).astype(int)

        roughIBMPoints = np.zeros((nElem,7))


        print(roughElement.shape)
        print(roughIBMPoints.shape)
        print(nRows)

        cont = 0
        for i in range(len(xElem)):
            x = xElem[i]
            if (i+1)%2 == 1:
                
                for z in zElemOdd:
                    roughIBMPoints[cont,:] = roughElement + np.array([x, x, 0, 0, z, z, 0])
                    cont += 1
                    
            elif (i+1)%2 == 0:
                
                for z in zElemEven:
                    roughIBMPoints[cont,:] = roughElement + np.array([x, x, 0, 0, z, z, 0])
                    cont += 1
            
                
        with open(self.directory + '/box_list.dat','w+') as out:
            out.write(str(nElem) +' volumes')
            for i in range(nElem):
                out.write('\n')
                out.write(str(np.round(roughIBMPoints[i,0],5))+' '+str(np.round(roughIBMPoints[i,1],5))+' '+str(np.round(roughIBMPoints[i,2],5)))
                out.write(' '+str(np.round(roughIBMPoints[i,3],5))+' '+str(np.round(roughIBMPoints[i,4],5))+' '+str(np.round(roughIBMPoints[i,5],5)))
                out.write(' '+str(np.round(roughIBMPoints[i,6],5)))
                
        return
            
    def generateGeometry(self):
        
        x0D = 0
        x1D = (self.upstreamLength+self.xABL+7.0)*self.scaling
        y0D = 0.0
        y1DInlet = self.hDomain*self.scaling
        y1DOutlet = y1DInlet+x1D*np.tan(0.488304*np.pi/180.0)
        z0D = -0.5*self.wDomain*self.scaling
        z1D =  0.5*self.wDomain*self.scaling

        print('==== Domain starts from ' +str(x0D)+' ====')
        
        self.writeRect([x0D, y0D, z0D], [x1D, y1DInlet, z0D], self.directory+'/Domain/left')
        self.writeTriangle([x0D, y1DInlet, z0D], [x1D, y1DInlet, z0D], [x1D, y1DOutlet, z0D], self.directory+'/Domain/leftTop')
        self.writeRect([x0D, y0D, z1D], [x1D, y1DInlet, z1D], self.directory+'/Domain/right')
        self.writeTriangle([x0D, y1DInlet, z1D], [x1D, y1DInlet, z1D], [x1D, y1DOutlet, z1D], self.directory+'/Domain/rightTop')
        self.writeRect([x0D, y1DInlet, z0D], [x1D, y1DOutlet, z1D], self.directory+'/Domain/top')
        self.writeRect([x0D, y0D, z0D], [x1D, y0D, z1D], self.directory+'/Domain/ground')
        self.writeRect([x1D, y0D, z0D], [x1D, y1DOutlet, z1D], self.directory+'/Domain/outlet')
        self.writeRect([x0D, y0D, z0D], [x0D, y1DInlet, z1D], self.directory+'/Domain/inlet')
            
        return
                
    def writeSurfer(self):
        
        with open(self.directory + '/surferDomain.in','w+') as out:
            out.write('SURF STL_GROUP ./Domain/ground ./Domain/top ./Domain/left ./Domain/right ./Domain/outlet ./Domain/inlet ./Domain/leftTop ./Domain/rightTop\n\n')
            out.write('ZIP_OPEN_EDGES\n\n')
            out.write('FLIP ZONE_NAMES ground,outlet,left,leftTop\n\n')
            out.write('MOVE_TO_ZONE NAME left ZONE_NAMES leftTop\n')
            out.write('MOVE_TO_ZONE NAME right ZONE_NAMES rightTop\n\n')
            out.write('SET_PERIODIC ZONES left right CART 0 0 '+str(3.0*self.scaling)+'\n\n')
            out.write('WRITE_SBIN emptyDomain.sbin')
            
        return
                
    def writeStitch(self):
        
        delta   = np.round(0.09*self.scaling,8)
        xEndRef = np.round((self.upstreamLength+self.xABL+1.0)*self.scaling*10000)/10000
        zEndRef = np.round(0.7*self.wDomain*self.scaling*10000,5)/10000
        yLevel2 = np.round(0.6*self.scaling,8)
        yLevel3 = np.round(0.2*self.scaling,8)
        
        with open(self.directory + '/stitchDomain.in','w+') as out:
            out.write('PART SURF SBIN emptyDomain.sbin\n\n')
            out.write(f'HCP_DELTA {delta:.10f} \n\n')
            out.write(f'HCP_WINDOW BOX 0 {xEndRef:.10f} 0 {yLevel2:.10f} {-zEndRef:.10f} {zEndRef:.10f} LEVEL=2   NLAYERS=10\n')
            out.write(f'HCP_WINDOW BOX 0 {xEndRef:.10f} 0 {yLevel3:.10f} {-zEndRef:.10f} {zEndRef:.10f} LEVEL=3   NLAYERS=10\n\n')
            #out.write('COUNT_POINTS\nINTERACTIVE\n\n')
            out.write('SMOOTH_MODE ALL\n')
            out.write('NSMOOTH 100\n\n')
            out.write('WRITE_RESTART emptyDomain.mles')
            
        return
                
    def writeCharlesInput(self,fName,prerun=True):
        
        if prerun == True:
            dt = np.round(0.0001*self.scaling*10000,5)/10000
            outputFile = self.directory + '/start.in'
        else:
            dt = np.round(0.0004*self.scaling*10000,5)/10000
            outputFile = self.directory + '/charles.in'
        #print(dt)
        #input()
        reset      = np.round(10*self.scaling,5)
        xEndSponge = np.round(self.lSponge*self.scaling*1000,5)/1000
        xEndTIG    = np.round(self.lBox*self.scaling*1000,5)/1000
        yEndTIG    = np.round(1.2*self.hDomain*self.scaling*1000,5)/1000
        zEndTIG    = np.round(0.7*self.wDomain*self.scaling*1000,5)/1000
        L          = np.round(1.0*self.scaling*1000,5)/1000
        relaxT     = np.round(0.0005*self.scaling*10000,5)/10000
        
        # Write_image variables
        target_x = (self.upstreamLength+self.xABL)*self.scaling
        target_y = (self.hDomain*0.5333333)*self.scaling
        target_z = 0.0
        
        camera_x = 2.0*target_x
        camera_y = 1.0*target_y
        camera_z = 1.0*target_z
        
        width    = 2.5*self.wDomain*self.scaling
        plane_x  = (self.upstreamLength+self.xABL)*self.scaling
        prefix   = f"{int(self.xABL)}p{str(self.xABL).split('.')[1]}_"
        
        camera_string = str(f' INTERVAL = 100000 TARGET {target_x:.10f} {target_y:.10f} {target_z:.10f} CAMERA {camera_x:.10f} {camera_y:.10f} {camera_z:.10f} UP 0 1 0 SIZE 1536 718 WIDTH {width:.10f} GEOM PLANE {plane_x:.10f} 0 0 1 0 0 VAR ')
        
        with open(outputFile,'w+') as out:
            
            out.write('# ============================\n\n')
            if prerun == True:
                out.write('RESTART = ./emptyDomain.mles\n\n')
                out.write('INIT u=1.0 0.0 0.0\nINIT p=10.0\nINIT time=0\nINIT step=0\n\n')
            else:
                out.write('RESTART = ./emptyDomain.mles ./data/result.00005000.sles\n\n')
                out.write('INIT time=0\nINIT step=0\n\n')
            
            out.write('# Equation of state\n')
            out.write('EOS HELMHOLTZ\nRHO = 1.225\nMU = 1.7894e-5\nHELMHOLTZ_SOS 340.65\n\n')
            
            out.write('# Time + output setup\n')
            if prerun == True:
                out.write(f'NSTEPS = 5000\nTIMESTEP DT = {dt:.10f}\nCHECK_INTERVAL 10\nWRITE_RESULT NAME=data/result INTERVAL=1000\n\n')
            else:
                out.write(f'NSTEPS = 100000\nTIMESTEP DT = {dt:.10f}\nCHECK_INTERVAL 1000\nWRITE_RESULT NAME=data/result INTERVAL=10000\n\n')
                
            out.write(f'RESET_STATS TIME={reset:.10f}\nSTATS u p mag(u)\n\nSGS_MODEL VREMAN\n\nA_SGS_SPONGE COEFF 100.0 GEOM PLANE {xEndSponge:.10f} 0.0 0.0 1.0 0.0 0.0\n\n')
            
            out.write('# Boundary conditions\n')
            out.write('OUTLET = OUTLET 1.0 0.1 0.0 0.0 LOCAL_U\nGROUND = WM_ALG\nTOP    = WM_ALG\nINLET  = INLET_PROFILE FILE ./'+fName+'_inflow_input.txt FORMAT ASCII\n')
            out.write(f'TURB_VOL_FORCING RELAX_T {relaxT:.10f} DATA_ALF PROFILE ./'+fName+f'_ALF_input.txt GEOM BOX 0 {xEndTIG:.10f} 0 {yEndTIG:.10f} {-zEndTIG:.10f} {zEndTIG:.10f} ESTIM_MEAN_V L {L:.10f}  LIMIT_L_WALL_DIST\n\n')
            
            out.write('#################################\n')
            out.write('######### SOLVER SETUP ##########\n')
            out.write('#################################\n\n')
            out.write('# advanced multigrid solver options\n')
            out.write('MOMENTUM_SOLVER PATR\nMOMENTUM_RELAX 1.0\nMOMENTUM_MAXITER 1000\nMOMENTUM_ZERO 1e-6\n\nFORCING_TERM ON\n\n')
            out.write('# Pressure equation\n')
            out.write('PRESSURE_SOLVER MG\nPRESSURE_AGGLOMERATION_FACTOR 64\nPRESSURE_SPLIT_ORPHANED_COLORS\nPRESSURE_NCG 2\nPRESSURE_SMOOTHER CG\nPRESSURE_NSMOOTH 10\nPRESSURE_RELAX 1.0\nPRESSURE_MAX_ITER 1000\nPRESSURE_ZERO 1e-6\n\n')
            if prerun == True:
                out.write('INTERACTIVE')
            else:
                out.write('#################################\n')
                out.write('###########  PROBING  ###########\n')
                out.write('#################################\n')
                
                out.write('\nWRITE_IMAGE NAME=image/'+prefix+'avg_u'+camera_string+'comp(avg(u),0)')
                out.write('\nWRITE_IMAGE NAME=image/'+prefix+'mag_u'+camera_string+'avg(mag(u))')
                out.write('\n\nWRITE_IMAGE NAME=image/'+prefix+'rms_u'+camera_string+'comp(rms(u),0)')
                out.write('\nWRITE_IMAGE NAME=image/'+prefix+'rms_v'+camera_string+'comp(rms(u),1)')
                out.write('\nWRITE_IMAGE NAME=image/'+prefix+'rms_w'+camera_string+'comp(rms(u),2)')
                out.write('\n\nWRITE_IMAGE NAME=image/'+prefix+'uv   '+camera_string+'comp(rey(u),2)')
                out.write('\nWRITE_IMAGE NAME=image/'+prefix+'rms_p'+camera_string+'rms(p)')
                
                out.write('\n\n#################################')
            
            
        return
    
try:
    os.mkdir('../GPRModels')
except:
    pass
    #print('GPRModels directory already exist!')

try:
    os.mkdir('../RegressionPlots')
except:
    pass
    #print('RegressionPlots directory already exist!')