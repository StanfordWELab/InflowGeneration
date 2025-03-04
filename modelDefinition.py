import random
import joblib
import numpy as np
import pandas as pd
import copy as cp
import warnings

from matplotlib import pyplot as plt
from datetime import datetime
from scipy.interpolate import interp1d

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