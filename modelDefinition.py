import random
import joblib
import numpy as np
import pandas as pd
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

def loadData(heights, location, roughness, yMax, plot=False, home='./GPRDatabase', yInterp=False):
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
            
            U_yMax_interp = (interp1d(interp_df['y'], interp_df['x-velocity'])(yMax)).item()
            #print(database_df['y'], database_df['x-velocity'])
            #U_yMax_interp=1.0
            
            temp['u'] = temp['u']/U_yMax_interp
            temp['uu'] = temp['uu']/(U_yMax_interp**2)
            temp['uv'] = temp['uv']/(U_yMax_interp**2)
            temp['vv'] = temp['vv']/(U_yMax_interp**2)
            temp['ww'] = temp['ww']/(U_yMax_interp**2)
            
            lastRow={'x':filtered_df['x'].iloc[0], 'y':yMax/yMax, 'u':U_yMax_interp/U_yMax_interp, 'h':filtered_df['h'].iloc[0],'r':filtered_df['r'].iloc[0]
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


class gaussianProcess:
    
    normalization = 'normal'
    
    model_loaded = False
    
    def __init__(self, trainPoints, devPoints, testPoints, yMax, homeDirectory, yInterp=False):
        
        self.yMax = yMax
        
        self.trainData = loadData(trainPoints['h'], trainPoints['x'], trainPoints['r'], self.yMax , False, homeDirectory, yInterp)
        self.devData  = loadData(devPoints['h'], devPoints['x'], devPoints['r'], self.yMax , False, homeDirectory, yInterp)
        self.testData = loadData(testPoints['h'], testPoints['x'], testPoints['r'], self.yMax , False, homeDirectory, yInterp)
        
        #print(self.devData)
        #print(self.testData)
        #input()
        
        self.meanVal = self.trainData.mean()
        self.stdVal  = self.trainData.std()
        
        
        return
    
    def loadModel(self,model):
        
        self.model_loaded = True
        self.predictive_model = joblib.load(model)
        print('Model loaded preemptively')
    
    def predict(self, model, cDF, features):
        
        contourData = cDF.copy(deep=True)
        
        if self.model_loaded == False:
            self.predictive_model = joblib.load(model)
            #print('Model loaded at prediction time')
        
        if self.normalization == 'log':
            warnings.filterwarnings("ignore")
            contourData['y_model'], contourData['y_std'] = self.predictive_model.predict(np.log(contourData[features]), return_std=True)
        elif self.normalization == 'normal':
            warnings.filterwarnings("ignore")
            contourData['y_model'], contourData['y_std'] = self.predictive_model.predict((contourData[features]-self.meanVal[features])/self.stdVal[features], return_std=True)
                
        return contourData
        

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
        #gpr = GaussianProcessRegressor(kernel=eval(expression),alpha = alpha, random_state=1)
        
        if self.normalization == 'log':
            gpr.fit(np.log(self.trainData[features].to_numpy()), self.trainData[[QoI]].to_numpy())
            y_dev, _  = gpr.predict(np.log(self.devData[features]), return_std=True)
            y_test, _ = gpr.predict(np.log(self.testData[features]), return_std=True)
        
        elif self.normalization == 'normal':
            gpr.fit(((self.trainData[features]-self.meanVal[features])/self.stdVal[features]).to_numpy(), self.trainData[[QoI]].to_numpy())
            y_dev, _  = gpr.predict(((self.devData[features]-self.meanVal[features])/self.stdVal[features]).to_numpy(), return_std=True)
            y_test, _ = gpr.predict(((self.testData[features]-self.meanVal[features])/self.stdVal[features]).to_numpy(), return_std=True)
            
        #print(y_dev)
        #print(self.devData[QoI].to_numpy())
        ##input()
        
        #print(y_dev)
        #print(self.devData[QoI].to_numpy())
        #plt.figure()
        #plt.plot(y_dev)
        #plt.plot(self.devData[QoI].to_numpy())
        #plt.show()
        
        dev_RMSE_relative  = np.linalg.norm((y_dev-self.devData[QoI].to_numpy())/self.devData[QoI].to_numpy())
        dev_RMSE  = np.linalg.norm(y_dev-self.devData[QoI].to_numpy())
        
        test_RMSE_relative = np.linalg.norm((y_test-self.testData[QoI].to_numpy())/self.testData[QoI].to_numpy())
        test_RMSE = np.linalg.norm(y_test-self.testData[QoI].to_numpy())
        
        #print(QoI + ' Relative RMSE: '+ str(dev_RMSE_relative))
        #print(QoI + ' Absolute RMSE:' + str(dev_RMSE))
        #input()
        
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

    def __init__(self, varDict, hTr, hD, hT, yMax, xList, refAbl, features, QoIs, home, testName):
    
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
        
        super().__init__(n_var=4,
                         n_obj=len(QoIs),
                         xl=[varDict[r'$h$'][0], varDict[r'$r$'][0],varDict[r'$\alpha$'][0], varDict[r'$x$'][0]],
                         xu=[varDict[r'$h$'][1], varDict[r'$r$'][1],varDict[r'$\alpha$'][1], varDict[r'$x$'][1]])
        
        self.var_names = QoIs
    
    def eval_model_delayed(self, h, r, alpha, x):
        
        xDiscrete = self.xList[int(np.round(x))]
        rDiscrete = np.round(r)
        hDiscrete = np.round(h*100)/100
        alphaDiscrete = np.round(alpha,2)
        
        fit_features = pd.DataFrame()
        fit_features['y'] = np.concatenate((np.linspace(0.01,0.3,1000),np.linspace(0.3,1.0,1000)),axis=0)
        fit_features['x'] = xDiscrete
        fit_features['r'] = rDiscrete
        fit_features['h'] = hDiscrete
        
        trainPoints = {'h':self.trainPoints[:,0],'r':self.trainPoints[:,1],'x':[xDiscrete]}
        devPoints = {'h':self.devPoints[:,0],'r':self.devPoints[:,1],'x':[xDiscrete]}
        testPoints = {'h':self.testPoints[:,0],'r':self.testPoints[:,1],'x':[xDiscrete]}
        gp_model = gaussianProcess(trainPoints, devPoints, testPoints, self.yMax, self.datasetLocation)

        prefix = str(str(xDiscrete)+'_').replace('.','p')
        
        scores = [None]*len(self.targetVars)
        
        cont = 0
            
        for QoI in self.targetVars:
        
            model = '../GPRModels/'+prefix+self.testPrefix+'_'+QoI+'.pkl'
            y_mean = gp_model.predict(model,fit_features,self.features)
            
            y_query, model_predictions = scale_predictions(y_mean, self.targetABL, alphaDiscrete, QoI)
            
            target = self.targetABL.loc[(self.targetABL['y']>=np.min(y_query)) & (self.targetABL['y']<=np.max(y_query)),QoI].to_numpy()
            
            scores[cont] = np.linalg.norm(model_predictions-target)
            cont+=1
            
        return scores

    def _evaluate(self, params, out, *args, **kwargs):
        
        nIndividuals = params.shape[0]
        
        temp = joblib.Parallel(n_jobs=12)(joblib.delayed(self.eval_model_delayed)(params[i,0],params[i,1],params[i,2],params[i,3]) for i in range(nIndividuals))
        
        self.nGenerations += 1
        
        print(self.nGenerations)
            
        out["F"] = np.array(temp)    
    
try:
    os.mkdir('../GPRModels')
except:
    print('GPRModels directory already exist!')

try:
    os.mkdir('../RegressionPlots')
except:
    print('RegressionPlots directory already exist!')
