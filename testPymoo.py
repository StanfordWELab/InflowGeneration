from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem
from pymoo.util.plotting import plot
import numpy as np

import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy          import optimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, WhiteKernel
from scipy.interpolate import interp1d
from modelDefinition import *
    
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

class MyProblem(Problem):

    def __init__(self, hTr, hD, hT, yMax, refAbl, QoIs, home, testName):
    
        self.trainPoints = hTr
        self.devPoints = hD
        self.testPoints = hT
        self.yMax = yMax
        
        self.targetABL  = refAbl
        
        self.targetVars = QoIs
        self.datasetLocation = home
        self.testPrefix = testName
        
        self.nGenerations = 0
        
        super().__init__(n_var=3,
                         n_obj=len(QoIs),
                         xl=[0.08, 52, 0.2],
                         xu=[0.16, 92, 1.0])
    
    def eval_model_delayed(self, h, x, r, alpha, gp_model):
    
        fit_features = pd.DataFrame()
        fit_features['y'] = np.concatenate((np.linspace(0.01,0.3,1000),np.linspace(0.3,1.0,1000)),axis=0)
        fit_features['x'] = x
        fit_features['h'] = h
        fit_features['r'] = r

        prefix = str(str(x)+'_').replace('.','p')
        
        scores = [None]*len(self.targetVars)
        
        cont = 0
            
        for QoI in self.targetVars:
        
            model = '../GPRModels2p5n8/'+prefix+self.testPrefix+'_'+QoI+'.pkl'
            y_mean = gp_model.predict(model,fit_features,features)
            
            y_query, model_predictions = scale_predictions(y_mean, self.targetABL, alpha, QoI)
            
            target = self.targetABL.loc[(self.targetABL['y']>=np.min(y_query)) & (self.targetABL['y']<=np.max(y_query)),QoI].to_numpy()
            
            scores[cont] = np.linalg.norm(model_predictions-target)
            cont+=1
            
        return scores

    def _evaluate(self, params, out, *args, **kwargs):
    
        features = ['y','h','r']
        
        h = params[:,0]
        x = 3.6
        r = params[:,1]
        alpha = params[:,2]
    
        x = 3.6
        trainPoints = {'h':self.trainPoints[:,0],'r':self.trainPoints[:,1],'x':[x]}
        devPoints = {'h':self.devPoints[:,0],'r':self.devPoints[:,1],'x':[x]}
        testPoints = {'h':self.testPoints[:,0],'r':self.testPoints[:,1],'x':[x]}
        gp_model = gaussianProcess(trainPoints, devPoints, testPoints, self.yMax, self.datasetLocation)
        
        nIndividuals = params.shape[0]
        
        temp = joblib.Parallel(n_jobs=12)(joblib.delayed(self.eval_model_delayed)(params[i,0],x,params[i,1],params[i,2],gp_model) for i in range(nIndividuals))
        
        self.nGenerations += 1
        
        print(self.nGenerations)
            
        out["F"] = np.array(temp)




PFDatabase = '../../TIGTestMatrixLong2p5n8'
fNames = ['themisABL']

testID = 'intensities'

hList = [0.08,0.10,0.14]
xList = [0.6,1.5,3.0,11.0]
alphaList = [1.0]
rList = [52,62,72,82,92]

targetQoIs = ['u','Iu']

everyQoI = ['u','Iu','Iv','Iw']

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
    if 'u min' in header:
        ref_abl['u min'] = ref_abl['u min']/Uref
    if 'u max' in header:
        ref_abl['u max'] = ref_abl['u max']/Uref
    QoIs = []

    for QoI in everyQoI:
        if (QoI in header) and (QoI in targetQoIs):
            QoIs.append(QoI)
            
    ######## Optimization starts

        
    algorithm = NSGA2(pop_size=48)

    problemUser = MyProblem(trainPairs,devPairs,testPairs,yMax,ref_abl,QoIs,PFDatabase,testID)
    resUser = minimize(problemUser,
                algorithm,
                ('n_gen', 20),
                seed=1,
                verbose=False)
    
print(resUser.F)
print(resUser.X)
#input()


plot = Scatter()
plot.add(resUser.F, facecolor="none",edgecolor='tab:blue')
plot.show()

