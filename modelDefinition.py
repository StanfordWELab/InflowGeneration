import random
from datetime import datetime
import joblib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, WhiteKernel
from scipy.interpolate import interp1d
from modelDefinition import *
import warnings

def loadData(heights, location, roughness, yMax, plot=False, home='../../TIGTestMatrixLong/', yInterp=False):
    ymin = 0.005
    u=15
    data = pd.DataFrame()
    
    for i in range(len(heights)):
            
        h=heights[i]
        r=roughness[i]
            
        for x in location:
            
            prefix = str(str(x)+'_').replace('.','p')
                
            #if h == 0.08 and (r == 32 or r == 42):
                #u = 18
            #elif h == 0.12 and r == 52:
                #u = 12
            #else:
                #u = 15
            
            temp         = pd.DataFrame()
            interpolated = pd.DataFrame()
            #print(r)
            
            avg_u = np.loadtxt(home+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(r)+'/'+prefix+'avg_u.00075000.collapse_width.dat',skiprows = 3)

            Umag = np.loadtxt(home+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(r)+'/'+prefix+'mag_u.00075000.collapse_width.dat',skiprows = 3)

            rms_u = np.loadtxt(home+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(r)+'/'+prefix+'rms_u.00075000.collapse_width.dat',skiprows = 3)
            rms_v = np.loadtxt(home+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(r)+'/'+prefix+'rms_v.00075000.collapse_width.dat',skiprows = 3)
            rms_w = np.loadtxt(home+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(r)+'/'+prefix+'rms_w.00075000.collapse_width.dat',skiprows = 3)

            uv = np.loadtxt(home+'/PFh'+'{0:.2f}'.format(h)+'u'+'{0:.0f}'.format(u)+'r'+'{0:.0f}'.format(r)+'/'+prefix+'uv.00075000.collapse_width.dat',skiprows = 3)
            
            temp['y'] = avg_u[(avg_u[:,3]>=ymin) & (avg_u[:,3]<=yMax),3]/yMax
            #temp['x'] = avg_u[(avg_u[:,3]>ymin) & (avg_u[:,3]<yMax),2]-29.5
            temp['x'] = x
            temp['u'] = avg_u[(avg_u[:,3]>=ymin) & (avg_u[:,3]<=yMax),5]
            temp['umin'] = avg_u[(avg_u[:,3]>=ymin) & (avg_u[:,3]<=yMax),6]
            temp['uMax'] = avg_u[(avg_u[:,3]>=ymin) & (avg_u[:,3]<=yMax),7]
            temp['Iu'] = rms_u[(rms_u[:,3]>=ymin) & (rms_u[:,3]<=yMax),5]/Umag[(Umag[:,3]>=ymin) & (Umag[:,3]<=yMax),5]
            temp['Iv'] = rms_v[(rms_v[:,3]>=ymin) & (rms_v[:,3]<=yMax),5]/Umag[(Umag[:,3]>=ymin) & (Umag[:,3]<=yMax),5]
            temp['Iw'] = rms_w[(rms_w[:,3]>=ymin) & (rms_w[:,3]<=yMax),5]/Umag[(Umag[:,3]>=ymin) & (Umag[:,3]<=yMax),5]
            temp['Iuv'] = abs(uv[(uv[:,3]>=ymin) & (uv[:,3]<=yMax),5]/(Umag[(Umag[:,3]>=ymin) & (Umag[:,3]<=yMax),5]**2))
            temp['uu'] = rms_u[(rms_u[:,3]>=ymin) & (rms_u[:,3]<=yMax),5]**2
            temp['vv'] = rms_v[(rms_v[:,3]>=ymin) & (rms_v[:,3]<=yMax),5]**2
            temp['ww'] = rms_w[(rms_w[:,3]>=ymin) & (rms_w[:,3]<=yMax),5]**2
            temp['uv'] = abs(uv[(uv[:,3]>=ymin) & (uv[:,3]<=yMax),5])
            temp['h'] = h
            temp['r'] = r
            
            U_yMax_interp = (interp1d(avg_u[:,3], avg_u[:,5])(yMax)).item()
            #U_yMax_temp = np.max(temp['u'])
            #U_normalize = max([U_yMax_interp,U_yMax_temp])
            temp['u'] = temp['u']/U_yMax_interp
            temp['uu'] = temp['uu']/(U_yMax_interp**2)
            temp['uv'] = temp['uv']/(U_yMax_interp**2)
            temp['vv'] = temp['vv']/(U_yMax_interp**2)
            temp['ww'] = temp['ww']/(U_yMax_interp**2)
            temp['umin'] = temp['umin']/U_yMax_interp
            temp['uMax'] = temp['uMax']/U_yMax_interp
            
            lastRow={'x':x, 'y':yMax/yMax, 'u':U_yMax_interp/U_yMax_interp, 'umin':(interp1d(avg_u[:,3], avg_u[:,6])(yMax)).item()/U_yMax_interp, 'uMax':(interp1d(avg_u[:,3], avg_u[:,7])(yMax)).item()/U_yMax_interp
                    ,'Iu':(interp1d(rms_u[:,3], rms_u[:,5]/Umag[:,5]))(yMax).item(), 'Iv':(interp1d(rms_v[:,3], rms_v[:,5]/Umag[:,5]))(yMax).item()
                    ,'Iw':(interp1d(rms_w[:,3], rms_w[:,5]/Umag[:,5]))(yMax).item(), 'Iuv':(interp1d(uv[:,3], np.sqrt(abs(uv[:,5])/(Umag[:,5]**2))))(yMax).item()
                     ,'uu':((interp1d(rms_u[:,3], rms_u[:,5]**2))(yMax).item())/(U_yMax_interp**2), 'vv':((interp1d(rms_v[:,3], rms_v[:,5]**2))(yMax).item())/(U_yMax_interp**2)
                    ,'ww':((interp1d(rms_w[:,3], rms_w[:,5]**2))(yMax).item())/(U_yMax_interp**2),'uv':abs((interp1d(uv[:,3], uv[:,5]))(yMax).item())/(U_yMax_interp**2),'h':h,'r':r}
            
            temp = pd.concat([pd.DataFrame([lastRow]),temp], ignore_index=True)
            
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
            print('Model loaded at prediction time')
        
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
    
try:
    os.mkdir('../GPRModels')
except:
    print('GPRModels directory already exist!')

try:
    os.mkdir('../RegressionPlots')
except:
    print('RegressionPlots directory already exist!')
