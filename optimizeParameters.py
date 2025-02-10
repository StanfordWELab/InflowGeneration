import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from modelDefinition import *

font={'size'   : 15}
matplotlib.rc('font', **font)
    
###############################################################################################

PFDatabase = './GPRDatabase'

#fNames = ['testABL']
#fNames = ['TPU_ABL']
fNames = ['themisABL']
#fNames = ['inflowProfile_U10_Cat4_1uu1vv1ww']
#fNames = ['inflowProfile_U10_Cat1_1uu1vv1ww','inflowProfile_U10_Cat2_1uu1vv1ww','inflowProfile_U10_Cat3_1uu1vv1ww','inflowProfile_U10_Cat4_1uu1vv1ww']
#fNames = ['inflowProfile_Cat2_1uu1vv1ww','inflowProfile_Cat3_1uu1vv1ww']
#fNames = ['TPU_highrise_14_middle_higher']
#fNames = ['TPU_highrise_14_middle']


##### Define these variables for Optimize-Gridsearch ####
#hList = [0.04,0.06,0.08,0.10,0.12,0.14,0.16]
#xList = [[0.3,0.6,0.9,1.2,1.5,2.1,2.4,2.7,3.0,3.3,3.6,4.0,5.0,6.0,7.0,9.0,11.0,13.0]]
#alphaList = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#rList = [52,57,62,67,72,77,82,87,92]
#weights={'u':0.25,'Iu':0.25,'Iv':0.25,'Iw':0.25}


#### Define these variables for Optimize-NSGA ####
xList = [0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3.0,3.3,3.6,4.0,5.0,6.0,7.0,9.0,11.0,13.0]
variables = {r'$h$':[0.035,0.1649],r'$r$':[51.5,92.49],r'$\alpha$':[0.3,0.5],r'$x$':[-0.49,len(xList)-0.51]}
population_size = 96
n_generations = 100


##### Define these variables for Plot-Gridsearch ####
#hSearch = hList
#xSearch = xList
#alphaSearch = alphaList
#rSearch = rList
#nResults = 4

#### metric is 'RMSE' or 'RMSE relative'
metric = 'RMSE'
uncertainty = True
testID = 'intensities'
nCpu = 12

targetQoIs = ['u','Iu','Iv','Iw']

###############################################################################################

options = ['Optimize-Gridsearch','Optimize-NSGA','Plot-Gridsearch','Plot-NSGA','Plot-Setup']
everyQoI = ['u','Iu','Iv','Iw']
features = ['y','h','r']
varNames = [r'$h$',r'$r$',r'$\alpha$',r'$x$']

yMax= 1.0

hTrain = [0.04,0.08,0.12,0.16]
rTrain = [52,62,72,82,92]

trainPairs = np.zeros((len(hTrain)*len(rTrain),2))
cont=0
for h in hTrain:
    for r in rTrain:
        trainPairs[cont,:] = [h, r]
        cont+=1

devPairs = np.array([[0.06,57],[0.06,87],[0.14,67],[0.14,77]])
testPairs = np.array([[0.06,67],[0.06,77],[0.14,57],[0.14,87]])

try: 
    mode = sys.argv[1]
except:
    mode = []
    
while not(mode in options):
    print('Choose one of the following options:')
    for opt in options:
        print('   - '+opt)
    mode = input('\n')
    
    if not(mode in options):
        print('Invalid selection, try again')


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
            
    effectiveQoIs = []

    for QoI in everyQoI:
        if (QoI in header) and (QoI in targetQoIs):
            effectiveQoIs.append(QoI)

    if mode == 'Optimize-Gridsearch':

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
            yMax,exploration_matrix[i,:],ref_abl,features,effectiveQoIs,PFDatabase,testID) for i in range(nTests))

        df = pd.DataFrame()
        for dct in temp:
            if df.empty:
                df = pd.DataFrame([dct])
            else:
                df = pd.concat([df, pd.DataFrame([dct])], ignore_index=True)

        df.to_csv('TestCases/'+fName+'_optimal.csv',index=False,index_label=False)
    
    elif mode =='Optimize-NSGA':

        algorithm = NSGA2(pop_size=population_size)

        problemUser = MyProblem(variables,trainPairs,devPairs,testPairs,yMax,xList,ref_abl,features,effectiveQoIs,PFDatabase,testID)
        resUser = minimize(problemUser, algorithm, ('n_gen', n_generations), seed=1, verbose=False)

        # Save Pareto front (Objective values) to CSV
        df_obj = pd.DataFrame(resUser.F, columns=[effectiveQoIs[i] for i in range(resUser.F.shape[1])])
        df_obj.to_csv('TestCases/'+fName+"_pareto_front.csv", index=False)

        # Save Decision Variables to CSV
        df_dec = pd.DataFrame(resUser.X, columns=[varNames[i] for i in range(resUser.X.shape[1])])
        df_dec.to_csv('TestCases/'+fName+"_decision_variables.csv", index=False)
        
    elif mode =='Plot-NSGA':

    # Create a pymoo-like result object
        RMSEValues    = pd.read_csv('TestCases/'+fName+"_pareto_front.csv")
        effectiveQoIs = RMSEValues.columns.tolist()
        decisionVars  = pd.read_csv('TestCases/'+fName+"_decision_variables.csv")
        
        NSGA_results = pd.concat([RMSEValues, decisionVars], axis=1).sort_values(by='u', ascending=False)
        NSGA_results['$x$'] = [xList[x] for x in NSGA_results['$x$'].round().astype(int).to_numpy()]
        NSGA_results['$h$'] = ((NSGA_results['$h$']*100).round().astype(int)/100).to_numpy()
        NSGA_results['$r$'] = NSGA_results['$r$'].round().astype(int).to_numpy()
        NSGA_results[r'$\alpha$'] = NSGA_results[r'$\alpha$'].round(2).astype(float).to_numpy()
        NSGA_results = NSGA_results.drop_duplicates(subset=varNames, keep='first').reset_index(drop=True)
        print(NSGA_results.to_string())
        
        resUser = Result()
        resUser.F = NSGA_results[effectiveQoIs].to_numpy()   # Pareto front
        resUser.X = NSGA_results[varNames].to_numpy()  # Decision variables

        parallelCoordinatesPlot(resUser, xList, varNames, effectiveQoIs)

    elif mode == 'Plot-Gridsearch':

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

        #plt.savefig('../RegressionPlots/'+fName+'_optimal.png', bbox_inches='tight')
        plt.show()
        plt.close('all')

    elif mode == 'Plot-Setup':
        
        my_dpi = 100
        plt.figure(figsize=(2260/my_dpi, 1300/my_dpi), dpi=my_dpi)

        cont=1
        
        stop = 'n'
        NSGA = ''
        while not(NSGA in ['y','n']):
            NSGA = input('Do you want to evaluate NSGA soutions?(y/n)\n')
            
        if NSGA == 'y':
            
            RMSEValues    = pd.read_csv('TestCases/'+fName+"_pareto_front.csv")
            effectiveQoIs = RMSEValues.columns.tolist()
            decisionVars  = pd.read_csv('TestCases/'+fName+"_decision_variables.csv")
            
            NSGA_results = pd.concat([RMSEValues, decisionVars], axis=1).sort_values(by='u', ascending=False)
            NSGA_results['$x$'] = [xList[x] for x in NSGA_results['$x$'].round().astype(int).to_numpy()]
            NSGA_results['$h$'] = ((NSGA_results['$h$']*100).round().astype(int)/100).to_numpy()
            NSGA_results['$r$'] = NSGA_results['$r$'].round().astype(int).to_numpy()
            NSGA_results[r'$\alpha$'] = NSGA_results[r'$\alpha$'].round(2).astype(float).to_numpy()
            NSGA_results = NSGA_results.drop_duplicates(subset=varNames, keep='first').reset_index(drop=True)
            print(NSGA_results.to_string())
            
            listSolutions = []
            
            while listSolutions == [] or not(isinstance(listSolutions, list)):
                
                listSolutions = input('List the pareto front dataset entries you want to visualize as numbers separated by spaces\n')
                #listSolutions = eval(listSolutions)
                try: 
                    listSolutions = list(map(int, listSolutions.split()))
                except:
                    print('Wrong format! You did not provide either a list or a list of integers. Try again\n')
            
            print('\n=================================\n')
            print('Visualizing entries ' +str(listSolutions))
            print('\n=================================\n')
            
            parameters = NSGA_results.loc[listSolutions, [r'$h$', r'$r$', r'$\alpha$', r'$x$']].to_dict(orient='list')
            
        else:
            hParams     = []
            rParams     = []
            alphaParams = []
            xParams     = []
            
            escape = 't'
            while not(escape == 'y'):
                more = 'a'
                
                listSolutions = input('What h, r, alpha, and x values do you want to plot? Write them as numbers separated by spaces\n')
                try:
                    listSolutions = list(map(float, listSolutions.split()))
                except:
                    listSolutions = [None]
                    
                if len(listSolutions) == 4:
                    h     = listSolutions[0]
                    r     = listSolutions[1]
                    alpha = listSolutions[2]
                    x     = listSolutions[3]
                    
                    if (h<0.04) or (h>0.16) or (r<52) or (r>92) or (alpha<0.1) or alpha>1.0 or not(x in xList):
                        print('Bad choice buddy...')
                        if (h<0.04) or (h>0.16):
                            print('h should be in [0.04,0.16]!')
                        if (r<52) or (r>92):
                            print('r should be in [52,92]!')
                        if (alpha<0.1) or (alpha>1.0):
                            print('alpha should be in ]0.1,1.0],!')
                        if not(x in xList):
                            print('x should be in '+str(xList)+'!')
                        print('Try again\n')
                    else:
                        hParams.append(h)
                        rParams.append(r)
                        alphaParams.append(alpha)
                        xParams.append(x)
                        while not(more in ['y','n']):
                            if more == 'a':
                                print('So far you chose:')
                                print('h     = '+str(hParams))
                                print('r     = '+str(rParams))
                                print('alpha = '+str(alphaParams))
                                print('x     = '+str(xParams))
                            more = input('Wanna add more test setups?(y/n)\n')
                            
                else:
                    print('You need to specify four numerical parameters, try again!\n')
                        
                if more == 'n':
                    escape = 'y'
                    
                        
            parameters = {r'$h$':hParams,r'$r$':rParams,r'$\alpha$':alphaParams,r'$x$':xParams}
            
            print('\n=================================\n')
            print('Visualizing setups ' +str(parameters))
            print('\n=================================\n')

        for QoI in ['u','Iu','Iv','Iw']:
            
            for i in range(len(parameters[r'$h$'])):

                xTemp = parameters[r'$x$'][i]
                hTemp = parameters[r'$h$'][i]
                rTemp = parameters[r'$r$'][i]
                alphaTemp = parameters[r'$\alpha$'][i]

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
                    
                plt.subplot(2,2,cont)
                
                if i==0 and (QoI in header):
                    plt.plot(ref_abl[QoI],ref_abl['y'],color='tab:red',label='Target',linewidth=3)
                    plt.fill_betweenx(ref_abl['y'], ref_abl[QoI]*0.9, ref_abl[QoI]*1.1, color='tab:red', alpha=0.2,label=r'Reference $\pm$10%')
                    
                line = plt.plot(y_mean['y_model'],y_mean['y'],linestyle='--',linewidth=3
                        ,label=str(i+1)+r': x='+'{0:.2f}'.format(xTemp)+'m,h='+'{0:.2f}'.format(hTemp)+'m'+r'm,$\alpha$='+'{0:.2f}'.format(alphaTemp)+r',r='+'{0:.2f}'.format(rTemp))
                
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

        plt.savefig('../RegressionPlots/'+fName+'_test.png', bbox_inches='tight')
        plt.show()
        plt.close('all')


