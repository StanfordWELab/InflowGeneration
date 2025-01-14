import copy
import sys
import os
import math
import numpy           as np
import multiprocessing as mp
import pandas as pd
import scipy as sp
import re
from scipy.interpolate import Rbf
from scipy             import optimize
from scipy.optimize    import minimize
from scipy.optimize    import Bounds


from joblib         import Parallel, delayed
#from scipy.signal   import welch, hanning
from scipy.optimize import curve_fit
from scipy.stats    import pearsonr


import matplotlib
matplotlib.use( 'TkAgg' )
import matplotlib.pyplot as plt

listQoIs = ['uu','vv','ww','u','uv']
directories = {'../PFh0.16Long/':[r'PF h=0.04',[25000,100000],'tab:orange']}

dictDF = {'./inflowProfile_U10_Cat1_1uu1vv1ww.dat':[r'Cat $1^{U_10}$','tab:green']
         ,'./inflowProfile_U10_Cat2_1uu1vv1ww.dat':[r'Cat $2^{U_10}$','tab:blue']
         ,'./inflowProfile_U10_Cat3_1uu1vv1ww.dat':[r'Cat $3^{U_10}$','tab:red']
         ,'./inflowProfile_U10_Cat4_1uu1vv1ww.dat':[r'Cat $4^{U_10}$','tab:grey']
         ,'./inflowProfile_Cat2_1uu1vv1ww.dat':[r'Cat $2^*$','tab:cyan']
         ,'./inflowProfile_Cat3_1uu1vv1ww.dat':[r'Cat $3^*$','tab:orange']}

prefixes = ['4p0_','6p0_','8p0_','10p0_','12p0_','14p0_']

my_dpi = 100
plt.figure(figsize=(2260/my_dpi, 630/my_dpi), dpi=my_dpi)

for DF in dictDF:

    outputDF = pd.read_csv(DF,sep=',')
    outputDF = outputDF.sort_values(by=['y'])
    len_df = len(outputDF)
    yMax   = (outputDF['y'].to_numpy())[len_df-1]
    U_yMax = (outputDF['u'].to_numpy())[len_df-1]

    cont = 230

    for stress in listQoIs:
        cont+=1
        plt.subplot(cont)
        if ('uv' in stress) or ('vw' in stress) or ('uw' in stress):
            QoIPlot = outputDF[stress]/(U_yMax**2)
        else:
            QoIPlot = outputDF[stress]/(U_yMax)
            
        plt.plot(QoIPlot,outputDF['y']/yMax,label=dictDF[DF][0],color=dictDF[DF][1])
        plt.title(stress)
        plt.gca().set_yticklabels([])

plt.legend()
plt.savefig('ABLData.png', bbox_inches='tight')
plt.show()
plt.close('all')

