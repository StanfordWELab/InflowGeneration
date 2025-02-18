import numpy  as np
import pandas as pd
from matplotlib import pyplot as plt
fNames = ['inflowProfile_U10_Cat1_1uu1vv1ww','inflowProfile_U10_Cat2_1uu1vv1ww','inflowProfile_U10_Cat3_1uu1vv1ww','inflowProfile_U10_Cat4_1uu1vv1ww']

for fName in fNames:
    
    yMax = 20.0
    ymin = 2.0
    
    target_df = (pd.read_csv(fName+'.dat'))
    target_df['uv'] = np.abs(target_df['uv'])
    target_df['Iu']  = np.sqrt(target_df['uu'])/target_df['u']
    target_df['Iv']  = np.sqrt(target_df['vv'])/target_df['u']
    target_df['Iw']  = np.sqrt(target_df['ww'])/target_df['u']
    target_df['Iuv'] = np.sqrt(target_df['uv'])/target_df['u']
    

    cont = 241
    for QoI in ['u','Iu','Iv','Iw','uu','vv','ww','uv']:
        
        plt.subplot(cont)
        plt.plot(target_df.loc[(target_df['y']>=ymin)&(target_df['y']<=yMax),[QoI]],target_df.loc[(target_df['y']>=ymin)&(target_df['y']<=yMax),['y']])
        plt.title(QoI)
        cont +=1
        
    plt.show()
    plt.close('all')
    target_df = target_df.loc[(target_df['y']>=ymin)&(target_df['y']<=yMax)]
    
    target_df.to_csv('./TestCases/'+fName+'.dat', index=False)
        
        