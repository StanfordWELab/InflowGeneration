import numpy  as np
import pandas as pd
from matplotlib import pyplot as plt
    
rDictInflow = {'52':'21p55'
              ,'57':'23p05'
              ,'62':'24p55'
              ,'67':'26p05'
              ,'72':'27p55'
              ,'77':'29p05'
              ,'82':'30p55'
              ,'87':'32p05'
              ,'92':'33p55'}

rDictALF    = {'52':'23p65'
              ,'57':'25p15'
              ,'62':'26p65'
              ,'67':'28p15'
              ,'72':'29p65'
              ,'77':'31p15'
              ,'82':'32p65'
              ,'87':'34p15'
              ,'92':'35p65'}


yMax = 1.5

for h in [0.04,0.06,0.08,0.12,0.14,0.16]:
    
    if h in [0.04,0.08,0.12,0.16]:
        rList = [52,62,72,82,92]
    
    elif h in [0.06,0.14]:
        rList = [57,67,77,87]
        
    my_dpi = 100
    plt.figure(figsize=(2260/my_dpi, 1300/my_dpi), dpi=my_dpi)
    for r in rList:
        
        inflow_df = (pd.read_csv('../../TIGTestMatrixLong/InflowProfiles/Dragh'+'{0:.2f}'.format(h)+'_'+rDictInflow[str(r)]+'_inflow_turbulence.txt',sep='\t'))
        inflow_df['uv-reynolds-stress'] = np.abs(inflow_df['uv-reynolds-stress'])
        inflow_df['pseudo_Iu']  = np.sqrt(inflow_df['uu-reynolds-stress'])/inflow_df['x-velocity']
        inflow_df['pseudo_Iv']  = np.sqrt(inflow_df['vv-reynolds-stress'])/inflow_df['x-velocity']
        inflow_df['pseudo_Iw']  = np.sqrt(inflow_df['ww-reynolds-stress'])/inflow_df['x-velocity']
        inflow_df['pseudo_Iuv'] = np.sqrt(inflow_df['uv-reynolds-stress'])/inflow_df['x-velocity']
        inflow_df['x-velocity-magnitude'] = -1
        inflow_df['x'] = -4.95
        inflow_df['h'] = h
        inflow_df['r'] = r
        
        ALF_df    = (pd.read_csv('../../TIGTestMatrixLong/InflowProfiles/Dragh'+'{0:.2f}'.format(h)+'_'+rDictALF[str(r)]+'_inflow_turbulence.txt',sep='\t'))
        ALF_df['uv-reynolds-stress'] = np.abs(ALF_df['uv-reynolds-stress'])
        ALF_df['pseudo_Iu']  = np.sqrt(ALF_df['uu-reynolds-stress'])/ALF_df['x-velocity']
        ALF_df['pseudo_Iv']  = np.sqrt(ALF_df['vv-reynolds-stress'])/ALF_df['x-velocity']
        ALF_df['pseudo_Iw']  = np.sqrt(ALF_df['ww-reynolds-stress'])/ALF_df['x-velocity']
        ALF_df['pseudo_Iuv'] = np.sqrt(ALF_df['uv-reynolds-stress'])/ALF_df['x-velocity']
        ALF_df['x-velocity-magnitude'] = -1
        ALF_df['x'] = -2.85
        ALF_df['h'] = h
        ALF_df['r'] = r
        
        database_df = (pd.read_csv('../GPRDatabase/PFh'+'{0:.2f}'.format(h)+'u15r'+'{0:.0f}'.format(r)+'.txt',sep='\t'))
        database_df['pseudo_Iu']  = np.sqrt(database_df['uu-reynolds-stress'])/database_df['x-velocity']
        database_df['pseudo_Iv']  = np.sqrt(database_df['vv-reynolds-stress'])/database_df['x-velocity']
        database_df['pseudo_Iw']  = np.sqrt(database_df['ww-reynolds-stress'])/database_df['x-velocity']
        database_df['pseudo_Iuv'] = np.sqrt(database_df['uv-reynolds-stress'])/database_df['x-velocity']
        
        allDFs = pd.concat([inflow_df, ALF_df, database_df], ignore_index=True)
        
        #allDFs.to_csv('./GPRDatabase/PFh'+'{0:.2f}'.format(h)+'u15r'+'{0:.0f}'.format(r)+'.txt', sep='\t', index=False)
        
    
        cont = 231
        for QoI in ['x-velocity-magnitude','x-velocity','uu-reynolds-stress','vv-reynolds-stress','ww-reynolds-stress','uv-reynolds-stress']:
            plt.subplot(cont)
            #plt.plot(inflow_df[QoI],inflow_df['y'], label = 'Inflow'+rDictInflow[str(r)])
            #plt.plot(ALF_df[QoI],ALF_df['y'], label = 'A'+rDictALF[str(r)])
            x = -4.95
            yPlot = np.sqrt(allDFs.loc[(abs(allDFs['x'] - x) < 1e-6) & (allDFs['y']<yMax), ['y']])
            #plt.plot(database_df.loc[(abs(database_df['x'] - x) < 1e-6),[QoI]]/database_df.loc[(abs(database_df['x'] - x) < 1e-6),['x-velocity-magnitude']].to_numpy(),database_df.loc[(abs(database_df['x'] - x) < 1e-6), ['y']], label = 'uMag, '+str(r))
            #plt.plot(database_df.loc[(abs(database_df['x'] - x) < 1e-6),[QoI]]/database_df.loc[(abs(database_df['x'] - x) < 1e-6),['x-velocity']].to_numpy(),database_df.loc[(abs(database_df['x'] - x) < 1e-6), ['y']], label = 'ux, '+str(r))
            plt.plot(allDFs.loc[(abs(allDFs['x'] - x) < 1e-6) & (allDFs['y']<yMax),[QoI]],yPlot, label = 'ux, '+str(r))
            plt.title(QoI)
            cont +=1
    plt.suptitle('h = ' +str(h))
    plt.legend()
        
    plt.show()
    plt.close('all')
        
        
        