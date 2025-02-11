import numpy as np
import pandas as pd
        
database_df = (pd.read_csv('/home/mattiafc/Downloads/test_WOW_x2.5_mod.dat',sep='\t'))

database_df = database_df.rename(columns={'y':'z','z':'y','y-velocity':'z-velocity','z-velocity':'y-velocity','vv-reynolds-stress':'ww-reynolds-stress','ww-reynolds-stress':'vv-reynolds-stress','uv-reynolds-stress':'uw-reynolds-stress','uw-reynolds-stress':'uv-reynolds-stress'})
database_df['Iu']  = np.sqrt(database_df['uu-reynolds-stress'])/database_df['x-velocity']
database_df['Iv']  = np.sqrt(database_df['vv-reynolds-stress'])/database_df['x-velocity']
database_df['Iw']  = np.sqrt(database_df['ww-reynolds-stress'])/database_df['x-velocity']
database_df['Iuv'] = np.sqrt(abs(database_df['uv-reynolds-stress']))/database_df['x-velocity']

database_df.to_csv('TestCases/WOW_x2.5.dat',index=False)