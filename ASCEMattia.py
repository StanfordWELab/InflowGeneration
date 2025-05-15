import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


#######################################################

#buildings = {'LRB':[6.0,[1,2],[1/25,1/10,1/1]]
            #,'MRB':[30.0,[2,3,4],[1/100,1/50,1/10,1/1]]
            #,'HRB':[100.0,[3,4],[1/100,1/50,1/10,1/1]]}


#buildings = {'LRB':[6.0,[1,2],[1/50,1/25,1/10,1/1]]}
#buildings = {'HRB':[100.0,['D','C','B'],[1.0/1.0]]}
buildings = {'LRB':[6.0,['D','C','B'],[1.0/1.0]]
            ,'MRB':[30.0,['D','C','B'],[1.0/1.0]]
            ,'HRB':[100.0,['D','C','B'],[1.0/1.0]]}
#buildings = {'HRB':[100.0,['D','C','B'],[1.0/1.0]]}

normalize = False


#######################################################

#kappa = 0.41
#exposure = {'A' :{'z0':2.000,'beta':4.40}
           #,'AB':{'z0':0.700,'beta':4.40}
           #,'B': {'z0':0.300,'beta':5.25}
           #,'BC':{'z0':0.150,'beta':5.25}
           #,'C': {'z0':0.020,'beta':6.00}
           #,'CD':{'z0':0.010,'beta':6.00}
           #,'D': {'z0':0.005,'beta':6.50}}

kappa = 0.41
exposure = {'B': {'z0':0.300,'beta':5.25}
           ,'C': {'z0':0.020,'beta':6.00}
           ,'D': {'z0':0.005,'beta':6.50}}

lat = 51.1 * np.pi / 180
xIn = -4.95

d = 0

### ABL Parameters
Uref_f = 12
zref_f = 10
#z0_f   = 0.05
zmin_f = 2
omega  = 7.27e-5

for bldg in buildings:
    
    H = buildings[bldg][0]
    
    for scale in buildings[bldg][2]:
    
        for tcat in buildings[bldg][1]:
            
            setup = pd.DataFrame()
            zmin = 0.2*H
            z0   = exposure[tcat]['z0']
            beta = exposure[tcat]['beta']
            
            setup['y'] = np.arange(zmin, H*1.5+1e-6, 0.05*H)

            # Estimate u_star (C2-1)
            uStar = Uref_f*kappa/np.log((zref_f -d)/z0)
            #print(uStar, tcat)
            #input()

            # Coriolis factor (C2-2-1 & C2-4)
            f_c = 2 * omega * np.sin(lat)
            z_g = 0.175 * uStar / f_c
            print(f_c)
            input()
            # Surface Layer and ABL Height (C2-3)
            z_s = 0.02 * uStar / f_c
            
            #print('Friction speed       = '+str(uStar)+'m/s')
            #print('Zero gradient height = '+str(z_g)+'m')
            #print('Surface layer height = '+str(z_s)+'m')
            #print('Re = '+str(uStar*z0/(1.5e-5))+' >= 2.5?')
            #input()

            # Velocity profile (C2-9) holds for z<z_s
            U_ASL = (uStar / kappa) * np.log((setup['y']-d) / z0)
            
            # Gradient wind speed (C2-8) holds for z<z_g
            U_g_ASL = (uStar / kappa) * np.log(uStar/np.abs(f_c*z0)+1.0)
            
            # Evaluate capping inversion conditions because they are the worst case scenario
            muN1 = 300
            muN2 = 50
            #U_ABL = (uStar / kappa) * (np.log((setup['y']-d)/ z0) + (1/290)*((uStar/(f_c*z_g))**0.3)*(muN2*f_c/uStar*(setup['y']-d)**2))
            csi = setup['y']/z_g
            U_ABL = (uStar / kappa) * (np.log(setup['y']/ z0) + 5.75*csi-1.87*(csi**2)-1.33*(csi**2)+0.25*(csi**3))

            ## Alternative correction
            #fun = lambda ustar: U_g_ASL - (ustar / kappa) * np.log((0.08 * ustar / f_c) / z0)
            #ustar = fsolve(fun, uStar)[0]
            #U_ASL_c = (ustar / kappa) * np.log(setup['y'] / z0)
            
            #zg_ASL_c = 0.08*ustar/f_c;
            #U_ABL_c = (ustar / kappa) * (np.log(setup['y'] / z0) + (1/290) * (ustar / (f_c * zg_ASL_c))**0.3 * ((muN2 * f_c / ustar) * setup['y']) ** 2)

            # Turbulence intensities
            setup['u']  = 1.0*U_ASL
            #setup['Iu'] = np.sqrt(beta)/(2.5*np.log(setup['y']/z0))
            setup['Iu'] = np.sqrt(beta*np.exp(-1.5*setup['y']/z_g))*uStar/setup['u']
            setup['Iv'] = 0.5 * setup['Iu']
            setup['Iw'] = 0.8 * setup['Iu']

            # Reynolds stresses
            uu = (setup['Iu']*setup['u']) ** 2
            vv = (setup['Iv']*setup['u']) ** 2
            ww = (setup['Iw']*setup['u']) ** 2
            uw = -uStar**2 * np.ones_like(setup['y'])

            # Length scales
            CL, mL = 40, 0.4
            setup['xLu'] = [CL * H ** mL]*len(setup['y'])
            setup['yLu'] = 0.5 * setup['xLu']
            setup['zLu'] = 0.3 * setup['xLu']
            setup['y'] = setup['y']*scale
            
            if normalize ==True:
                normZ = H*scale
                yLab  = 'z/H [-]'
            else:
                normZ = 1.0
                yLab  = 'z [m]'

            # Plot
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1)
            plt.plot(setup['u'], setup['y']/normZ, label='U')
            #plt.plot(U_ABL, setup['y']/normZ, label='U')
            plt.xlabel('U [m/s]')
            plt.ylabel(yLab)
            #plt.xlim([0, 30])

            plt.subplot(1, 3, 2)
            plt.plot(setup['Iu'], setup['y'],label='Iu')
            plt.plot(setup['Iv'], setup['y'],label='Iv')
            plt.plot(setup['Iw'], setup['y'],label='Iw')
            plt.xlabel('I[]')
            plt.ylabel(yLab)
            plt.legend()

            plt.subplot(1, 3, 3)
            plt.plot(setup['xLu'], setup['y'],label='Lu,x')
            plt.plot(setup['yLu'], setup['y'],label='Lv,x')
            plt.plot(setup['zLu'], setup['y'],label='Lw,x')
            plt.xlabel('L[m]')
            plt.ylabel(yLab)
            plt.legend()
            plt.show()
            plt.close('all')
            
            #setup.to_csv('TestCases/'+bldg+'_Cat_'+tcat+'.dat',index=False)
