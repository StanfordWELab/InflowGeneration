import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


#######################################################

#buildings = {'LRB':[6.0,[1,2],[1/25,1/10,1/1]]
            #,'MRB':[30.0,[2,3,4],[1/100,1/50,1/10,1/1]]
            #,'HRB':[100.0,[3,4],[1/100,1/50,1/10,1/1]]}


#buildings = {'LRB':[6.0,[1,2],[1/50,1/25,1/10,1/1]]}
buildings = {'MRB':[30.0,[2,3,4],[1/100,1/50]]}
            #,'HRB':[100.0,[3,4]]}

normalize = False


#######################################################

kappa = 0.41

for bldg in buildings:
    
    H = buildings[bldg][0]
    
    for scale in buildings[bldg][2]:
    
        for tcat in buildings[bldg][1]:
                
            zmin = 0.2*H
            
            setup = pd.DataFrame()

            # Define building specific parameters
            z0list = [0.003, 0.01, 0.05, 0.3, 1.0]

            UF, VF, WF = 4, 4, 4
            xIn = -4.95

            z0 = z0list[tcat]
            lat = 51.1 * np.pi / 180

            Uref_f = 25
            zref_f = 10
            z0_f = 0.05
            zmin_f = 2
            
            setup['y'] = np.arange(zmin, H*1.5+1e-6, 0.05*H)

            # Coriolis factor
            omega = 7.27e-5
            fc = 2 * omega * np.sin(lat)

            # Estimate u_star
            ustar_f = Uref_f * kappa / np.log((zref_f + z0_f) / z0_f)

            # Surface Layer and ABL Height
            z_ASL_f = 0.02 * ustar_f / fc
            zg_ABL_f = 0.08 * ustar_f / fc

            # Velocity profile
            U_ASL_f = (ustar_f / kappa) * np.log(setup['y'] / z0_f)
            Ug_ASL_f = (ustar_f / kappa) * np.log(zg_ABL_f / z0_f)
            
            muN1 = 300;
            muN2 = 50
            zg_ABL1 = zg_ABL_f
            U_ABL1 = (ustar_f / kappa) * (np.log(setup['y'] / z0_f) + (1/290) * (ustar_f / (fc * zg_ABL1))**0.3 * ((muN2 * fc / ustar_f) * setup['y']) ** 2)
            Ug_ABL1 = (ustar_f / kappa) * (np.log(zg_ABL1 / z0_f) + (1/290) * (ustar_f / (fc * zg_ABL1))**0.3 * ((muN2 * fc / ustar_f) * zg_ABL1) ** 2)

            # Eurocode correction
            kr = 0.19 * (z0 / z0_f) ** 0.07
            cr = kr * np.log(zref_f / z0)
            Uref = cr * Uref_f
            ustar = Uref * kappa / np.log((zref_f + z0) / z0)
            U_ASL_Eurocode = (ustar / kappa) * np.log(setup['y'] / z0)

            # Alternative correction
            fun = lambda ustar: Ug_ASL_f - (ustar / kappa) * np.log((0.08 * ustar / fc) / z0)
            ustar = fsolve(fun, ustar_f)[0]
            U_ASL_c = (ustar / kappa) * np.log(setup['y'] / z0)
            
            zg_ASL_c = 0.08*ustar/fc;
            U_ABL_c = (ustar / kappa) * (np.log(setup['y'] / z0) + (1/290) * (ustar / (fc * zg_ASL_c))**0.3 * ((muN2 * fc / ustar) * setup['y']) ** 2)

            # Turbulence intensities
            setup['u'] = (ustar / kappa) * np.log((setup['y'] + z0) / z0)
            setup['Iu'] = np.sqrt(5.25 * ustar**2) / setup['u']
            setup['Iv'] = 0.5 * setup['Iu']
            setup['Iw'] = 0.8 * setup['Iu']

            # Reynolds stresses
            uu = (setup['Iu']*setup['u']) ** 2
            vv = (setup['Iv']*setup['u']) ** 2
            ww = (setup['Iw']*setup['u']) ** 2
            uw = -ustar**2 * np.ones_like(setup['y'])

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
            #plt.show()
            plt.close('all')
            
            setup.to_csv('TestCases/'+bldg+'_Cat'+str(int(tcat))+'_scale1to'+str(int(1/scale))+'.dat',index=False)
