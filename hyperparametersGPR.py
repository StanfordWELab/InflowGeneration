import numpy as np
# Specify features used as GPR inputs and build feature DataFrame
features = ['y','h','r']

xList = [0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3.0,3.3,3.6,4.0,5.0,6.0,7.0,9.0,11.0,13.0]

hTrain = [0.04,0.08,0.12,0.16]
rTrain = [52,62,72,82,92]

devPairs = np.array([[0.06,57],[0.06,87],[0.14,67],[0.14,77]])
testPairs = np.array([[0.06,67],[0.06,77],[0.14,57],[0.14,87]])

trainPairs = np.zeros((len(hTrain)*len(rTrain),2))
cont=0
for hTemp in hTrain:
    for rTemp in rTrain:
        trainPairs[cont,:] = [hTemp, rTemp]
        cont+=1