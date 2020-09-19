"""
Figure 2.4: The Strain-Controlled Experiment
Parallel fibers hae the same elastic constant k (normalized to unity) and different breaking threshodls x_i
"""

from array import *
import matplotlib.pyplot as plt
import numpy as np

#Parameters:
N = 20 #Number of parallel fibers in a bundle
x_r = 1 #Length scale, normalized
dx = x_r/N #Change in elongation between the threshold values

#Save arrays for the threshold values (x), values in between the thresholds (X), the force on a bundle (F), and the limiting curve of the force per fiber as N --> infinity (f).
xsave = array('f',[])
Xsave = array('f',[])
Fsave = array('f',[])
fsave = array('f',[])

#Calculating the threshold values and the force on the bundle for all bundles:
for i in range(0,21):
    x = (x_r*i)/N
    F = x*(N-i)
    xsave.append(x)
    Xsave.append(x)
    Fsave.append(F)
    
    #Calculating the limiting curve:
    f_x = x*(1-x)
    fsave.append(f_x)
    
    #Number of points in between threshold values
    for j in range(5):
        x = x + dx/5
        #Setting the condition that the force on the bundle is calculated when we go beyond the threshold values:
        if x < (x_r*(i+1)/N):
            Xsave.append(x)
            F = x*(N-i)
            Fsave.append(F)

#Dividing the force per bundle array by the number of fibers in a bundle to get the force per fiber
Fsave = np.divide(Fsave,20)

#print(Fsave)
#print("Xsave:",Xsave)
#print("xsave:",xsave)

#Plotting the force per fiber vs. elongation
plt.plot(Xsave,Fsave,"k",label="Force per fiber")
#plotting the limitting curve
plt.plot(xsave,fsave,"k--",label="limiting curve")
plt.xlabel("x/x_r")
plt.ylabel("F/N")
plt.legend()
plt.show()


