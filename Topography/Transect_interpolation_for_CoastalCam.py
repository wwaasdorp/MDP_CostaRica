#This code is for altering the local transect and interpolate, so that it can be used as the iz input for matlab CoastalCam

unknown1 = 3
unknown2 = 6

#import
import numpy as np
import matplotlib.pyplot as plt

#########known#########
laserheight = 0.02
#well base
z_wells = [3.6, 3.301, 2.981] #height wells above sea level
distance_wells = [0, unknown1, unknown2] #distance from well 1 (either 228, or 226 degree North)


#########input 04_10_24#########
#daily height fluctuations
fluc_wells = [23.2, 19.4, 19.8] / 100 #height well above sand level

transect_dist = [3.353, 6.203, 9.049, 11.871, 14.757, 17.582, 20.509, 23.422, 26.228] #distance from well 2
transect_z = [-0.575, -0.812, -1.03, -1.23, -1.49, -1.7, -1.94, -2.15, -2.35]

#########automatic part#########
#combine into one dist/z list
dist = distance_wells + (transect_dist - distance_wells[1]) #distance from well 1
z = z_wells -fluc_wells + (transect_z - fluc_wells[1] - laserheight) #height above sea level

#print
print(dist)
print(z)

#plot
plt.plot(dist, z)