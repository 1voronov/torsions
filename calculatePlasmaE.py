import numpy as np
from scipy.interpolate import interpn
import csv
import sys 
from scipy import interpolate
import time
from fit_drag import fit

class Plasma:
    def __init__(self):
        n_z = 250
        n_r = 50
        n_q = 12
        self.z_line = np.zeros(n_z)
        self.r_line = np.zeros(n_r)

        self.values_f_r_array = np.zeros((n_q, n_r,n_z))
        self.values_f_z_array = np.zeros((n_q, n_r,n_z))
        
        for q_index in range(1,n_q):
            filename = "./data/data_{}/E_r.txt".format(q_index)
            f = open(filename,'r')
            index = 0
            for line in f:
                index_r = int(index/n_z)
                index_z = int(index%n_z)
                self.r_line[index_r] = line.split("\t")[0]
                self.z_line[index_z] = line.split("\t")[1]
                self.values_f_r_array[q_index][index_r][index_z] = line.split("\t")[2]
                index += 1
            f.close()

            filename = "./data/data_{}/E_z.txt".format(q_index)
            f = open(filename,'r')
            index = 0
            for line in f:
                index_r = int(index/n_z)
                index_z = int(index%n_z)
                self.r_line[index_r] = line.split("\t")[0]
                self.z_line[index_z] = line.split("\t")[1]
                self.values_f_z_array[q_index][index_r][index_z] = line.split("\t")[2]
                index += 1
            f.close()
        
        r_D_e = 0.0016622715189364113
        self.r_line = self.r_line/r_D_e
        self.z_line = self.z_line/r_D_e + 1.0
        self.q_line = np.array([-3.3,-3.2,-3.1,-3.0,-2.9,-2.8,-2.7,-2.6,-2.5,-2.4,-2.3,-2.2])*1.0e-15

        self.points = (self.q_line, self.r_line, self.z_line) 

    def calculatePlasmaE(self,x,y,z,q):
        N = x.shape[0]
        n_z = 250
        n_r = 50
        n_q = 12
        r_D_e = 0.0016622715189364113

        ion_drag = np.zeros(self.q_line.shape[0])
        f = open("force_data.txt",'r')
        index = 0
        for line in f:
            ion_drag[index] = float(line.split("\t")[0])
            index += 1
        f.close()
        drag_0, drag_1, drag_2 = fit(self.q_line, ion_drag)
        #drag_force = interpolate.interp1d(self.q_line,ion_drag,bounds_error=False, kind='cubic', fill_value="extrapolate")
        #print(self.q_line, drag_force(self.q_line))
        E_x = np.zeros(N)
        E_y = np.zeros(N)
        E_z = np.zeros(N)

        r_0 = 0.0001
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    r = np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2)
                    h = z[j]-z[i]
                    point = np.array([q[j],r,h])
                    if r <= 0.5 and h <= 2.0 and h >= -0.5:
                        E_r = interpn(self.points, self.values_f_r_array, point, method = "linear", bounds_error=False, fill_value=None)[0]
                        E_h = -interpn(self.points, self.values_f_z_array, point, method = "linear", bounds_error=False, fill_value=None)[0]
                    else:
                        E_r = 0
                        E_h = 0

                    if r > r_0:
                        E_x[i] += q[i]*E_r*(x[i]-x[j])/r
                        E_y[i] += q[i]*E_r*(y[i]-y[j])/r
                    E_z[i] += q[i]*E_h
            E_z[i] += -(drag_0 + drag_1*q[i] + drag_2*q[i]**2)
        return E_x, E_y, E_z


if __name__ == "__main__":
    N = 2
    n = 100
    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    z1 = np.linspace(0.004, 0.001,n)

    r_D_e = 0.0016622715189364113
    z = np.array([0.0032248049595676263, 0.002742747106163061])/r_D_e
    q = np.array([-3.05e-15, -3.05e-15])
    plasma = Plasma()
    Ex, Ey, Ez = plasma.calculatePlasmaE(x,y,z,q)
    '''    
    for i in range(n):
        z = np.array([0.0032248049595676263/r_D_e, z1[i]/r_D_e])
        Ex, Ey, Ez = plasma.calculatePlasmaE(x,y,z,q)
        print(z1[i], Ez[1])
    '''
