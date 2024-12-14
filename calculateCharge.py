import numpy as np
from scipy.interpolate import interpn
import csv
import time
from numpy import random
import warnings


class Charge:
    def __init__(self, directory):
        self.directory = directory
        filename = "data_charge.csv"
        f = open(filename, "r")
        reader = csv.reader(f)

        self.values_array = []
        read_index = 0
        for row in reader:
            if read_index > 0:
                self.values_array.append(float(row[7]))
            read_index += 1
        f.close()

    def calculateLinearCharge(self,x,y,z):
        N = x.shape[0]

        q_line = np.array([-3.3,-3.2,-3.1,-3.0,-2.9,-2.8,-2.7,-2.6,-2.5,-2.4,-2.3,-2.2])*1.0e-15
        x_line = np.array([0.0,0.02,0.05,0.1,0.2,0.3,0.4,0.5])
        z_line = np.array([-1.5,-1.4,-1.3,-1.2,-1.1,-1.05,-1.02,-1.0,-0.98,-0.96,-0.94,-0.92,-0.9,-0.8,-0.6,-0.4,-0.2,0.0,0.5,1.0]) + 1.0

        points = (q_line, x_line, z_line) 


        n_q = q_line.shape[0]
        n_x = x_line.shape[0]
        n_z = z_line.shape[0]
        values = np.zeros((n_q,n_x,n_z))
        
        index = 0
        for i in range(n_q):
            for j in range(n_x):
                for k in range(n_z):
                    if x_line[j] != 0.0 or z_line[k] != 0.0:
                        values[i][j][k] = self.values_array[index]
                        index += 1
        for i in range(n_q):
            for j in range(n_x):
                for k in range(n_z):
                    if x_line[j] == 0.0 and z_line[k] == 0.0:
                        values[i][j][k] = values[i][j+1][k]
       
        points = (q_line, x_line, z_line)
        
        q = np.ones(N)*-3.09e-15
        q_iter = np.ones(N)*-3.09e-15
        n = 20
        
        for k in range(n):
            for i in range(N):
                q[i] = -3.09e-15
                for j in range(N):
                    if i != j:
                        r = np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2)
                        h = z[j]-z[i]
                        
                        if r <= 0.5 and h <= 2.0 and h >= -0.5:
                            point = np.array([q_iter[j],r,h])
                            f = interpn(points, values, point, method = "linear", bounds_error=False, fill_value=None)[0]
                        else:
                            f = -3.09e-15
                        q[i] += (f+3.09e-15)
            for i in range(N):
                q_iter[i] = q[i]
                
        
        return q
    
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    N = 7
    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)

    r_D_e = 0.0016622715189364113


