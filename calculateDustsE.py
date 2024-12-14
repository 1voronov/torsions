import numpy as np

def calculateDustsE(x,y,z,q):
    eps_0 = 8.85418781762039e-12 # vacuum dielectric permitivity
    k = 1.0 / (eps_0 * 4.0 * np.pi)
    r_D_e = 0.0016622715189364113

    N = x.shape[0]
    E_x = np.zeros(N)
    E_y = np.zeros(N)
    E_z = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if i != j:
                r_ij = np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2+(z[i]-z[j])**2)
                E_x[i] += (x[i]-x[j])*k*q[j]/r_ij**3
                E_y[i] += (y[i]-y[j])*k*q[j]/r_ij**3
                E_z[i] += (z[i]-z[j])*k*q[j]/r_ij**3
    return E_x/r_D_e**2, E_y/r_D_e**2, E_z/r_D_e**2

