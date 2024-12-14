import numpy as np
from scipy.optimize import curve_fit

def E(z, E_0, alpha):
    return E_0 + alpha*z

def fit(z, E_z):
    r_D_e = 0.0016622715189364113
    E_z_0 = E_z[0]

    E_z = E_z /  E_z_0
    z = z / r_D_e

    popt, pcov = curve_fit(E, z, E_z)
    E_0 = popt[0]*E_z_0
    alpha = popt[1]*E_z_0/r_D_e
    return E_0, alpha

if __name__ == "__main__":
    r_D_e = 0.0016622715189364113
    z = np.array([-1.94*r_D_e, -1.65*r_D_e, -1.37*r_D_e, -1.13*r_D_e, -0.89*r_D_e, -0.62*r_D_e, -0.33*r_D_e])
    E_z = np.array([2251, 2431, 2386, 2528, 2597, 2539, 2607])
    E_0, alpha, beta = fit(z, E_z)

    for i in range(len(z)):
        print("{}\t{}".format(z[i], E_z[i]))

    print(E_0, alpha, beta)
    
