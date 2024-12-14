import numpy as np
from scipy.optimize import curve_fit

def f(q, drag_0, drag_1, drag_2):
    return drag_0 + drag_1*q + drag_2*q**2

def fit(q, drag):
    q_coeff = 1.0e-15
    drag_coeff = 1.0e-13

    drag = drag / drag_coeff
    q = q / q_coeff

    popt, pcov = curve_fit(f, q, drag)
    drag_0 = popt[0]*drag_coeff
    drag_1 = popt[1]*drag_coeff/q_coeff
    drag_2 = popt[2]*drag_coeff/q_coeff**2
    return drag_0, drag_1, drag_2

if __name__ == "__main__":
    q = np.array([-3.3e-15, -3.2e-15, -3.1e-15, -3.0e-15, -2.9e-15, -2.8e-15, -2.7e-15, -2.6e-15, -2.5e-15, -2.4e-15, -2.3e-15, -2.2e-15])
    drag = np.array([6.57535486e-13, 6.68645719e-13, 5.94998558e-13, 5.61995700e-13, 5.72045910e-13, 5.04119853e-13, 4.95714651e-13, 4.64478871e-13, 4.30360214e-13, 4.16878633e-13, 3.71988939e-13, 3.16749761e-13])
    
    drag_0, drag_1, drag_2 = fit(q, drag)

    for i in range(len(q)):
        print("{}\t{}\t{}".format(q[i], drag[i], f(q[i], drag_0, drag_1, drag_2)))

    print(drag_0, drag_1, drag_2)
    
