import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def approx(x, R1, R2, Delta_phi, z1, z2, f):
    R1_diff = R1 - R1.mean()
    R2_diff = R2 - R2.mean()
    Delta_phi_diff = Delta_phi - Delta_phi.mean()
    z1_diff = z1 - z1.mean()
    z2_diff = z2 - z2.mean()
    f_diff = f - f.mean()
    return 1e12 * ( (x[0] * R1_diff + x[1] * R2_diff + x[2] * Delta_phi_diff + x[3] * z1_diff + x[4] * z2_diff) - f_diff )


def calc_spectrum(f, dx):
    spectrum = np.abs(np.fft.rfft(f))
    freqs = np.fft.rfftfreq(f.shape[0], d=dx)
    return freqs, spectrum


start_step = 2000

###################################
# Calculation of averaged spectra #
###################################

spectra_outer = []
for iter_no in range(1, 51):
    data = np.loadtxt(f"limit_cycle_{iter_no}/data_limit.txt")
    # 0:iteration 
    # 1:x1, 2:y1, 3:z1, 4:v_x1, 5:v_y1, 6:v_z1, 7:q1, 8:a_x1*m_p-f_therm_p_x1, 9:a_z1*m_p-f_therm_p_z1, 10:f_tau12, 11:f_norm12, 12:f_zet12
    # 13:x2, 14:y2, 15:z2, 16:v_x2, 17:v_y2, 18:v_z2, 19:q2, 20:a_x2*m_p-f_therm_p_x2, 21:a_z2*m_p-f_therm_p_z2, 22:f_tau21, 23:f_norm21, 24:f_zet21

    t = data[:, 0]

    x1 = data[:, 1]
    x2 = data[:, 13]
    y1 = data[:, 2]
    y2 = data[:, 14]
    z1 = data[:, 3]
    z2 = data[:, 15]

    vx1 = data[:, 4]
    vy1 = data[:, 5]
    vz1 = data[:, 6]
    vx2 = data[:, 16]
    vy2 = data[:, 17]
    vz2 = data[:, 18]

    f_tau_12 = data[:, 10]
    f_tau_21 = data[:, 22]
    f_norm_12 = data[:, 11]
    f_norm_21 = data[:, 23]
    f_zet_12 = data[:, 12]
    f_zet_21 = data[:, 24]

    R1 = np.sqrt(x1**2 + y1**2)
    R2 = np.sqrt(x2**2 + y2**2)
    Delta_phi = np.arccos((x1 * x2 + y1 * y2) / (R1 * R2))
    V1 = np.sqrt(vx1**2 + vy1**2)
    V2 = np.sqrt(vx2**2 + vy2**2)
    omega1 = V1 / R1
    omega2 = V2 / R2

    f_list = [R1, R2, Delta_phi, z1, z2, omega1, omega2]
    f_labels = ["$R_1$, m", "$R_2$, cm", "$\Delta \phi$, rad", "$z_1$, cm", "$z_2$, cm", "$\omega_1$, rad/s", "$\omega_2$, rad/s"]

    spectra_inner = [[], []]
    for i in range(len(f_list)):
        a, b = calc_spectrum(f_list[i][start_step:] - f_list[i][start_step:].mean(), t[1]-t[0])
        spectra_inner[0].append(a)
        spectra_inner[1].append(b)
    spectra_outer.append(spectra_inner)
spectra_outer = np.array(spectra_outer)
spectra_outer = np.mean(spectra_outer, axis=0)

fig, axs = plt.subplots(nrows=len(f_list), layout="tight", figsize=(6, 8))
for i in range(len(f_list)):
    a, b = spectra_outer[0, i], spectra_outer[1, i]
    axs[i].plot(a, b)
    axs[i].set_ylabel(f"Spectrum of\n{f_labels[i]}")
    axs[i].set_xlim(0, 40)
    axs[i].xaxis.set_minor_locator(MultipleLocator(1))
    axs[i].xaxis.set_major_locator(MultipleLocator(10))
axs[-1].set_xlabel("Frequency, Hz")
plt.savefig("signal_spectra_2d")
plt.show()


with open("limit_cycle/spectra.txt", "w") as fout:
    output_string = "\t".join(f_labels)
    fout.write(output_string+"\n")

    for j in range(spectra_outer.shape[2] // 5):
        output_string = f"{spectra_outer[0, 0, j]}"
        for i in range(spectra_outer.shape[1]):
            output_string += f"\t{spectra_outer[1, i, j]}"
        fout.write(output_string+"\n")


##############################
# Calculation of derivatives #
##############################

data = np.loadtxt(f"limit_cycle/data_limit.txt")
# 0:iteration 
# 1:x1, 2:y1, 3:z1, 4:v_x1, 5:v_y1, 6:v_z1, 7:q1, 8:a_x1*m_p-f_therm_p_x1, 9:a_z1*m_p-f_therm_p_z1, 10:f_tau12, 11:f_norm12, 12:f_zet12
# 13:x2, 14:y2, 15:z2, 16:v_x2, 17:v_y2, 18:v_z2, 19:q2, 20:a_x2*m_p-f_therm_p_x2, 21:a_z2*m_p-f_therm_p_z2, 22:f_tau21, 23:f_norm21, 24:f_zet21

t = data[:, 0]

x1 = data[:, 1]
x2 = data[:, 13]
y1 = data[:, 2]
y2 = data[:, 14]
z1 = data[:, 3]
z2 = data[:, 15]

vx1 = data[:, 4]
vy1 = data[:, 5]
vz1 = data[:, 6]
vx2 = data[:, 16]
vy2 = data[:, 17]
vz2 = data[:, 18]

f_tau_12 = data[:, 10]
f_tau_21 = data[:, 22]
f_norm_12 = data[:, 11]
f_norm_21 = data[:, 23]
f_zet_12 = data[:, 12]
f_zet_21 = data[:, 24]

R1 = np.sqrt(x1**2 + y1**2)
R2 = np.sqrt(x2**2 + y2**2)
Delta_phi = np.arccos((x1 * x2 + y1 * y2) / (R1 * R2))
V1 = np.sqrt(vx1**2 + vy1**2)
V2 = np.sqrt(vx2**2 + vy2**2)
omega1 = V1 / R1
omega2 = V2 / R2


prop_list = [f_tau_12, f_norm_12, f_zet_12, f_tau_21, f_norm_21, f_zet_21, R1, R2, Delta_phi]
prop_labels = ["f_tau_12", "f_norm_12", "f_zet_12", "f_tau_21", "f_norm_21", "f_zet_21", "R1", "R2", "Delta_phi"]
for prop, prop_label in zip(prop_list, prop_labels):
    print(f"<{prop_label}> = {prop[start_step:].mean()} +/- {prop[start_step:].std()/np.sqrt(prop[start_step:].shape[0])}")

x0 = np.array([1, 1, 1, 1, 1])
f_labels = ["f_tau_12", "f_tau_21", "f_norm_12", "f_norm_21", "f_zet_12", "f_zet_21"]
arg_labels = ["R1", "R2", "Delta_phi", "z1", "z2"]
f_list = [f_tau_12, f_tau_21, f_norm_12, f_norm_21, f_zet_12, f_zet_21]

for f, f_label in zip(f_list, f_labels):
    print(f">>>>> {f_label}")
    print(f"<{f_label}> / <R1> =", f[start_step:].mean() / R1[start_step:].mean())
    print(f"<{f_label}> / <R2> =", f[start_step:].mean() / R2[start_step:].mean())

    res_f = least_squares(
        lambda x: approx(x, R1[start_step:], R2[start_step:], Delta_phi[start_step:], z1[start_step:], z2[start_step:], f[start_step:]),
        x0
    )
    dftau12_dR1, dftau12_dR2, dftau12_dDeltaPhi, dftau12_dz1, dftau12_dz2 = res_f.x
    for arg, arg_label in zip(res_f.x, arg_labels):
        print(f"d{f_label} / d{arg_label} =", arg)

