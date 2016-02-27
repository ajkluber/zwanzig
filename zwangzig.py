import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyemma.msm

import scipy.linalg
from scipy.special import erf

"""
The purpose of this script is to determine how the timescales change as a
function of 'landscape roughness'. Landscape roughness is variations in
potential energy that are small compared to the depth of the global minimum.

The roughness has two parameters:
    - magnitude
    - spatial correlation length

Zwanzig's theory indicates that if you can spatially average over the roughness
then the effective diffusion coefficient goes as:

    Deff = D0*exp(-(dE/kT)^2)

where 
    dE = energetic scale of roughness
    D0 = diffusion coefficient on equivalent smooth landscape
"""

def omega(V, D, dx, i, j):
    """Calculate matrix element of symmetrized rate matrix"""
    return 0.5*((D[i] + D[j])/(dx**2))*np.exp(-0.5*(V[j] - V[i]))

def dynamical_timescales(F, D, dx, keep=10):
    """Get timescales from rate matrix"""
    n_bins = F.shape[0]
    M = np.zeros((n_bins, n_bins))
    for i in range(1,n_bins - 1):
        M[i,i] = -(omega(F,D,dx,i,i + 1) + omega(F,D,dx,i,i - 1))
        M[i,i + 1] = np.sqrt(omega(F,D,dx,i,i + 1)*omega(F,D,dx,i + 1,i))
        M[i + 1,i] = np.sqrt(omega(F,D,dx,i + 1,i)*omega(F,D,dx,i,i + 1))
        M[i,i - 1] = np.sqrt(omega(F,D,dx,i,i - 1)*omega(F,D,dx,i - 1,i))
        M[i - 1,i] = np.sqrt(omega(F,D,dx,i - 1,i)*omega(F,D,dx,i,i - 1))

    M[0,0] = -omega(F,D,dx,0,1)
    M[n_bins - 1,n_bins - 1] = -omega(F,D,dx,n_bins - 1,n_bins - 2)

    vals = scipy.linalg.eigvals(M)
    vals = vals[np.argsort(vals)[::-1]]
    return -1./np.real(vals[1:keep + 1])


def transition_matrix_metropolis(V_arr,P,v=0.5):
    # fill it
    for i in range(nbins):
        if i == 0:
            P[i,i+1] = v*min(1.0,np.exp(-(V_arr[i+1]-V_arr[i])))
            P[i,i] = 1.0 - P[i,i+1]
        elif i == nbins-1:
            P[i,i-1] = v*min(1.0,np.exp(-(V_arr[i-1]-V_arr[i])))
            P[i,i] = 1.0 - P[i,i-1]
        else:
            P[i,i-1] = v*min(1.0,np.exp(-(V_arr[i-1]-V_arr[i])))
            P[i,i+1] = v*min(1.0,np.exp(-(V_arr[i+1]-V_arr[i])))
            P[i,i] = 1.0 - P[i,i-1] - P[i,i+1]


def get_msm_timescales(Varr, D, dx, keep=10):
    """Make MSM for potential"""
    P = transition_matrix(Varr, D, dx)
    msm = pyemma.msm.markov_model(P)
    return msm.timescales(keep) 

def get_rand_potential(dE, x, nmodes=100, sigma=50):
    """Create a random potential"""
    rand_modes = np.random.normal(loc=0, scale=1, size=nmodes)
    rand_phase = np.random.uniform(low=0, high=2*np.pi, size=nmodes)

    Vrand = np.zeros(len(x))
    for i in range(nmodes):
        Vrand += dE*np.sqrt(2./nmodes)*np.cos(rand_modes[i]*(sigma*x) + rand_phase[i])
    return Vrand

if __name__ == "__main__":
    dE = 0.1
    xmin = -4
    xmax = 4
    nbins = 1000
    nsamples = 20
    ndEs = 20
    keep = 10
    kT = 1.
    b = 2
    D0 = 1.
    #dEs = np.logspace(-2, 0.3, 10)
    dEs = np.linspace(0.01, 1, ndEs)

    saveplots = False

    nmodes = 200 # Number of random Fourier modes
    sigma = 10. # sets length scale of correlations

    # Needs to be called each time a new potential is created (?)
    x = np.linspace(xmin, xmax, num=nbins)
    dx = x[1] - x[0]
    Vsmooth = 0.5*b*x**2
    D = D0*np.ones(nbins, float)
    
    # True timescales for smooth harmonic oscillator
    true_smooth_timescales = kT/(b*D0*np.arange(1, keep + 1))
    zwanzig_rough_timescales = np.array([ np.exp(idE**2)*kT/(b*D0*np.arange(1, keep + 1)) for idE in dEs ])
    bagchi_rough_timescales = np.array([ (np.exp(idE**2)/(1. + erf(idE/2)))*kT/(b*D0*np.arange(1, keep + 1)) for idE in dEs ])

    # The reference, smooth potential
    smooth_timescales = dynamical_timescales(Vsmooth, D, dx)

    ratios = np.zeros((ndEs, keep))
    all_Vrands = np.zeros((ndEs,nbins), float)
    for n in range(ndEs):
        dE = dEs[n]
        c_val = float(n)/ndEs
        print dE
        Vrands = np.zeros((nsamples, nbins))
        rough_times = np.zeros((nsamples, keep))
        for i in range(nsamples):
            Vrand = get_rand_potential(dE, x)
            rough_timescales = dynamical_timescales(Vsmooth + Vrand, D, dx)
            Vrands[i,:] = Vrand + Vsmooth
            rough_times[i,:] = rough_timescales

        avg_rough_times = np.mean(rough_times, axis=0)
        std_rough_times = np.std(rough_times, axis=0)

        ratios[n,:] = avg_rough_times/smooth_timescales

        all_Vrands[n,:] = Vrand + Vsmooth

        plt.figure(1)
        if n == 0:
            plt.plot(avg_rough_times, c=cm.spectral(c_val),label="matrix")
            plt.plot(zwanzig_rough_timescales[n,:], c=cm.spectral(c_val), ls='-.',label="zwanzig")
            #plt.plot(bagchi_rough_timescales[n,:], c=cm.spectral(c_val), ls='--',label="bagchi")
        else:
            plt.plot(avg_rough_times, c=cm.spectral(c_val))
            plt.plot(zwanzig_rough_timescales[n,:], c=cm.spectral(c_val), ls='-.')
            #plt.plot(bagchi_rough_timescales[n,:], c=cm.spectral(c_val), ls='--')

        plt.figure(2)
        plt.plot(avg_rough_times/smooth_timescales, c=cm.spectral(c_val))

    if not os.path.exist("harmonic_plots"):
        os.mkdir("harmonic_plots")
    os.chdir("harmonic_plots")

    fig_temp = plt.figure(2)
    plt.xlabel("index")
    plt.ylabel("Ratio of timescales $\\frac{t_i}{t_i^0}$")
    fig_temp.savefig("ti_norm_vs_index.png",bbox_inches="tight")
    fig_temp.savefig("ti_norm_vs_index.pdf",bbox_inches="tight")

    fig1 = plt.figure(1)
    plt.plot(smooth_timescales, 'k')
    plt.plot(true_smooth_timescales, 'k', ls='--')
    plt.ylabel("Implied timescales")
    plt.legend()
    fig1.savefig("ti_vs_index.png", bbox_inches="tight")
    fig1.savefig("ti_vs_index.pdf", bbox_inches="tight")

    fig2 = plt.figure()
    plt.plot(dEs, ratios, 'o', color="#5DA5DA")
    plt.plot(dEs, np.exp(dEs**2), c='k', label="Zwanzig") 
    plt.xlabel("Roughness $dE$")
    plt.ylabel("Ratio of timescales $\\frac{t_i}{t_i^0}$")
    fig2.savefig("ti_norm_vs_dE.png", bbox_inches="tight")
    fig2.savefig("ti_norm_vs_dE.pdf", bbox_inches="tight")

    # Plot random potentials and comparison with smoothed potential
    fig3 = plt.figure()
    for i in range(ndEs)[::-1][::4]:
        c_val = float(i)/ndEs
        plt.plot(x, all_Vrands[i,:], c=cm.spectral(c_val), label=str(dEs[i]))
    plt.plot(x, Vsmooth, '--', color="gray", lw=2, label="smooth")
    plt.xlabel("x")
    plt.ylabel("V(x)")
    plt.title("Rough harmonic potential")
    plt.legend()
    fig3.savefig("Vrand.png", bbox_inches="tight")
    fig3.savefig("Vrand.pdf", bbox_inches="tight")

    os.chdir("..")

    plt.show()
