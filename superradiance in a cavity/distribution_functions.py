import numpy as np
import scipy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from decimal import Decimal
from math import floor
from matplotlib import cm, colors
from scipy.special import sph_harm
from scipy.special import factorial
from sympy.physics.wigner import wigner_3j
from matplotlib import colors
from math import factorial
# from qutip import *


class Atomic_state_on_bloch_sphere:
    def __init__(self, N, k=0):
        pass

    def Wigner_BlochSphere(self, Npoints, N, psi, rho, s_type, bar=True):
        '''

        :param Npoints:
        :param N:
        :param psi:
        :param rho:
        :param s_type:
        :return:
        '''
        resolution = Npoints
        phi = np.linspace(0, 2 * np.pi, resolution)
        theta = np.linspace(0, np.pi, resolution)
        theta, phi = np.meshgrid(theta, phi)
        X = np.sin(theta) * np.cos(phi)
        Y = np.sin(theta) * np.sin(phi)
        Z = np.cos(theta)

        W = np.zeros(np.shape(phi))
        j = N / 2
        for k in np.linspace(0, 2 * j, floor(2 * j + 1)):
            for q in np.linspace(-k, k, floor(2 * k + 1)):
                if q >= 0:
                    Ykq = sph_harm(q, k, phi, theta)
                else:
                    Ykq = (-1) ** q * np.conj(sph_harm(-q, k, phi, theta))
                Gkq = 0
                for m1 in np.linspace(-j, j, floor(2 * j + 1)):
                    for m2 in np.linspace(-j, j, floor(2 * j + 1)):
                        if -m1 + m2 + q == 0:
                            if s_type == 'psi':
                                tracem1m2 = np.conj(psi[floor(m1 + j)]) * psi[floor(m2 + j)]
                            elif s_type == 'rho':
                                tracem1m2 = rho[floor(m1 + j), floor(m2 + j)]
                            else:
                                print('Invalid statetype')
                            Gkq = Gkq + tracem1m2 * np.sqrt(2 * k + 1) * (-1) ** (j - m1) * np.conj(
                                np.complex(wigner_3j(j, k, j, -m1, q, m2)))
                W = W + Ykq * Gkq
        if np.max(abs(np.imag(W))) > 1e-3:
            print('The wigner function has non negligible imaginary part ', str(np.max(abs(np.imag(W)))))
        W = np.real(W)

        fmax, fmin = W.max(), W.min()

        fcolors = W / np.max(np.abs(W))
        # Set the aspect ratio to 1 so our sphere looks spherical
        fig = plt.figure(figsize=(6, 6))
        # fig.tight_layout()
        ls = LightSource(azdeg=-90, altdeg=0)

        ax = fig.add_subplot(111, projection='3d')

        a = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm.bwr(fcolors / 2 + 0.5))
        m = cm.ScalarMappable(cmap=cm.bwr)
        m.set_array(fcolors)



            # h = plt.colorbar(m, orientation="vertical", pad=0.1)
            # h.ax.tick_params(labelsize=30)
            # h.set_label('$W(\\theta,\phi)$', fontsize=16)
        # plt.show()
        # np.max(np.abs(W)),np.max(np.abs(W)))
        # Turn off the axis planes

        ax.set_axis_off()
        ax.quiver(0, -1, 0, 0, -0.5, 0, alpha=1, lw=3, color='black')  # z arrow
        ax.quiver(1, 0, 0, 0.5, 0, 0, alpha=1, lw=3, color='black')  # z arrow
        ax.quiver(0, 0, 1, 0, 0, 1, alpha=1, lw=3, color='black')  # z arrow
        ax.view_init(45, 245)
        # ax.text(0, 0, 2.4, 'Z', fontsize=20)

        return fig


class Wigner:
    def __init__(self, N_alpha=100, alpha_max=4):
        self.N_alpha = N_alpha
        self.alpha_max = alpha_max

    def zeros_below_diaganol(self, N):
        return np.array([[i >= j for i in range(N)] for j in range(N)], dtype=complex)

    def calc_Wigner_from_moments(self, moments, a_avg=0):
        '''

        :param moments:
        :return:
        '''
        # plt.imshow(np.abs(moments))
        # plt.colorbar()
        # plt.show()
        NUM_OF_MOMENTS = np.size(moments, 1)
        mask = self.zeros_below_diaganol(NUM_OF_MOMENTS)
        mask_eye = np.logical_not(np.eye(NUM_OF_MOMENTS, dtype=bool))
        x = np.linspace(-self.alpha_max, self.alpha_max, self.N_alpha)
        p = np.linspace(-self.alpha_max, self.alpha_max, self.N_alpha)
        X, P = np.meshgrid(x, p)
        Wigner = np.zeros([self.N_alpha, self.N_alpha], dtype=complex)
        alpha = X - np.real(a_avg) + 1j * (P - np.imag(a_avg))

        for n in range(self.N_alpha):
            for m in range(self.N_alpha):
                ξ = self.calc_ξ(alpha[n, m], NUM_OF_MOMENTS)
                Wmn = ξ * moments.T * mask
                Wmn[mask_eye] = 2 * np.real(Wmn[mask_eye])
                Wigner[n, m] = np.sum(Wmn)

        Wigner = np.real(Wigner)
        dx = X[1, 1] - X[1, 0]
        print(np.sum(np.sum(Wigner * (X ** 2 + P ** 2 - 1 / 2)) * dx ** 2))
        return X, P, Wigner

    def calc_ξ(self, alpha, NUM_OF_MOMENTS):
        '''

        :param alpha:
        :param NUM_OF_MOMENTS:
        :return:
        '''

        ξ = np.zeros([NUM_OF_MOMENTS, NUM_OF_MOMENTS], dtype=np.complex)
        ξ[0, :] = np.array([2 / np.pi * np.exp(-2 * np.abs(alpha) ** 2) * 2 ** m * alpha ** m / factorial(m) for m in
                            range(NUM_OF_MOMENTS)], dtype=np.complex)
        ξ[1, 1:] = np.array([2 / np.pi * np.exp(-2 * np.abs(alpha) ** 2) * 2 ** m * alpha ** (m - 1) / \
                             factorial(m) * (2 * np.abs(alpha) ** 2 - m) for m in range(1, NUM_OF_MOMENTS)],
                            dtype=np.complex)

        m = np.linspace(0, NUM_OF_MOMENTS - 1, NUM_OF_MOMENTS)

        for l in range(2, NUM_OF_MOMENTS):
            ξ[l, :] = -1 / (alpha * l) * (2 * np.conj(alpha) * ξ[l - 2, :] + \
                                          (m - l + 1 - 2 * np.abs(alpha) ** 2) * ξ[l - 1, :])
        return ξ
