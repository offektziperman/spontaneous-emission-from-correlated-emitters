from distribution_functions import Atomic_state_on_bloch_sphere
import numpy as np
import matplotlib.pyplot as plt
from distribution_functions import Wigner
from Operators import Operators
# from qutip.wigner import _wigner_laguerre

# from qutip import displace, basis, ket2dm, fidelity, squeeze
from scipy.linalg import expm
import scipy

NUM_MOMENTS = 11
N = 40
# for k in range(1):
#     rho = np.zeros([N + 1, N + 1])
#
#     rho[k, k] = 1
#     op = Operators(N)
#
#
#     rho = scipy.linalg.expm(1j * np.pi * Sy) @ rho@scipy.linalg.expm(-1j * np.pi * np.conj(Sy.T))
#

#     ax2 = fig.add_subplot(122)
#     ax2.set_axis_off()
#     w = Wigner(200,alpha_max=3)
#     a, a_dagger = op.create_a_and_a_dagger(N)
#     light_rho = np.zeros([N + 1, N + 1])
#     light_rho[0, 0] = 1
#     light_rho = scipy.linalg.expm(1j * np.pi / 2 * Sy) @ light_rho@scipy.linalg.expm(-1j * np.pi / 2 * np.conj(Sy.T))
#     # m, n = np.meshgrid(np.linspace(0, NUM_MOMENTS, NUM_MOMENTS + 1), np.linspace(0, NUM_MOMENTS, NUM_MOMENTS + 1))
#     moments = op.operator_moments(NUM_MOMENTS, light_rho, a, a_dagger)
#
#     x, p, wigner = w.calc_Wigner_from_moments(moments)
#     print(type(wigner))
#     alphas = np.abs(wigner)
#     ax2.pcolormesh(x,p,wigner,alpha=alphas/np.max(alphas), cmap='bwr')
#     m = ax2.collections[0]
#
#     # ax2.set_xlabel('$q$', fontsize=16)
#     # ax2.set_ylabel('$p$', fontsize=16)
#     ax2.set_aspect('equal')
#     # g = plt.colorbar(m, shrink=0.5)
#     # g.set_label('$W(q,p)$', fontsize=16)
#     m.set_clim(-np.max(np.abs(wigner)), np.max(np.abs(wigner)))
#
#     fig.savefig('temp'+str(k)+'.png', transparent=True)
#     plt.clf()
f = plt.figure()
ax2 = f.add_subplot()
NUM_STEPS = 100
x = np.linspace(-4, 4, NUM_STEPS)
y = np.linspace(-4, 4, NUM_STEPS)


X, Y = np.meshgrid(x, y)
alpha = 5
Z = 1 / (np.pi * (1 + np.exp(-2 * (alpha ** 2)))) * (np.exp(-2 * (X - alpha) ** 2 - 2 * Y ** 2) + \
                                                     np.exp(-2 * (X + alpha) ** 2 - 2 * Y ** 2) - 2 * np.exp(
            -2 * X ** 2 - 2 * Y ** 2) * np.cos(4 * Y * alpha))
f = plt.figure()
psi = np.zeros(N + 1)
psi[0] = 1
op = Operators(N)
x = np.linspace(-4, 4, 1000)
t_0 = 0.3
ax2 = f.add_subplot(111)
Sx, Sy, Sz = op.create_spin_matrices(N)

a, a_dagger = op.create_a_and_a_dagger(N)


# psi = expm(1j * Sx @ Sx * 0.1) @ psi
# psi = expm(1j * Sy * 0.3) @ psi + expm(-1j * Sy * 0.3) @ psi
# psi = psi/np.linalg.norm(psi)
# psi = expm(1j * Sy * 0.3) @ psi + expm(-1j * Sy * 0.3) @ psi
# psi = psi/np.linalg.norm(psi)
# psi = expm(1j * Sy * 0.3) @ psi + expm(-1j * Sy * 0.3) @ psi
# psi = psi/np.linalg.norm(psi)
atomic_wigner = Atomic_state_on_bloch_sphere(N)
# plt.pcolormesh(np.abs(rho))
fig = atomic_wigner.Wigner_BlochSphere(200, np.size(psi, 0) - 1, psi, [], 'psi', bar=False)
plt.show()
# rho = expm(1j * a_dagger @ a * t_0) @ rho @ expm(-1j * a_dagger @ a * t_0)

# # w = _wigner_laguerre(rho, x, x, 1, 1)
#
# ax2.pcolormesh(x, x, w, cmap='bwr')
# m = ax2.collections[0]
# ax2.set_aspect('equal')
# m.set_clim(-np.max(np.abs(w)), np.max(np.abs(w)))
# plt.xticks([])
# plt.yticks([])

# plt.xlim([-4, 4])
# plt.ylim([-2, 2])
#
# plt.show()
#
# g = plt.figure()
# ax2 = g.add_subplot()
#
# n = 8
# Hn = scipy.special.hermite(n)
# H0 = scipy.special.hermite(0)
# Ln = scipy.special.laguerre(n)
# s_1 = np.linspace(-4, 4, NUM_STEPS)
# s_2 = np.linspace(-4, 4, NUM_STEPS)
# S1, S2 = np.meshgrid(s_1, s_2)
# dx = s_1[1] - s_1[0]
# q1 = np.linspace(-4, 4, NUM_STEPS)
# q2 = np.linspace(-4, 4, NUM_STEPS)
# Z = np.zeros([NUM_STEPS, NUM_STEPS])
# Q1, Q2 = np.meshgrid(q1, q2)
# f2 = plt.figure()
# # for i in range(NUM_STEPS):
# #     print(i)
# #     for j in range(NUM_STEPS):
# #         Z[i,j] = np.sum(np.exp(-q1[i]**2/2 - S1**2/8) *np.exp(-q2[j]**2/2 - S2**2/8)* \
# #                         (Hn(q1[i] - S1 / 2)*Hn(q2[j] + S2 / 2) + Hn(q1[i]-S1/2)*Hn(q2[j]+S2/2))) +\
# #             np.exp(-(q1[i]**2))*Ln(2*q1[i]**2)*np.exp(-(q2[j]**2)) + np.exp(-(q2[j]**2))*Ln(2*q2[j]**2)*np.exp(-(q1[i]**2))*dx**2
# Z1 = np.exp(-(Q1 ** 2 + Q2 ** 2)) * Ln(2 * Q1 ** 2 + 2 * Q2 ** 2) * (-1) ** n / np.pi
# fig, ((ax1, ax2, ax3, ax4, ax5, ax6)) = plt.subplots(1, 6)
# print(np.max(Z))
# ax1.pcolormesh(Q1, Q2, Z1, cmap='bwr')
# m = ax1.collections[0]
# ax1.set_aspect('equal')
# m.set_clim(-np.max(np.abs(Z1)) / 2, np.max(np.abs(Z1)) / 2)
# Z2 = np.exp(-(Q1 ** 2 + Q2 ** 2)) / np.pi
# ax5.pcolormesh(Q1, Q2, Z2, cmap='bwr')
#
# m = ax5.collections[0]
# ax5.set_aspect('equal')
# m.set_clim(-np.max(np.abs(Z2)) / 2, np.max(np.abs(Z2)) / 2)
#
# ax6.pcolormesh(Q1, Q2, Z1, cmap='bwr')
# m = ax6.collections[0]
# ax6.set_aspect('equal')
# m.set_clim(-np.max(np.abs(Z1)) / 2, np.max(np.abs(Z1)) / 2)
# Z2 = np.exp(-(Q1 ** 2 + Q2 ** 2)) / np.pi
# ax2.pcolormesh(Q1, Q2, Z2, cmap='bwr')
#
# m = ax2.collections[0]
# ax2.set_aspect('equal')
# m.set_clim(-np.max(np.abs(Z2)) / 2, np.max(np.abs(Z2)) / 2)
# ax1.tick_params(top=False, bottom=False, left=False, right=False,
#                 labelleft=False, labelbottom=False)
# ax2.tick_params(top=False, bottom=False, left=False, right=False,
#                 labelleft=False, labelbottom=False)
# ax5.tick_params(top=False, bottom=False, left=False, right=False,
#                 labelleft=False, labelbottom=False)
# ax6.tick_params(top=False, bottom=False, left=False, right=False,
#                 labelleft=False, labelbottom=False)
# N = 20
#
# ax3.axis('off')
# ax4.axis('off')
# plt.show()
