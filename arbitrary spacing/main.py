import qutip
import matplotlib.pyplot as plt
import matplotlib
from qutip import *
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq
import numpy as np

OMEGA_0 = 0  # atomic transition frequency
RES = 200   # Number of sample points
GAMMA = 1
GAMMA_s = 0.
Tf = 4  # GAMMA ** -1 * 30
cmap = matplotlib.cm.bwr


def coupling1_coef(t, args):
    coupling_eta = args['coupling_eta']
    times = args['times']
    index = np.where(np.abs(times - t) == np.min(np.abs(t - times)))[0]
    return coupling_eta[index]


def coupling1_coef_conj(t, args):
    return np.conj(coupling1_coef(t, args))


def coupling2_coef(t, args):
    coupling_nu = args['coupling_nu']
    times = args['times']
    index = np.where(np.abs(times - t) == np.min(np.abs(t - times)))[0]
    return coupling_nu[index]


def coupling2_coef_conj(t, args):
    return np.conj(coupling2_coef(t, args))


def H2_coef(t, args):
    return np.conj(coupling1_coef(t, args)) * coupling2_coef(t, args)


def H2_coef_conj(t, args):
    return np.conj(H2_coef(t, args))


def create_spin_operators(phase_factor, num_emitters):
    '''
    Create collective spin operators considoring locations accross a 1d waveguide - easy to generalize for 3d if needed
    :param phase_factor: relative phases of the different atoms phi_n = 2 pi z_n / lambda
    :param num_emitters: number of atoms
    :return: Collective spin operators as Qobjects
    '''

    sig_z, sig_p, sig_m = qutip.jmat(1 / 2, 'z'), qutip.jmat(1 / 2, '+'), qutip.jmat(
        1 / 2, '-')
    S_z, S_m, S_p = tensor([qzero(2) for i in range(num_emitters)]), tensor([qzero(2) for i in range(num_emitters)]), \
                    tensor([qzero(2) for i in range(num_emitters)])
    L_lst = []
    for i in range(num_emitters):
        sig_z_n = [qeye(2) for i in range(num_emitters)]
        sig_z_n[i] = sig_z
        S_z += tensor(sig_z_n)

        sig_m_n = [qeye(2) for i in range(num_emitters)]
        sig_m_n[i] = sig_m
        S_m += tensor(sig_m_n) * np.exp(-1j * phase_factor[i])

    S_p = S_m.dag()
    return S_m, S_p, S_z


def create_operators(atom_dim, light_dim, phase_factor, num_modes, num_emitters):
    '''
    Create the operators for the cascaded master equation
    :param atom_dim:
    :param light_dim:
    :param phase_factor:
    :param num_modes:
    :param num_emitters:
    :return: a_eta for the mode in the virtual cavity, sp,m for the collective p(+) or m(-) operator and sz for sum sigma_z
    '''

    S_m, S_p, S_z = create_spin_operators(phase_factor, num_emitters)

    a_eta, sp, sm, sz = tensor(tensor([qeye(atom_dim) for i in range(num_emitters)]), destroy(light_dim)), \
                        tensor(S_p, qeye(light_dim)), \
                        tensor(S_m, qeye(light_dim)), \
                        tensor(S_z, qeye(light_dim))
    return a_eta, sp, sm, sz


def get_coupling_from_mode(mode, times, position=1, coupling_prev=None):
    '''
    get the coupling for the virtual cavity from the wanted mode.
    :param mode:
    :param times:
    :param position:
    :param coupling_prev:
    :return:
    '''

    def dalpha_nu_dt(t, y):
        '''
        for the case of two output modes
        :return:
        '''
        index = np.where(np.abs(times - t) == np.min(np.abs(t - times)))[0]
        return - (coupling_prev * mode)[index] - np.abs(coupling_prev)[index] ** 2 / 2 * y

    dt = times[2] - times[1]
    coupling = np.zeros_like(mode)
    if not np.any(mode):
        return coupling
    # correct the mode due to reflection from first cavity
    if position == 2:
        result = solve_ivp(dalpha_nu_dt, [times[0], times[-1]], [0. + 0.j], method='RK45', t_eval=times)
        alpha_nu = result.y[0]
        mode = (mode.T + np.conj(coupling_prev.T) * alpha_nu).T

    for i in range(len(mode)):
        S = np.sum(np.abs(mode[:i + 1]) ** 2 * dt) ** 0.5
        if S > 0:
            coupling[i] = -np.conj(mode[i]) / S
    return coupling


def get_light_outside(rho_init, times, GAMMA, light_dim, atom_dim, phase_factor, mode_eta, num_emitters):
    '''
    get the density matrix of the light emitted into the virtual cavity eta.
    :param rho_init: initial condition
    :param times: time series
    :param GAMMA: decay rate
    :param light_dim: dimension of virtual cavity
    :param atom_dim: dimension atoms (normally 2 for two levels)
    :param phase_factor: locations of emitters in units of phase
    :param mode_eta: mode of interest for virtual cavity
    :param num_emitters:
    :return: time series and light in the virtual cavity at final time
    '''

    num_modes = 1
    a_eta, sp, sm, sz = create_operators(atom_dim, light_dim, phase_factor, num_modes, num_emitters)

    H1 = 1j / 2 * sm.dag() * a_eta * GAMMA ** 0.5
    H1_dag = -1j / 2 * sm * a_eta.dag() * GAMMA ** 0.5

    OMEGA_0 = 0.
    H0 = OMEGA_0 * sz

    coupling_eta = get_coupling_from_mode(mode_eta, times)

    opts = Options()
    opts.store_states = True

    args = {'coupling_eta': coupling_eta, 'times': times}
    H_lst = [H0, [H1, coupling1_coef], [H1_dag, coupling1_coef_conj]]
    L_lst = [GAMMA ** 0.5 * sm, [a_eta, coupling1_coef_conj]]

    result = mesolve(H_lst, rho_init, times, [L_lst], args=args, options=opts)
    rho = result.states[-1].ptrace([num_emitters])  # partial trace only the light in the virtual cavity at final time

    return times, rho


def plot_modes(times, modes, ax):
    for mode in modes:
        ax.plot(times, np.real(mode), linewidth=2)
    ax.set_xlabel('time $\\Gamma^{-1}$', fontsize=12)
    ax.set_ylabel('$Real(amplitude)$', fontsize=12)
    ax.legend(['mode 1', 'mode 2'], fontsize=12)
    ax.set_title('')


def plot_intensity(times, result, ax):
    ax.plot(times, np.array(result.expect).T)
    # ax.set_xticks(fontsize=10)
    # ax.set_yticks(fontsize=10)
    ax.set_xlabel('time $\\Gamma^{-1}$', fontsize=12)
    ax.set_ylabel('$<S_z>$', fontsize=12)


def get_g1_modes(H, psi_init, times, s_m, s_p, num_modes=4):
    '''
    calculate g1 with quantum regression theorum,
    :param H: hamiltonian
    :param psi_init: initial condition
    :param times:
    :param s_m: collective lowering
    :param s_p: collective raising
    :param num_modes:
    :return:
    '''
    dt = times[1] - times[0]
    corr = correlation_2op_2t(H, psi_init, times, times,
                              [GAMMA ** 0.5 * s_m, GAMMA_s ** 0.5 * s_m], s_p, s_m)
    corr_cor = np.zeros_like(corr)
    for i in range(np.size(corr, 0)):
        corr_cor[i, i + 1:] = corr[i, :-i - 1]
    corr_cor = corr_cor + np.conj(corr_cor.T)
    for i in range(np.size(corr, 0)):
        corr_cor[i, i] = corr[i, 0]

    g1 = np.zeros_like(corr_cor)
    for i in range(len(corr_cor)):
        if np.abs(corr_cor[i, i]) > 0.01:
            g1[i, :] = corr_cor[i, :]

    w, v = np.linalg.eig(corr_cor * GAMMA * dt)  # diagonalize gamma matrix

    modes = []
    occupations = []
    for i in range(4):
        modes.append(v[:, w == np.sort(w)[-i - 1]])
        occupations.append(np.sort(np.abs(w))[-i - 1])

    return modes, occupations


def make_plots(times, modes, rho_eta_mode_a, result, occupations):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.set_figwidth(10)
    fig.set_figheight(7)
    plot_intensity(times, result, ax1)
    plot_modes(times, modes, ax2)
    qutip.plot_wigner(qutip.Qobj(rho_eta_mode_a), cmap=cmap, colorbar=True, fig=fig, ax=ax4)
    ax4.set_title('')

    ax3.bar(np.linspace(1, len(occupations), len(occupations)), np.real(occupations))
    ax3.set_xlabel('Mode',fontsize=12)
    ax3.set_ylabel('Occupation',fontsize=12)
    plt.show()


def simulate_emitters(times, psi_init, num_modes, atom_dim, light_dim,
                      phase_factor, num_emitters,
                      modes=False, plot=False, state_outside=False):
    '''
    Run the main simulation
    :param times: time series
    :param psi_init: initial condition
    :param num_modes: 1 or 2 output modes
    :param atom_dim: levels in each atom.
    :param light_dim: dimension of hilbert space kept in virtual cavity
    :param phase_factor: locations of emitters in units of phase
    :param num_emitters:
    :param modes:
    :param plot: should it plot?
    :param state_outside: should it calculate the state outside?
    :return:
    '''
    sig_z, sig_p, sig_m = qutip.jmat(1 / 2, 'z'), qutip.jmat(1 / 2, '+'), qutip.jmat(1 / 2, '-')
    S_z, S_m, S_p = tensor([qzero(2) for i in range(num_emitters)]), tensor([qzero(2) for i in range(num_emitters)]), \
                    tensor([qzero(2) for i in range(num_emitters)])
    L_lst = []
    for i in range(num_emitters):
        sig_z_n = [qeye(2) for i in range(num_emitters)]
        sig_z_n[i] = sig_z
        S_z += tensor(sig_z_n)

        sig_m_n = [qeye(2) for i in range(num_emitters)]
        sig_m_n[i] = sig_m
        S_m += tensor(sig_m_n) * np.exp(-1j * phase_factor[i])

    S_p = S_m.dag()

    H = 0 * S_m
    opts = qutip.Options(nsteps=15000, atol=1e-13, rtol=1e-13)
    opts.store_states = True

    dt = times[1] - times[0]
    result = qutip.mesolve(H, psi_init, times, [GAMMA ** 0.5 * S_m], [S_z],
                           options=opts)
    purity = np.array([np.trace(result.states[i] * result.states[i]) for i in range(len(times))])

    if not state_outside and not modes:
        return result.expect[0], purity

    modes, occupations = get_g1_modes(H, psi_init, times, S_m, S_p, num_modes=num_modes)

    mode_1 = modes[0]
    mode_2 = modes[1]

    mode_1 = np.array(mode_1 / (np.sum(np.abs(mode_1) ** 2 * dt) ** 0.5))
    mode_2 = np.array(mode_2 / (np.sum(np.abs(mode_2) ** 2 * dt) ** 0.5))

    if not state_outside:
        return occupations

    rho_init = qutip.tensor(psi_init, qutip.basis(light_dim, 0))

    times, rho_eta_mode_a = get_light_outside(rho_init, times, GAMMA, light_dim, atom_dim,
                                              phase_factor,
                                              mode_1, num_emitters)

    if plot:
        make_plots(times, [mode_1,mode_2], rho_eta_mode_a, result,
                   occupations)
    return rho_eta_mode_a


def create_fig5b(times, distances, psi_inits, num_emitters, atom_dim, light_dim):
    emitted_energies = np.zeros([len(times), 100])
    purities = np.zeros([len(times), 100])
    for k in range(len(psi_inits)):
        psi_init = psi_inits[k]
        for i in range(100):
            phase_factor = np.array(np.linspace(1, num_emitters, num_emitters) * distances[i])

            emitted_energies[:, i], purities[:, i] = simulate_emitters(times, psi_init, 1, atom_dim,
                                                                       light_dim,
                                                                       phase_factor, num_emitters,
                                                                       plot=False,
                                                                       state_outside=False, modes=False)
        np.save('emitted_energy' + str(k), emitted_energies)
        np.save('purity' + str(k), purities)
        np.save('distance' + str(k), distances)

def straighten_phase(rho,final_phase):
    '''

    :param rho:
    :param final_phase:
    :return:
    '''
    phi = qutip.phase(N)


def coherent_state(num_emitters, theta):
    phase_factor = np.array(np.linspace(1, num_emitters, num_emitters) *0)
    psi_init = qutip.tensor([basis(2, 1) for i in range(num_emitters)])
    S_m, S_p, S_z = create_spin_operators(phase_factor, num_emitters)
    to_exp = 1j * (S_p + S_m) / 2 * theta
    return (to_exp.expm()) * psi_init


def cat_state(num_emitters, theta):
    phase_factor = np.array(np.linspace(1, num_emitters, num_emitters) *0)
    psi_init = qutip.tensor([basis(2, 1) for i in range(num_emitters)])
    S_m, S_p, S_z = create_spin_operators(phase_factor, num_emitters)
    to_exp = 1j * (S_p + S_m) / 2 * theta
    to_exp2 = -1j * (S_p + S_m) / 2 * theta
    psi_init = (to_exp.expm() + to_exp2.expm()) * psi_init
    return psi_init / psi_init.norm()


def dicke_state(num_emitters, k):
    phase_factor = np.array(np.linspace(1, num_emitters, num_emitters) *0)
    psi_init = qutip.tensor([basis(2, 1) for i in range(num_emitters)])
    S_m, S_p, S_z = create_spin_operators(phase_factor, num_emitters)
    for i in range(k):
        psi_init = S_p * psi_init
        print(i)
    return psi_init / psi_init.norm()


def get_density_matrix_in_virtual_cavity(times, psi_init, param, atom_dim, light_dim, d, num_emitters, plot,
                                          state_outside,
                                          modes):
    phase_factor = np.array(np.linspace(1, num_emitters, num_emitters) * d)
    rho = simulate_emitters(times, psi_init, 1, atom_dim, light_dim, phase_factor, num_emitters, plot=True,
                            state_outside=True, modes=modes)
    return rho


def mod(t, tau):
    if t > 0:
        mod_a = (t / tau - int(t / tau)) * tau
        mod_a = (t / tau - int(t / tau)) * tau
        return mod_a
    return t + tau * (1 + int(-t / tau))


def plot_maps_distance():
    '''
    Figure 5a
    :return:
    '''
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    distance = np.load('Data/distance.npy')
    energy_coherent = np.load('Data/emitted_energy0.npy')
    energy_cat = np.load('Data/emitted_energy1.npy')
    energy_half_excited = np.load('Data/emitted_energy2.npy')
    energy_all_excited = np.load('Data/emitted_energy3.npy')
    times = np.linspace(0, Tf, RES)

    ax1.pcolormesh(times, distance / (2 * np.pi), energy_coherent.T, cmap='inferno', shading='gouraud')
    m = ax1.collections[0]
    m.set_clim(-3, 3)
    times = np.linspace(0, Tf, RES)

    ax2.pcolormesh(times, distance / (2 * np.pi), energy_cat.T, cmap='inferno', shading='gouraud')
    m = ax2.collections[0]
    m.set_clim(-3, 3)

    ax3.pcolormesh(times, distance / (2 * np.pi), energy_half_excited.T, cmap='inferno', shading='gouraud')
    m = ax3.collections[0]
    m.set_clim(-3, 3)

    ax4.pcolormesh(times, distance / (2 * np.pi), energy_all_excited.T, cmap='inferno', shading='gouraud')
    m = ax4.collections[0]
    m.set_clim(-3, 3)
    ax1.set_xlim([0, 2])
    ax2.set_xlim([0, 2])
    ax3.set_xlim([0, 2])
    ax4.set_xlim([0, 2])

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4.set_xticks([])
    ax4.set_yticks([])
    plt.tight_layout()

    f2, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    purity_coherent = np.load('Data/purity0.npy')
    purity_cat = np.load('Data/purity1.npy')
    purity_half_excited = np.load('Data/purity2.npy')
    purity_all_excited = np.load('Data/purity3.npy')

    times = np.linspace(0, Tf, RES)

    ax1.pcolormesh(times, distance / (2 * np.pi), purity_coherent.T, cmap='inferno', shading='gouraud')
    m = ax1.collections[0]
    m.set_clim(0, 1)
    times = np.linspace(0, Tf, RES)

    ax2.pcolormesh(times, distance / (2 * np.pi), purity_cat.T, cmap='inferno', shading='gouraud')
    m = ax2.collections[0]
    m.set_clim(0, 1)

    ax3.pcolormesh(times, distance / (2 * np.pi), purity_half_excited.T, cmap='inferno', shading='gouraud')
    m = ax3.collections[0]
    m.set_clim(0, 1)

    ax4.pcolormesh(times, distance / (2 * np.pi), purity_all_excited.T, cmap='inferno', shading='gouraud')
    m = ax4.collections[0]
    m.set_clim(0, 1)
    ax1.set_xlim([0, 2])
    ax2.set_xlim([0, 2])
    ax3.set_xlim([0, 2])
    ax4.set_xlim([0, 2])

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4.set_xticks([])
    ax4.set_yticks([])
    plt.tight_layout()
    plt.show()


def main():
    times = np.linspace(0, Tf, RES)
    atom_dim = 2
    num_emitters = 6
    light_dim = num_emitters + 1
    d = 1/2 # in units of wavelength, adding a wavelength doesnt matter
    # d = 0 this is lambda spacing
    psi_init = coherent_state(num_emitters, np.pi / 2)
    # psi_init = dicke_state(num_emitters, 3)

    rho_eta_final = get_density_matrix_in_virtual_cavity(times, psi_init, 1, atom_dim, light_dim, 2 * np.pi * d, num_emitters,
                                          plot=True,
                                          state_outside=True, modes=False)

    # distances = np.linspace(0, 2 * np.pi, 100)

    # psi_inits = coherent_state(num_emitters, np.pi / 2, phase_factor), cat_state(num_emitters, np.pi / 2, phase_factor), \
    #             dicke_state(num_emitters, (num_emitters - 1) // 2, phase_factor), dicke_state(num_emitters,
    #                                                                                           num_emitters,
    #                                                                                           phase_factor),

    # create_fig5b(times, distances, psi_inits, num_emitters, atom_dim, light_dim)


if __name__ == '__main__':
    # plot_maps_distance()
    main()
