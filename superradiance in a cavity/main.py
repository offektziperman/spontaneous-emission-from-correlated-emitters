# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import scipy.linalg
from calculate_W_t_no_markov import W_t_no_markov
from Operators import Operators
from distribution_functions import Wigner
from distribution_functions import Atomic_state_on_bloch_sphere
import qutip
import matplotlib.pyplot as plt

INFINITY = 1000
NUM_POINTS = 1000


def make_video(label, length=500):
    img_clips = []
    path_list = []
    # accessing path of each image
    for i in range(length):
        path_list.append(os.path.join('Images/' + label + str(i) + '.png'))
    # creating slide for each image
    for img_path in path_list:
        slide = ImageClip(img_path, duration=1 / 10)
        img_clips.append(slide)
    # concatenating slides
    video_slides = concatenate_videoclips(img_clips, method='compose')
    # exporting final video
    video_slides.write_videofile(label[:-1] + ".mp4", fps=24)


def plot_wigner_videos(atomic_wigner, time_s, rho_atoms, rho_light, rho_eta):
    w = Wigner(N_alpha=400, alpha_max=5)
    rho_atoms = np.rot90(np.rot90(rho_atoms).T, k=3)
    N = np.size(rho_light, 1) - 1
    print(N)
    M = np.size(rho_atoms, 0) - 1
    op = Operators(N)
    DELTA = 100

    for index in range(0, len(time_s), DELTA):
        print(str(index) + '/' + str(len(time_s)))
        fig = atomic_wigner.Wigner_BlochSphere(500, np.size(rho_atoms, 1) - 1, [], rho_atoms[index], 'rho')

        ax2 = fig.add_subplot(312)
        x = np.linspace(-4.5, 4.5, 400)
        wigner = qutip.wigner(qutip.Qobj(rho_light[index]), x, x)
        # divnorm = matplotlib.colors.TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=1)

        ax2.pcolormesh(x, x, wigner / np.max(np.abs(wigner)), cmap='bwr')
        m = ax2.collections[0]
        ax2.set_aspect('equal')

        m.set_clim(-1, 1)
        ax2.set_aspect('equal')
        plt.xticks([], fontsize=30)
        plt.yticks([], fontsize=30)
        ax3 = fig.add_subplot(313)

        wigner = qutip.wigner(qutip.Qobj(rho_eta[index]), x, x)
        ax3.pcolormesh(x, x, wigner / np.max(np.abs(wigner)), cmap='bwr')
        plt.xticks([], fontsize=30)
        plt.yticks([], fontsize=30)
        m = ax3.collections[0]
        ax3.set_aspect('equal')
        plt.colorbar(m)
        m.set_clim(-1, 1)
        ax3.set_aspect('equal')

        plt.savefig('Images/time/' + str(index // DELTA) + '.png')
        plt.close(fig)
    make_video('time/', length=len(time_s) // DELTA)


def single_cycle(W_t, kappa, kappa_s, atomic_rho, light_rho, M, N, state):
    W_t.kappa = kappa
    W_t.kappa_s = kappa_s

    rho_f_initial = qutip.tensor(atomic_rho, light_rho)
    times = np.linspace(0, 8, 100)
    dt = times[1] - times[0]
    colors = ['red', 'green', 'black', 'blue', 'orange']

    # W_t.kappa = kappa
    time_s, rho_f_t, negativity, mode, number = W_t.solve_cavity_sr(rho_f_initial, M, N, times)
    rho_f_initial = qutip.tensor(atomic_rho, light_rho, light_rho)

    time_s, rho_f_t = W_t.get_light_outside(rho_f_initial, mode, M, N, times)
    rho_eta = np.array([rho_f_t[i].ptrace(2) for i in range(len(times))])
    qutip.plot_wigner(qutip.Qobj(rho_eta[-1]),cmap = 'bwr')
    plt.xlim([-5,5])
    plt.ylim([-5,5])

    plt.show()
    fid = qutip.fidelity(qutip.Qobj(rho_eta[-1]), state)
    return fid


def make_maps(gammas, xsis, atomic_rho, light_rho, N, M, W_t):
    '''

    :return:
    '''

    fidelity_state_cat = np.zeros([len(xsis), len(gammas)])
    fidelity_state_fock = np.zeros([len(xsis), len(gammas)])
    Sx = qutip.jmat((N - 1) / 2, 'x')

    to_exp = 1j * Sx * np.pi
    state_cat = atomic_two_cat(M, np.pi/2*1j)
    rho_cat = qutip.ket2dm(state_cat)
    rho_cat = to_exp.expm() * rho_cat * (to_exp.dag()).expm()  # qutip uses density matrices that are flipped

    state_fock = qutip.basis(M, M - 1)
    rho_fock = qutip.ket2dm(state_fock)
    rho_fock = to_exp.expm() * rho_fock * (to_exp.dag()).expm()  # qutip uses density matrices that are flipped


    for i, xsi in enumerate(xsis):
        for j, gamma in enumerate(gammas):
            kappa = (2 * (M - 1) / xsi) ** 0.5  # kappa = 2N/xsi
            print(kappa)
            print(gamma)
            fidelity_state_cat[i, j] = single_cycle(W_t, kappa, gamma, rho_cat, light_rho, M, N, state_cat)

            fidelity_state_fock[i, j] = single_cycle(W_t, kappa, gamma, rho_fock, light_rho, M, N, state_fock)

            print(fidelity_state_fock[i, j])
        np.save('data/fidelities/fidelity_cat', fidelity_state_cat)
        np.save('data/fidelities/fidelity_fock', fidelity_state_fock)

        np.save('data/fidelities/reabsorbtion efficiencies', xsis)
        np.save('data/fidelities/gammas', gammas)

    return fidelity_state


def squeezed_state(N, alpha):
    psi = qutip.basis(N, N - 1)
    Sx = qutip.jmat((N - 1) / 2, 'x')
    Sy = qutip.jmat((N - 1) / 2, 'y')
    to_exp = 1j * Sx * Sx * alpha
    psi = to_exp.expm() * psi
    return psi / psi.norm()


def atomic_two_cat(N, alpha):
    psi = qutip.basis(N, 0)
    Sx = qutip.jmat((N - 1) / 2, 'x')
    Sy = qutip.jmat((N - 1) / 2, 'y')
    to_exp = 1j * Sx * alpha
    to_exp2 = -1j * Sx * alpha
    psi = ((to_exp.expm() + to_exp2.expm()) * psi)

    return psi / psi.norm()


def atomic_three_cat(N, alpha):
    psi = qutip.basis(N, 0)
    Sx = qutip.jmat((N - 1) / 2, 'x')
    Sy = qutip.jmat((N - 1) / 2, 'y')
    to_exp = 1j * Sx * alpha
    to_exp2 = 1j * Sx * alpha * np.cos(2 * np.pi / 3) + 1j * Sy * alpha * np.sin(2 * np.pi / 3)
    to_exp3 = 1j * Sx * alpha * np.cos(4 * np.pi / 3) + 1j * Sy * alpha * np.sin(4 * np.pi / 3)

    psi = ((to_exp.expm() + to_exp2.expm() + to_exp3.expm()) * psi)
    return psi / psi.norm()


def plot_fidelity():
    xsis = np.load('data/fidelities/reabsorbtion efficiencies.npy')
    gammas = np.load('data/fidelities/gammas.npy')
    fidelity_cat = np.load('data/fidelities/fidelity_cat.npy')
    fidelity_fock = np.load('data/fidelities/fidelity_fock.npy')

    f, (ax1, ax2) = plt.subplots(1, 2)

    plt.yscale('log')

    ax1.pcolormesh(gammas, xsis, fidelity_fock, cmap='inferno', shading='gouraud')
    m = ax1.collections[0]
    m.set_clim(0.3, 1)
    ax1.set_yscale('log')
    # m2.set_clim(0, 1)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.pcolormesh(gammas, xsis, fidelity_cat, cmap='inferno', shading='gouraud')
    m = ax2.collections[0]
    m.set_clim(0.7, 1)
    ax2.set_yscale('log')
    plt.yscale('log')
    # plt.colorbar(m)
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.tight_layout()

    plt.show()


def plot_intensity_entropy():
    f, (ax1, ax2) = plt.subplots(1, 2)
    xsi = np.load('Archive/xsis.npy')
    intensity = np.load('data/intensity.npy')
    entanglement = np.load('data/entanglement_entropy.npy')
    times = np.load('Archive/times.npy')

    plt.yscale('log')

    ax1.pcolormesh(times, xsi, intensity, cmap='inferno', shading='gouraud')
    m = ax1.collections[0]
    m.set_clim(0, 8)
    ax1.set_yscale('log')
    ax2.pcolormesh(times, xsi, entanglement, cmap='inferno', shading='gouraud')
    # m.set_clim(-3, 3)
    m2 = ax2.collections[0]
    plt.colorbar(m2)

    # m2.set_clim(0, 1)

    plt.yscale('log')

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.tight_layout()
    plt.show()


def calc_intensity_and_entanglement_entropy(atomic_rho, light_rho, xsi, M, N, times, W_t, colors, spec=False):
    '''

    :return:
    '''
    kappas = (2 * (M - 1) / xsi) ** 0.5

    W_t.kappa_s = 0
    entanglement_entropy = np.zeros([len(kappas), len(times)])
    intensity = np.zeros([len(kappas), len(times)])
    S = np.zeros([len(kappas), 100])

    for j, kappa in enumerate(kappas):
        print(j)
        W_t.kappa = kappa
        rho_f_initial = qutip.tensor(atomic_rho, light_rho)
        intensity[j, :], entanglement_entropy[j, :], w_lst, S[j, :] = W_t.solve_cavity_sr(rho_f_initial, M, N, times,
                                                                                          spec=False,
                                                                                          intensity_and_entropy=True)
        plt.plot(w_lst, S[j, :])
        plt.show()
        # time_s, rho_f_t, entropy, mode, number = W_t.solve_cavity_sr(rho_f_initial, M, N, times, spec=False)
    np.save('data/entanglement_entropy.npy', entanglement_entropy)
    np.save('data/intensity.npy', intensity)
    np.save('times', times)
    np.save('xsis', xsi)


def main():
    '''
    '''
    N = 7  # Size of atom basis (number of atoms + 1)
    M = 7 # Size of light basis

    # atomic_rho = qutip.Qobj(nir_rho)  #uncomment this line

    # atomic_rho = qutip.ket2dm(atomic_cat(N, np.pi/2*1j))  # initial atomic state example cat state
    atomic_rho = qutip.ket2dm(qutip.basis(N, N - 1))  # initial atomic state example fock state
    light_rho = qutip.ket2dm(qutip.basis(M, 0))  # initial light state
    Sx = qutip.jmat((N - 1) / 2, 'x')

    to_exp = 1j * Sx * np.pi
    atomic_rho = to_exp.expm() * atomic_rho * (to_exp.dag()).expm()  # qutip uses density matrices that are flipped
    rho_f_initial = qutip.tensor(atomic_rho, light_rho)
    op = Operators(N)
    times = np.linspace(0, 10, 100)  # time vector - in units of g^-1
    dt = times[1] - times[0]

    xsi = np.logspace(-1,1, 10)  # reabsorbtion efficiency
    gammas = np.linspace(0, 0.15, 10)
    kappa = (2 * (M - 1) / xsi) ** 0.5  # kappa = 2N/xsi

    RES = 100

    colors = ['red', 'green', 'black', 'blue', 'orange', 'brown']
    W_t = W_t_no_markov(op, N, M, atomic_rho, light_rho)
    W_t.kappa = (2 * (M - 1) / xsi) ** 0.5
    W_t.kappa_s = 0
    plot_fidelity()
    # calc_intensity_and_entanglement_entropy(atomic_rho, light_rho, xsi, M, N, times, W_t, colors)
    fidelity = make_maps(gammas, xsi, atomic_rho, light_rho, N, M, W_t)
    #

    time_s, rho_f_t, negativity, mode, number = W_t.solve_cavity_sr(rho_f_initial, M, N, times)
    #


if __name__ == '__main__':
    # plot_intensity_entropy()
    main()
