# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
import scipy.linalg
import superradiance_no_cavity
from Operators import make_3d_exp
from superradiance_no_cavity import Superradiance_no_cavity
from Operators import Operators
from distribution_functions import Wigner
from pylab import *
from moviepy.editor import *
from distribution_functions import Atomic_state_on_bloch_sphere

INFINITY = 1000
NUM_POINTS = 1000


# Press the green button in the gutter to run the script.


def plot_wigner(x, p, wigner, f):
    ax2 = f.add_subplot()
    ax2.pcolormesh(x, p, wigner, cmap='bwr')
    m = ax2.collections[0]
    ax2.set_xlabel('$q$', fontsize=16)
    ax2.set_ylabel('$p$', fontsize=16)
    ax2.set_aspect('equal')
    g = plt.colorbar(m, shrink=0.5)
    g.set_label('$W(q,p)$', fontsize=16)
    m.set_clim(-np.max(np.abs(wigner)), np.max(np.abs(wigner)))
    plt.tight_layout()
    plt.show()


def test_Wigner_from_moments():
    '''

    :return:
    '''
    N = 1
    NUM_MOMENTS = 11
    op = Operators(N)
    w = Wigner(alpha_max=4)
    rho = np.zeros([11, 11])
    for i in range(10):
        rho[i, i] = 1 / 11
    a, a_dagger = op.create_a_and_a_dagger(N)

    # m, n = np.meshgrid(np.linspace(0, NUM_MOMENTS, NUM_MOMENTS + 1), np.linspace(0, NUM_MOMENTS, NUM_MOMENTS + 1))
    moments = op.operator_moments(NUM_MOMENTS, rho, a, a_dagger)

    x, p, wigner = w.calc_Wigner_from_moments(moments)
    plot_wigner(x, p, wigner, f)


def make_matrix_movie(x, p, tensor, omega, xmax=2, ymax=2, text_label='$\\frac{\Omega - \Omega_0}{\Gamma}=$ ',
                      color_label='$W(q,p,\omega)$', x_label='$q$', y_label='$p$'):
    for index in range(len(omega)):
        print(str(index) + '/' + str(len(omega)))
        f = plt.figure()
        plt.pcolormesh(x, p, np.real(tensor[:, :, index]), cmap='bwr')
        ax = f.axes[0]
        m = ax.collections[0]
        plt.xlabel('$q$', fontsize=16)
        plt.ylabel('$p$', fontsize=16)
        plt.text(0.1, 1.7, '$\\frac{\omega -\Omega_0}{\Gamma} \;= \;$' + str(
            format((omega[index] - 1) / 0.03, ".2f")), fontsize=16)
        g = plt.colorbar(m, shrink=0.5)
        g.set_label('$W(q,p)$', fontsize=16)
        m.set_clim(-np.max(np.abs(tensor[:, :, index])), np.max(np.abs(tensor[:, :, index])))
        plt.tight_layout()
        plt.savefig('Images/omega/' + str(index) + '.png')
        plt.close(f)
    make_video('omega/', length=len(omega))


def solve_sr_equations(op, N, atomic_rho, omega):
    sr_solver = Superradiance_no_cavity(op, N, atomic_rho)
    time, rho_t = sr_solver.solve_ode()
    dt = time[1] - time[0]
    matrix_exp_iwt = make_3d_exp(N, time, omega)
    # Splus_cumsum = np.cumsum((Sx_t + 1j * Sy_t) * dt * np.conj(matrix_exp_iwt), axis=2)
    return rho_t, time


def calc_wigner(NUM_MOMENTS, Sminus_t, op, omega, time, t_0, g_w, atomic_rho, w, omega_0=0):
    '''

    :return:
    '''
    N = np.size(Sminus_t, 0) - 1
    dt = time[1] - time[0]
    matrix_exp_iwt = make_3d_exp(N, time, omega)
    Sminus_cumsum = np.cumsum(Sminus_t * dt * matrix_exp_iwt, axis=2)
    if np.size(g_w) > 1:
        moments, a_avg = op.operator_moments_b(NUM_MOMENTS, atomic_rho,
                                               np.exp(-1j * omega * time[t_0]) * np.conj(g_w[omega_0]) * Sminus_cumsum[
                                                                                                         :, :,
                                                                                                         t_0],
                                               np.exp(1j * omega * time[t_0]) * g_w[omega_0] *
                                               np.conj(Sminus_cumsum[:, :, t_0].T))
    else:
        moments, a_avg = op.operator_moments_b(NUM_MOMENTS, atomic_rho,
                                               np.exp(-1j * omega * time[t_0]) * np.conj(g_w) * Sminus_cumsum[:, :,
                                                                                                t_0],
                                               np.exp(1j * omega * time[t_0]) * g_w *
                                               np.conj(Sminus_cumsum[:, :, t_0].T))
    plt.pcolormesh(np.abs(moments))
    x, p, wigner = w.calc_Wigner_from_moments(moments, a_avg)
    print(a_avg)
    dx = x[1, 2] - x[1, 1]
    dp = p[2, 1] - p[1, 1]
    return x, p, wigner, Sminus_cumsum

def calc_wigner_for_omega(NUM_MOMENTS, time, op, omega_arr, Sminus_t, g_w, atomic_rho):
    N = np.size(Sminus_t, 0) - 1
    dt = time[1] - time[0]
    t_0 = 10000
    w = Wigner(N_alpha=2000, alpha_max=4)

    tensor = np.zeros([w.N_alpha, w.N_alpha, len(omega_arr)])
    photon_number = np.zeros(np.shape(omega_arr))
    for index, omega in enumerate(omega_arr):
        x, p, wigner, Sminus_cumsum = calc_wigner(NUM_MOMENTS, Sminus_t, op, omega, time, t_0, g_w, atomic_rho, w,
                                                  index)
        photon_number[index] = np.abs(g_w[index]) ** 2 * np.trace(
            atomic_rho @ np.conj(Sminus_cumsum[:, :, t_0].T) @ Sminus_cumsum[:, :, t_0])
        tensor[:, :, index] = wigner
        print(str(index) + '/' + str(len(omega_arr)))
    domega = omega_arr[1] - omega_arr[0]
    print('sum of n=' + str(sum(photon_number) * domega))
    return x, p, tensor, photon_number


def make_video(label, length=100):
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


def plot_wigner_videos(NUM_MOMENTS, atomic_wigner, Sx_t, Sy_t, Sz_t, rho_t, time, omega, op, g_w):
    w = Wigner(N_alpha=500)
    Splus_t = (Sx_t + 1j * Sy_t)
    Sminus_t = np.zeros(np.shape(Splus_t))

    for i in range(len(time)):
        Sminus_t[:, :, i] = np.conj(Splus_t[:, :, i].T)

    for index in range(0, 10000, 5000):
        print(str(index) + '/' + str(len(time)))
        fig = atomic_wigner.Wigner_BlochSphere(100, np.size(rho_t, 0) - 1, [], rho_t[:, :, index], 'rho')
        ax2 = fig.add_subplot(122)
        x, p, wigner, Sminus_cumsum = calc_wigner(NUM_MOMENTS, Sminus_t, op, omega, time, index, g_w, rho_t[:, :, 0], w)
        ax2.pcolormesh(x, p, wigner, cmap='bwr')
        m = ax2.collections[0]
        # ax2.set_xlabel('$q$', fontsize=16)
        # ax2.set_ylabel('$p$', fontsize=16)
        ax2.set_aspect('equal')
        # ax2.text(0.1, 1.7, '$t \;= \;' + str(format(time[index] / 0.03 ** -1, ".2f")) + '\;\Gamma^{-1}$', fontsize=16)
        g = plt.colorbar(m, shrink=0.5)
        g.ax.tick_params(labelsize=20)
        # g.set_label('$W(q,p)$', fontsize=20)
        m.set_clim(-np.max(np.abs(wigner)), np.max(np.abs(wigner)))
        fig.tight_layout()
        ax2.set_aspect('equal')
        plt.xticks([-4, -2, 0, 2, 4], fontsize=20)
        plt.yticks([-4, -2, 0, 2, 4], fontsize=20)

        plt.savefig('Images/time/' + str(index) + '.png')
        plt.close(fig)

    make_video('time/')


def plot_Sz(rho_t, Sz_t, time, deriv=False):
    T = 1000
    expectation_h = [np.trace(rho_t[:, :, 0] @ Sz_t[:, :, i]) for i in range(len(time))]
    expectation_s = [np.trace(rho_t[:, :, i] @ Sz_t[:, :, 0]) for i in
                     range(len(time))]
    # moments_h = [[np.trace(rho_t[:, :, 0] @ np.linalg.matrix_power(np.conj(Splus_t[:, :, T]).T, n) @
    #                        np.linalg.matrix_power(Splus_t[:, :, T], m)) for m in range(5)] for n in range(5)]
    # moments_s = [[np.trace(rho_t[:, :, T] @ np.linalg.matrix_power(np.conj(Splus_t[:, :, 0]).T, n) @
    #                        np.linalg.matrix_power(Splus_t[:, :, 0], m)) for m in range(5)] for n in range(5)]
    # f, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.imshow(np.abs(moments_h))
    # ax1.set_title('Heisenberg')
    # ax2.imshow(np.abs(moments_s))
    # ax2.set_title('Schrodinger')
    # f3 = plt.figure()
    dt = (time[2] - time[1]) * superradiance_no_cavity.GAMMA
    if deriv:
        # line, = plt.plot(time[:-1] * superradiance_no_cavity.GAMMA, -np.diff(expectation_h) / dt, linewidth=3)
        # plt.ylabel('$I(t)\;(\hbar \Omega_0 \Gamma)$', fontsize=16)
        t = np.linspace(-1, 0, 1000)
        # plt.xlim([0, 3])
        return time[:-1] * superradiance_no_cavity.GAMMA, -np.diff(expectation_h) / dt

    else:
        line, = plt.plot(time[:-1] * superradiance_no_cavity.GAMMA, expectation_h, linewidth=3)

        line.set_label('Heisenberg')
        line2, = plt.plot(time * superradiance_no_cavity.GAMMA, expectation_s, linewidth=3, linestyle='--')
        line2.set_label('Schrodinger')
        plt.ylabel('$<S_z>(t)$', fontsize=16)
        line3, = plt.plot(time * superradiance_no_cavity.GAMMA, -0.5 + np.exp(-superradiance_no_cavity.GAMMA * time),
                          linewidth=1, linestyle='-', color='black')
        line3.set_label('Analytic')

    plt.xlim([0, 2])

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()


def plot_intensity(Splus_cumsum, Sminus_t, k, g_w, c, atomic_rho, op, N):
    k = np.linspace(-INFINITY, INFINITY, NUM_POINTS, dtype=complex)
    I_x = np.zeros([len(k), 100])
    jump = 1000
    a_k = op.find_a_k(Sminus_t, k, time, 10000, g_w, c)
    mat = np.zeros([N + 1, N + 1])
    mat[1, 1] = 1
    print(a_k @ mat)
    for t_0 in range(0, 20000, jump):
        E_xplus, E_xminus, x = op.find_a_x(a_k, k)
        x, I_x[:, t_0 // jump] = op.find_intensity(E_xminus, atomic_rho, x)
        plt.plot(x, I_x[:, t_0 // jump], linewidth=2)
        plt.xlabel('x', fontsize=14)
        plt.ylabel('$<E^{\dagger}(x)E(x)>$', fontsize=14)
        plt.ylim([-0.02, 1.1])
        plt.savefig('Images/intensity/' + str(t_0 // jump) + '.png')
        plt.clf()
    plt.show()

    plt.legend(['Simulation', '$-1 + 2e^{-\\Gamma t}$ fit'], fontsize=16)
    plt.legend(fontsize=14, loc='right')
    plt.xlim([0, 2])
    plt.show()


def plot_spectrum(omega, N, photon_number, domega):
    f2 = plt.figure()
    plt.plot(omega, N * photon_number / (sum(photon_number * omega) * domega), linewidth=2)
    plt.xlabel('$\Omega$', fontsize=16)
    plt.ylabel('$\\frac{dn}{d\Omega}$', fontsize=16)
    GAMMA = 0.03

    # y = omega / (np.pi * GAMMA) * (GAMMA ** 2 / ((omega - 1) ** 2 + GAMMA ** 2))
    # plt.plot(omega, y / sum(y * domega), linewidth=2,
    #          linestyle='dashed')
    # plt.legend(['Simulation', 'Analytic fit'], fontsize=16)

    plt.show()


def plot_Pn(rho):
    m = np.linspace(0, np.size(rho, 1) - 1, np.size(rho, 1))
    plt.plot(m, [rho[n, n] for n in range(np.size(rho, 0))], linewidth=2)
    plt.xlabel('$n$', fontsize=16)
    plt.ylabel('$p_n$', fontsize=16)


def main():
    '''
    ### Hi Nir :) to use this:
    ### (1) change N for the number of atoms,
    ### (2) change Num moments, should be greater than number of photons that are emmited. the code gets slow if this number is large (>40)
    ### (3) change atomic_rho as a numpy array.(N+1XN+1) in the symetric state basis. (rho_mn=<m|rho|n>)
    '''
    N = 2
    M = 2
    g_w = superradiance_no_cavity.GAMMA ** 0.5 / ((np.pi) ** 0.5)
    kappa = superradiance_no_cavity.GAMMA
    g = superradiance_no_cavity.g
    c = 0.3
    L = [22, 18]
    Itot = 0
    I_n = np.zeros([10, 10])

    NUM_MOMENTS = N + 1  ###change###

    atomic = Atomic_state_on_bloch_sphere(N)
    op = Operators(N)
    Sx, Sy, Sz = op.create_spin_matrices(N)
    S_plus = Sx + 1j * Sy
    S_minus = np.conj(S_plus.T)
    atomic_rho = op.fock_light_state(N, N)  ###change###
    # atomic_rho = scipy.linalg.expm(-1j * pi / 4 * Sy) @ atomic_rho @ scipy.linalg.expm(1j * pi / 4 * np.conj(Sy).T)
    t_0 = 0.2
    # atomic_rho = np.load('Data/rho_squeezed.npy')
    # atomic_rho = scipy.linalg.expm(-1j * pi / 2 * Sy) @ atomic_rho @ scipy.linalg.expm(1j * pi / 2 * np.conj(Sy).T)
    # fig = atomic.Wigner_BlochSphere(100, np.size(atomic_rho, 0) - 1, [], atomic_rho, 'rho')
    # plt.show()
    omega1 = 1
    S_plus_t, time = solve_sr_equations(op, N, S_plus, omega1)
    spectrum_markov = np.zeros(2000)
    omega_markov = np.linspace(-20, 20, 2000)
    domega_markov = omega_markov[1] - omega_markov[0]
    for i, omega in enumerate(omega_markov):
        print(i)
        matrix_exp_iwt = make_3d_exp((N), time, omega)
        a_dagger_omega = 2 * g ** 2 / kappa * np.cumsum(S_plus_t * matrix_exp_iwt, axis=2)

        spectrum_markov[i] = np.trace(a_dagger_omega[:, :, -1] @ np.conj(a_dagger_omega[:, :, -1].T) @ atomic_rho)
    spectrum_markov = spectrum_markov*7/(sum(spectrum_markov*domega_markov))
    plt.plot(omega_markov, spectrum_markov)
    plt.show()
    np.save('spectrum_markov_kappa=' + str(kappa), spectrum_markov)

    # Splus_t = (Sx_t + 1j * Sy_t)
    # Sminus_t = op.dagger_3d(Splus_t)
    intensity = 2 * g ** 2 / kappa * np.array(
        [np.trace(rho_t[:, :, i] @ S_plus @ S_minus) for i in range(np.size(rho_t, 2))])
    plt.plot(time, intensity)
    dt = time[2] - time[1]
    print(sum(intensity * dt))
    plt.show()
    np.save('intensity_markov_kappa=' + str(kappa), intensity)
    # t, I = plot_Sz(rho_t, Sz_t, time, deriv=True)
    # names = ('Intensity_fock.npy', 'Intensity_fock.npy', 'Intensity_coherent.npy', 'Intensity_coherent.npy',
    #          'Intensity_squeezed.npy')
    # linestyles = ('solid', 'dashed', 'solid', 'dashed', 'solid')
    # for i in range(len(names)):
    #     I = np.load(names[i])
    #     plt.plot(t, I, linewidth=2, linestyle=linestyles[i])
    # plt.legend(['         ', '         ', '         ', '         ', '         '], fontsize=28)
    # fig,  = plt.plot(t, Itot, linewidth=3)
    Itot = 0
    # plt.xlim([0, 0.6])

    # plt.legend(['$\\frac{1}{\sqrt{2}}(|0,N,0,0>+|0,0,N,0>)$', '$|0,N/2,N/2,0>$'],fontsize=14)
    # plt.ylabel('$I(t)$', fontsize=14)
    # plt.xlabel('time $(\Gamma^{-1})$', fontsize=14)
    # fig.axes.get_xaxis().set_visible(False)
    # fig.axes.axis('off')
    plt.xticks([0, 0.25, 0.5, 0.75, 1], fontsize=18)
    plt.yticks(fontsize=18)

    plt.show()

    if not os.path.isdir('images'):
        os.mkdir('images')
    if not os.path.isdir('images/time'):
        os.mkdir('images/time')
    if not os.path.isdir('images/omega'):
        os.mkdir('images/omega')
    # for k in range(10):
    #     plot_Pn(rho_t[:,:,1000+k*1000])
    # plt.legend(np.array(['time = ' + str(format(time[1000+1000*k]*superradiance_no_cavity.GAMMA,".2f"))+'$\Gamma^{-1}$' for k in range(10)],dtype=str))
    # plt.show()
    # plot_Sz(rho_t,Sz_0, time, deriv=True)

    # plot_wigner_videos(NUM_MOMENTS, atomic, Sx_t, Sy_t, Sz_t, rho_t, time, omega1, op, g_w)

    w = Wigner()
    omega = np.linspace(0.6, 1.6, 200)
    g_omega = g_w * (omega / superradiance_no_cavity.OMEGA_0) ** 0.5
    domega = omega[1] - omega[0]

    #
    x, p, wigner_w, photon_number = calc_wigner_for_omega(NUM_MOMENTS, time, op, omega, Sminus_t, g_omega, atomic_rho)
    make_matrix_movie(x, p, wigner_w, omega, x_label='$q$', y_label='$p$',
                      text_label='$\\frac{\Omega - \Omega_0}{\Gamma}\;=\;$', color_label='$W(q,p)$')
    plot_spectrum(omega, N, photon_number, domega)


def test_unitary():
    prob = np.zeros([198, 1])
    N = 200
    print('a')
    atomic_psi = np.zeros([N + 1, 1])
    atomic_psi[20] = 1
    for N in range(2, 200):
        atomic = Atomic_state_on_bloch_sphere(N)
        op = Operators(N)
        Sx, Sy, Sz = op.create_spin_matrices(N)
        atomic_rho = np.zeros([N + 1, N + 1])

        for n in range(N):
            atomic_rho[n, n] = math.comb(N, n) / 2 ** N
        print(N)
        atomic_rho = scipy.linalg.expm(-1j * pi / 2 * Sx) @ atomic_rho @ scipy.linalg.expm(1j * pi / 2 * Sx.T)
        prob[N - 2] = atomic_rho[0, 0] * 2
    fig = plt.figure()
    t = np.linspace(2, 200, 198)
    f, ax = plt.subplots(1, 1)
    ax.plot(t, prob, linewidth=2)
    ax.plot(t, t ** -0.5, linewidth=2, linestyle='dashed')
    ax.legend(['           ', '           '], frameon=False, fontsize=30)
    plt.xlim([0, 200])
    plt.ylim([0, 1])
    plt.xticks([0, 50, 100, 150, 200], fontsize=24)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=24)
    plt.setp(ax.spines.values(), linewidth=2)

    plt.show()
    # plt.imshow(np.abs(scipy.linalg.expm(1j * pi / 2 * Sx)))
    # plt.xlabel('n',fontsize=16)
    # plt.ylabel('m',fontsize=16)
    # h = plt.colorbar()
    # h.set_label('$|\\zeta_{N,m,n}|$',fontsize=16)
    # plt.show()
    atomic_psi = np.zeros([N + 1])
    atomic_psi[1] = 1
    atomic_psi_new = scipy.linalg.expm(1j * pi / 2 * Sx) @ atomic_psi.T
    atomic_psi_new[np.abs(atomic_psi_new) < 10 ** -3] = 0

    # plt.show()


if __name__ == '__main__':
    # print(pi)
    main()
    # test_unitary()
    # make_video('time/')
    # test_Wigner_from_moments()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
