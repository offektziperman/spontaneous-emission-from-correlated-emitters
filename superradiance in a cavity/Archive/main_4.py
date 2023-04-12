# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import calculate_W_t_no_markov
from Operators import make_3d_exp, find_modes
from calculate_W_t_no_markov import W_t_no_markov
from Operators import Operators
from distribution_functions import Wigner
from pylab import *
from moviepy.editor import *
from distribution_functions import Atomic_state_on_bloch_sphere
import qutip

INFINITY = 1000
NUM_POINTS = 1000


# Press the green button in the gutter to run the script.
def plot_wigner(x, p, wigner, f):
    ax2 = f.add_subplot()
    ax2.pcolormesh(x, p, wigner, cmap='bwr')
    m = ax2.collections[0]
    # ax2.set_xlabel('$q$', fontsize=16)
    # ax2.set_ylabel('$p$', fontsize=16)
    # ax2.set_aspect('equal')
    # g = plt.colorbar(m)
    # g.set_label('$W(q,p)$', fontsize=16)
    m.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()


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
    # indices = [0,10,20,30,40,50,60,70]
    # fig = plt.figure()
    # for i in range(len(indices)):
    #     atomic_wigner.Wigner_BlochSphere(100, np.size(rho_atoms, 1) - 1, [], rho_atoms[indices[i]], 'rho',ax[0,i])
    #     x = np.linspace(-w.alpha_max, w.alpha_max, w.N_alpha)
    #     wigner = qutip.wigner(qutip.Qobj(rho_light[indeces[i]]), x, x)
    #     ax1.pcolormesh(x, x, wigner, cmap='bwr')
    #     m = ax[1,i].collections[0]
    #     ax[1,i].set_aspect('equal')
    #
    #     m.set_clim(-0.15, 0.15)  # np.max(np.abs(wigner)), np.max(np.abs(wigner)))
    #     ax[1,i].set_aspect('equal')
    #     plt.xticks([], fontsize=30)
    #     plt.yticks([], fontsize=30)
    #
    #     wigner = qutip.wigner(qutip.Qobj(rho_eta[indeces[i]]), x, x)
    #     ax[2,i].pcolormesh(x, x, wigner, cmap='bwr')
    #     m = ax3.collections[0]
    #     ax[2,i].set_aspect('equal')
    #     m.set_clim(-0.4, 0.4)
    #     ax[2,i].set_aspect('equal')
    #     plt.xticks([], fontsize=30)
    #     plt.yticks([], fontsize=30)
    # plt.show()


def single_cycle(W_t, kappa, kappa_s, atomic_rho, light_rho, M, N, state):
    W_t.kappa = kappa
    W_t.kappa_s = kappa_s

    rho_f_initial = qutip.tensor(atomic_rho, light_rho)
    times = np.linspace(0, 20, 200)
    dt = times[1] - times[0]
    colors = ['red', 'green', 'black', 'blue', 'orange']

    # W_t.kappa = kappa
    time_s, rho_f_t, negativity, mode, number = W_t.solve_cavity_sr(rho_f_initial, M, N, times)
    rho_f_initial = qutip.tensor(atomic_rho, light_rho, light_rho)
    time_s, rho_f_t = W_t.get_light_outside(rho_f_initial, mode, M, N, times)
    rho_eta = np.array([rho_f_t[i].ptrace(2) for i in range(len(times))])
    fid = qutip.fidelity(qutip.Qobj(rho_eta[-1]), state)
    print(fid)
    return fid


import multiprocessing as mp

fn = 'output_test.txt'


def worker(args, q):
    """
    do some work, put results in queue
    """
    i, kappa, W_t, kappa_s_vec, atomic_rho, light_rho, M, N, state = args
    arr = np.zeros(len(kappa_s_vec))
    for j, kappa_s in enumerate(kappa_s_vec):
        arr[j] = single_cycle(W_t, kappa, kappa_s, atomic_rho, light_rho, M, N, state)
    done = str((kappa,arr))

    # res = 'Process' + str(i), str(size), str((kappa, arr))

    res = done

    print(res)
    q.put(res)
    return res

# def worker(a, q):
#     '''stupidly simulates long running process'''
#     arg = a
#     start = time.localtime()
#     s = 'this is a test'
#     txt = s
#     for i in range(200000):
#         txt += s
#     done = time.localtime()
#     with open(fn, 'rb') as f:
#         size = len(f.read())
#     res = 'Process' + str(arg), str(size), done
#     q.put(res)
#     return res


def listener(q):
    """
    continue to listen for messages on the queue and writes to file when receive one
    if it receives a '#done#' message it will exit
    """
    with open(fn, 'w') as f:
        while 1:
            m = q.get()
            print(m)
            if m == 'kill':
                f.write('killed')
                break
            f.write(str(m) + '\n')
            f.flush()


def make_maps(kappa_s_vec, kappa_vec, atomic_rho, light_rho, N, M, W_t):
    '''

    :return:
    '''

    fidelity_state = np.zeros([len(kappa_vec), len(kappa_s_vec)])

    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count() + 2)

    watcher = pool.apply_async(listener, (q,))

    # state = light_cat(M, 1.3j)
    state = qutip.basis(M, M - 1)

    # fire off workers
    jobs = []
    for i, kappa in enumerate(kappa_vec):
        args = i, kappa, W_t, kappa_s_vec, atomic_rho, light_rho, M, N, state
        job = pool.apply_async(worker, (args, q))
        jobs.append(job)
        print(i)

    # collect results from the workers through the pool result queue

    for job in jobs:
        job.wait()
    q.put('kill')
    pool.close()
    pool.join()


def atomic_cat(N, alpha):
    psi = qutip.basis(N, 0)
    Sx = qutip.jmat((N - 1) / 2, 'x')
    Sy = qutip.jmat((N - 1) / 2, 'y')
    # to_exp = 1j * Sx * alpha
    # to_exp2 = 1j * Sx * alpha * np.cos(2 * np.pi / 3) + 1j * Sy * alpha * np.sin(2 * np.pi / 3)
    # to_exp3 = 1j * Sx * alpha * np.cos(4 * np.pi / 3) + 1j * Sy * alpha * np.sin(4 * np.pi / 3)
    to_exp = 1j * Sx * alpha
    to_exp2 = -1j * Sx * alpha
    # psi = ((to_exp.expm() + to_exp2.expm() + to_exp3.expm()) * psi)
    psi = ((to_exp.expm() + to_exp2.expm()) * psi)

    return psi / psi.norm()


def light_cat(N, alpha):
    psi = qutip.basis(N, 0)
    Sx = qutip.jmat((N - 1) / 2, 'x')
    Sy = qutip.jmat((N - 1) / 2, 'y')
    to_exp = 1j * Sx * alpha
    to_exp2 = 1j * Sx * alpha * np.cos(2 * np.pi / 3) + 1j * Sy * alpha * np.sin(2 * np.pi / 3)
    to_exp3 = 1j * Sx * alpha * np.cos(4 * np.pi / 3) + 1j * Sy * alpha * np.sin(4 * np.pi / 3)

    psi = ((to_exp.expm() + to_exp2.expm() + to_exp3.expm()) * psi)
    return psi / psi.norm()


def calc_intensity_and_entanglement_entropy(atomic_rho, light_rho, kappas, M, N, times, W_t, colors, spec=False):
    '''

    :return:
    '''
    W_t.kappa_s = 0
    entanglement_entropy = np.zeros([len(kappas), len(times)])
    intensity = np.zeros([len(kappas), len(times)])
    S = np.zeros([len(kappas), len(times)])

    a = qutip.tensor(qutip.qeye(M), qutip.destroy(N))
    f1 = plt.figure()
    f2 = plt.figure()
    f3 = plt.figure()
    ax1 = f1.add_subplot()
    ax2 = f2.add_subplot()
    ax3 = f3.add_subplot()

    number_op = a.dag() * a

    n = M - 1
    for j, kappa in enumerate(kappas):
        print(j)
        W_t.kappa = kappa
        rho_f_initial = qutip.tensor(atomic_rho, light_rho)
        time_s, rho_f_t, entropy, mode, number, w, S[j, :] = W_t.solve_cavity_sr(rho_f_initial, M, N, times, spec=True)
        # time_s, rho_f_t, entropy, mode, number = W_t.solve_cavity_sr(rho_f_initial, M, N, times, spec=False)

        print(sum(S[j, :]) * (w[2] - w[1]))
        entanglement_entropy[j, :] = np.array([qutip.entropy_vn(rho_f_t[i], 2) for i in range(len(times))])
        intensity[j, :] = np.array([kappa * qutip.expect(number_op, rho_f_t[i]) for i in range(len(times))])
        ax1.plot(times, intensity[j, :], linewidth=2, color=colors[j])
        ax2.plot(times, entanglement_entropy[j, :], linewidth=2, color=colors[j])
        dw = w[2] - w[1]
        ax3.plot(w, n * S[j, :] / (sum(S[j, :] * dw)), linewidth=2, color=colors[j])

    ax1.set_xlim([0, 10])
    ax2.set_xlim([0, 10])
    ax3.set_xlim([-10, 10])

    # plt.legend(['     ' for j in range(len(kappas))], fontsize=30)
    plt.show()
    np.save('intensity_vs_kappa', intensity)
    np.save('entanglement_entropy', entanglement_entropy)
    np.save('spectrum_fock', S)
    np.save('omega_fock', w)

    np.save('times', times)
    np.save('kappas', kappas)


def squeezed_state(N, alpha):
    psi = qutip.basis(N, N - 1)
    Sx = qutip.jmat((N - 1) / 2, 'x')
    Sy = qutip.jmat((N - 1) / 2, 'y')
    to_exp = 1j * Sx * Sx * alpha
    psi = to_exp.expm() * psi
    return psi / psi.norm()


def plot_fidelity(kappa_s_vec, kappa_vec, ):
    with open(fn, 'rd') as f:



def main():
    '''
    '''
    N = 2  # Size of atom basis (number of atoms + 1)
    M = 2  # Size of light basis

    # atomic_rho = qutip.Qobj(nir_rho)  #uncomment this line

    # atomic_rho = qutip.ket2dm(atomic_cat(N, 1.3j))  # initial atomic state example cat state
    atomic_rho = qutip.ket2dm(qutip.basis(N, N - 1))  # initial atomic state example fock state
    light_rho = qutip.ket2dm(qutip.basis(M, 0))  # initial light state
    Sx = qutip.jmat((N - 1) / 2, 'x')

    to_exp = 1j * Sx * np.pi
    atomic_rho = to_exp.expm() * atomic_rho * (to_exp.dag()).expm()  # qutip uses density matrices that are flipped
    rho_f_initial = qutip.tensor(atomic_rho, light_rho)
    op = Operators(N)
    # times = np.linspace(0, 80, 600)  # time vector - in units of g^-1
    # dt = times[1] - times[0]

    # xsi = np.array([0.1, 0.3, 2, 10, 100])  # reabsorbtion efficiency
    # kappa = (2 * (M - 1) / xsi) ** 0.5
    # kappa_s = 0
    RES = 60
    # times = np.linspace(0, 50, 200)
    # dt = times[1] - times[0]
    colors = ['red', 'green', 'black', 'blue', 'orange', 'brown']
    kappa_s_vec = np.linspace(0, 0.3, RES)
    kappa_vec = np.linspace(0, 6, RES) * (M - 1) ** 0.5
    W_t = W_t_no_markov(op, N, M, atomic_rho, light_rho)
    # W_t.kappa = (2 * (M - 1) / xsi) ** 0.5
    # W_t.kappa_s = 0

    # calc_intensity_and_entanglement_entropy(atomic_rho, light_rho, kappa, M, N, times, W_t, colors)
    fidelity = make_maps(kappa_s_vec, kappa_vec, atomic_rho, light_rho, N, M, W_t)
    #
    # time_s, rho_f_t, negativity, mode, number = W_t.solve_cavity_sr(rho_f_initial, M, N, times)
    #
    # rho_f_initial = qutip.tensor(atomic_rho, light_rho, light_rho)
    # time_s, rho_f_t = W_t.get_light_outside(rho_f_initial, mode, M, N, times)
    #
    # rho_atoms = np.array([rho_f_t[i].ptrace(0) for i in range(len(times))])
    # rho_light = np.array(
    #     [rho_f_t[i].ptrace(1) for i in range(len(times))])  # light inside the cavity as a function of time
    # rho_eta = np.array(
    #     [rho_f_t[i].ptrace(2) for i in range(len(times))])  # light outside in a traveling pulse as a function of time
    #
    # # Save data to file
    # np.save('rho_atoms', rho_atoms)
    # np.save('rho_light', rho_light)
    # np.save('rho_eta', rho_eta)
    # qutip.plot_wigner(qutip.Qobj(rho_eta[-1]), cmap='bwr', colorbar=True)
    # plt.title('Quantum state of mode')
    #
    # plt.show()
    #
    # plt.plot(times, np.abs(mode) ** 2)
    # plt.title('Shape of mode')
    # plt.xlabel('time $(g^{-1})$', fontsize=14)
    # plt.ylabel('Intensity $(g)$')
    # plt.show()


def test_unitary():
    prob = np.zeros([198, 1])
    N = 200
    atomic_psi = np.zeros([N + 1, 1])
    atomic_psi[20] = 1
    for N in range(2, 200):
        atomic = Atomic_state_on_bloch_sphere(N)
        op = Operators(N)
        Sx, Sy, Sz = op.create_spin_matrices(N)
        atomic_rho = np.zeros([N + 1, N + 1])
        for n in range(N):
            atomic_rho[n, n] = math.comb(N, n) / 2 ** N
        atomic_rho = scipy.linalg.expm(-1j * pi / 2 * Sx) @ atomic_rho @ scipy.linalg.expm(1j * pi / 2 * np.conj(Sx.T))

        prob[N - 2] = atomic_rho[0, 0] * 2
    fig = plt.figure()
    t = np.linspace(2, 200, 198)
    plt.plot(t, prob, linewidth=2)
    plt.plot(t, t ** -0.5, linewidth=2, linestyle='dashed')
    # plt.legend(['           ', '           '], frameon=False, fontsize=30)
    plt.xlim([0, 200])
    plt.ylim([0, 1])
    plt.xticks([], fontsize=18)
    plt.yticks([], fontsize=18)

    plt.show()
    atomic_psi = np.zeros([N + 1])
    atomic_psi[1] = 1
    atomic_psi_new = scipy.linalg.expm(1j * pi / 2 * Sx) @ atomic_psi.T
    atomic_psi_new[np.abs(atomic_psi_new) < 10 ** -3] = 0
    # plt.show()


def fid_under_v(fid, v, kappa, kappa_s):
    kappa_under = []
    print(v)
    for i in range(len(kappa)):
        j = where(fid[i, :] < v)[0]
        if not np.any(j):
            j = np.size(kappa_s) - 1
        else:
            j = j[0]
        print(j)
        kappa_under.append(kappa_s[j])
    return kappa_under


if __name__ == '__main__':
    main()
