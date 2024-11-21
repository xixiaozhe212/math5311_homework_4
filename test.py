import numpy as np
import matplotlib.pyplot as plt


def gauss_seidel(A, b, max_iter=50000, tol=1e-8):
    N = len(b)
    x = np.zeros(N)
    r_history = []
    for k in range(max_iter):
        x_new = np.zeros(N)
        for i in range(N):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i, N))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        x = x_new
        r = np.dot(A, x) - b
        r_norm = np.linalg.norm(r)
        b_norm = np.linalg.norm(b)
        r_history.append(r_norm)
        if r_norm / b_norm < tol:
            break
    return x, r_history


def main():
    N_values = [49, 99, 199, 399]
    for N in N_values:
        x_exact = np.array([np.sin(10 * j * np.pi / N) for j in range(N)])
        A = np.zeros((N, N))
        for i in range(N):
            A[i][i] = 2
            if i > 0:
                A[i][i - 1] = -1
            if i < N - 1:
                A[i][i + 1] = -1
        b = np.dot(A, x_exact)
        x_approx, r_history = gauss_seidel(A, b)
        final_error = np.linalg.norm(x_approx - x_exact)
        plt.plot(r_history, label=f'N = {N}')
        print(f'Final error for N = {N}: {final_error}')
    plt.xlabel('Iteration')
    plt.ylabel('Residual norm')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()