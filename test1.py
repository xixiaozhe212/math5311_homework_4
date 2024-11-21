import numpy as np
import matplotlib.pyplot as plt


def gauss_seidel(A, b, max_iter=50000, tol=1e-8):
    x = np.zeros_like(b)
    r_history = []
    for _ in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(len(A)):
            x_new[i] = (b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
        r = np.dot(A, x_new) - b
        r_norm = np.linalg.norm(r)
        b_norm = np.linalg.norm(b)
        r_history.append(r_norm)
        if r_norm / b_norm < tol:
            break
        x = x_new
    return x, r_history


def main():
    N_values = [49, 99, 199, 399]
    for N in N_values:
        x_exact = np.array([np.sin(10 * j * np.pi / N) for j in range(N)])
        A = np.diag([2] * N) + np.diag([-1] * (N - 1), k=1) + np.diag([-1] * (N - 1), k=-1)
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