import numpy as np
import matplotlib.pyplot as plt
import time

def f(x):
    return 1 / (x + 1)

def F(x):
    return 2*x*x + 3

def f_n(x, n, x_nodes):
    #  f_n(x)=f(x_{k+1}) на [x_k, x_{k+1})
    res = np.zeros_like(x)
    for i in range(n):
        mask = (x >= x_nodes[i]) & (x < x_nodes[i+1])
        res[mask] = f(x_nodes[i+1])
    res[x == 4] = f(4)
    return res

# --- Построение графиков ---
x_plot = np.linspace(0, 4, 1000)
plt.figure(figsize=(10, 6))
plt.plot(x_plot, f(x_plot), 'k-', label='$f(x)=1/(x+1)$')
for n in [5, 10, 20]:
    x_nodes = np.linspace(0, 4, n+1)
    y = f_n(x_plot, n, x_nodes)
    plt.step(x_plot, y, where='post', label=f'$f_{n}$ (n={n})')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Аппроксимация $f$ ступенчатыми функциями $f_n$')
plt.legend()
plt.grid(True)
plt.savefig('f_n_approx.png', dpi=150)
plt.show()

# --- Вычисление интегралов ---
n_vals = [10, 100, 1000]
true_lebesgue = np.log(5)
true_stieltjes = 16 - 4*np.log(5)

print("Интегралы Лебега от f_n:")
for n in n_vals:
    dx = 4 / n
    x_right = np.linspace(dx, 4, n)
    I = np.sum(f(x_right) * dx)
    err = true_lebesgue - I
    print(f"n = {n:4d} : {I:.8f}, ошибка = {err:.8f}")

print(f"\nТочное значение (Лебег): {true_lebesgue:.8f}\n")

print("Интегралы Лебега-Стилтьеса от f_n:")
for n in n_vals:
    x_nodes = np.linspace(0, 4, n+1)
    delta_F = F(x_nodes[1:]) - F(x_nodes[:-1])
    f_right = f(x_nodes[1:])
    J = np.sum(f_right * delta_F)
    err = true_stieltjes - J
    print(f"n = {n:4d} : {J:.8f}, ошибка = {err:.8f}")

print(f"\nТочное значение (Стилтьес): {true_stieltjes:.8f}")
