import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math
import random

FUNCTIONS = {
    "1) 1 / sqrt(2x² + 1.3)   (інтервал [1, 2])": {
        "func_np": lambda x: 1.0 / np.sqrt(2.0 * x**2 + 1.3),
        "func_scalar": lambda x: 1.0 / math.sqrt(2.0 * x**2 + 1.3),
        "a": 1.0, "b": 2.0
    },
    "2) tan(x²) / (x² + 1)     (інтервал [0.2, 1])": {
        "func_np": lambda x: np.tan(x**2) / (x**2 + 1.0),
        "func_scalar": lambda x: math.tan(x**2) / (x**2 + 1.0),
        "a": 0.2, "b": 1.0
    },
    "3) 1 / sqrt(2x² + 1.6)   (інтервал [0.15, 0.5])": {
        "func_np": lambda x: 1.0 / np.sqrt(2.0 * x**2 + 1.6),
        "func_scalar": lambda x: 1.0 / math.sqrt(2.0 * x**2 + 1.6),
        "a": 0.15, "b": 0.5
    }
}

N_VALUES = [10, 20, 50, 100, 1000]


def midpoint_rule(func_np, a, b, N):
    xs = np.linspace(a, b, N + 1)
    mids = (xs[:-1] + xs[1:]) / 2.0
    fx = func_np(mids)
    return np.sum(fx * (b - a) / N)


def trapezoid_rule(func_np, a, b, N):
    xs = np.linspace(a, b, N + 1)
    fx = func_np(xs)
    h = (b - a) / N
    return (h / 2) * (fx[0] + 2 * np.sum(fx[1:-1]) + fx[-1])


def monte_carlo_rule(func_scalar, a, b, N):
    total = 0
    for _ in range(N):
        x = random.random() * (b - a) + a
        total += func_scalar(x)
    return total * (b - a) / N


class IntegralApp:
    def __init__(self, root):
        self.root = root
        root.title("Обчислення визначених інтегралів — Бондар Віктор")
        root.geometry("850x580")

        #  Вибір функції
        ttk.Label(root, text="Оберіть функцію:", font=("Arial", 11, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 2))
        self.combo = ttk.Combobox(root, values=list(FUNCTIONS.keys()), state="readonly", width=55)
        self.combo.pack(anchor=tk.W, padx=10)
        self.combo.current(0)
        self.combo.bind("<<ComboboxSelected>>", self.update_limits)

        # --- Межі інтегрування ---
        frm_limits = ttk.Frame(root)
        frm_limits.pack(anchor=tk.W, padx=10, pady=6)
        ttk.Label(frm_limits, text="a =").grid(row=0, column=0)
        self.entry_a = ttk.Entry(frm_limits, width=10)
        self.entry_a.grid(row=0, column=1, padx=4)
        ttk.Label(frm_limits, text="b =").grid(row=0, column=2)
        self.entry_b = ttk.Entry(frm_limits, width=10)
        self.entry_b.grid(row=0, column=3, padx=4)
        ttk.Button(frm_limits, text="Оновити межі", command=self.update_limits).grid(row=0, column=4, padx=10)
        # Вибір N 
        frm_n = ttk.Frame(root)
        frm_n.pack(anchor=tk.W, padx=10, pady=(5, 5))
        ttk.Label(frm_n, text="Кількість січень (N):", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=(0, 10))
        self.combo_n = ttk.Combobox(frm_n, values=N_VALUES, state="readonly", width=10)
        self.combo_n.grid(row=0, column=1)
        self.combo_n.current(0)

        # Кнопки методів 
        frm_buttons = ttk.Frame(root)
        frm_buttons.pack(anchor=tk.W, padx=10, pady=(10, 5))
        ttk.Label(frm_buttons, text="Методи обчислення:", font=("Arial", 11, "bold")).pack(anchor=tk.W)
        ttk.Button(frm_buttons, text="Метод прямокутників", width=28, command=self.run_midpoint).pack(pady=2)
        ttk.Button(frm_buttons, text="Метод трапецій", width=28, command=self.run_trapezoid).pack(pady=2)
        ttk.Button(frm_buttons, text="Метод Монте-Карло", width=28, command=self.run_montecarlo).pack(pady=2)

        # Таблиця результатів
        ttk.Label(root, text="Результати обчислень:", font=("Arial", 11, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 2))
        cols = ("N", "Аналітичне", "Прямокутники", "Трапеції", "Монте-Карло")
        self.tree = ttk.Treeview(root, columns=cols, show="headings", height=10)
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=130, anchor=tk.CENTER)
        self.tree.pack(padx=10, pady=5)

        self.update_limits()

    def update_limits(self, event=None):
        fdata = FUNCTIONS[self.combo.get()]
        self.entry_a.delete(0, tk.END)
        self.entry_b.delete(0, tk.END)
        self.entry_a.insert(0, fdata["a"])
        self.entry_b.insert(0, fdata["b"])

    def compute_results(self, func_np, func_scalar, a, b):
        true_val, _ = quad(func_scalar, a, b)
        results = []
        for N in N_VALUES:
            m = midpoint_rule(func_np, a, b, N)
            t = trapezoid_rule(func_np, a, b, N)
            mc = monte_carlo_rule(func_scalar, a, b, N)
            results.append((N, true_val, m, t, mc))
        return results, true_val

    def update_table(self, results):
        for r in self.tree.get_children():
            self.tree.delete(r)
        for (N, true_val, m, t, mc) in results:
            self.tree.insert("", tk.END, values=(N, f"{true_val:.8f}", f"{m:.8f}", f"{t:.8f}", f"{mc:.8f}"))

    def show_plot(self, method, func_np, func_scalar, a, b, true_val, N):
        fig, ax = plt.subplots(figsize=(7, 5))
        xs = np.linspace(a, b, 1000)
        ys = func_np(xs)
        ax.plot(xs, ys, 'b', linewidth=1.8)
        ax.grid(True)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.fill_between(xs, ys, alpha=0.15, color='skyblue')

        if method == "прямокутників":
            xs_div = np.linspace(a, b, N + 1)
            mids = (xs_div[:-1] + xs_div[1:]) / 2
            heights = func_np(mids)
            width = (b - a) / N
            for xi, h in zip(mids, heights):
                ax.add_patch(plt.Rectangle((xi - width/2, 0), width, h, color='royalblue', alpha=0.4))

        elif method == "трапецій":
            xs_div = np.linspace(a, b, N + 1)
            ys_div = func_np(xs_div)
            for i in range(N):
                x0, x1 = xs_div[i], xs_div[i+1]
                y0, y1 = ys_div[i], ys_div[i+1]
                ax.fill([x0, x0, x1, x1], [0, y0, y1, 0], alpha=0.3, color='orange')

        elif method == "монте-карло":
            N_points = N
            xs_rand = np.random.uniform(a, b, N_points)
            ys_func = np.array([func_scalar(x) for x in xs_rand])
            y_max = max(ys)
            ys_rand = np.random.uniform(0, y_max, N_points)
            below = ys_rand <= ys_func
            ax.scatter(xs_rand[below], ys_rand[below], s=15, color='green', alpha=0.5)
            ax.scatter(xs_rand[~below], ys_rand[~below], s=15, color='red', alpha=0.5)

        plt.tight_layout()
        plt.show(block=False)

    def run_method(self, method):
        key = self.combo.get()
        data = FUNCTIONS[key]
        try:
            a = float(self.entry_a.get())
            b = float(self.entry_b.get())
        except ValueError:
            messagebox.showerror("Помилка", "Введіть числові межі інтегрування.")
            return
        if a >= b:
            messagebox.showerror("Помилка", "Потрібно, щоб a < b.")
            return

        func_np = data["func_np"]
        func_scalar = data["func_scalar"]
        results, true_val = self.compute_results(func_np, func_scalar, a, b)
        self.update_table(results)

        N_selected = int(self.combo_n.get())
        self.show_plot(method, func_np, func_scalar, a, b, true_val, N_selected)

    def run_midpoint(self):
        self.run_method("прямокутників")

    def run_trapezoid(self):
        self.run_method("трапецій")

    def run_montecarlo(self):
        self.run_method("монте-карло")


if __name__ == "__main__":
    root = tk.Tk()
    app = IntegralApp(root)
    root.mainloop()

