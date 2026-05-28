import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from plot_main import save_tradeoff_data


# ============================================================
# Helpers
# ============================================================
def ul_rate(B_ul, alpha_ul, p_ul):
    """Per-user uplink rate [bps]."""
    return B_ul * np.log2(1.0 + alpha_ul * p_ul)


def ul_latency(K_rounds, S_ul, B_ul, alpha_ul, p_ul):
    """Per-user uplink latency over all rounds [s]."""
    rate = ul_rate(B_ul, alpha_ul, p_ul)
    return K_rounds * S_ul / np.maximum(rate, 1e-30)


def ul_energy(K_rounds, S_ul, B_ul, alpha_ul, p_ul):
    """Per-user uplink transmit energy over all rounds [J]."""
    return p_ul * ul_latency(K_rounds, S_ul, B_ul, alpha_ul, p_ul)


def ul_energy_grad(K_rounds, S_ul, B_ul, alpha_ul, p_ul):
    """
    Derivative of:
        E_ul(p) = c * p / log(1 + alpha p)
    where c = K_rounds * S_ul * ln(2) / B_ul
    """
    z = 1.0 + alpha_ul * p_ul
    logz = np.log(z)
    c = K_rounds * S_ul * np.log(2.0) / B_ul
    return c * (logz - (alpha_ul * p_ul) / z) / np.maximum(logz**2, 1e-30)


def dl_broadcast_rate(B_dl, alpha_dl_min, p_dl):
    """
    Broadcast downlink rate [bps], limited by weakest user.
    """
    return B_dl * np.log2(1.0 + alpha_dl_min * p_dl)


def dl_latency(K_rounds, S_dl, B_dl, alpha_dl_min, p_dl):
    """Total broadcast downlink latency over all rounds [s]."""
    rate = dl_broadcast_rate(B_dl, alpha_dl_min, p_dl)
    return K_rounds * S_dl / max(rate, 1e-30)


def dl_energy(K_rounds, S_dl, B_dl, alpha_dl_min, p_dl):
    """Server downlink broadcast transmit energy over all rounds [J]."""
    return p_dl * dl_latency(K_rounds, S_dl, B_dl, alpha_dl_min, p_dl)


def dl_energy_grad(K_rounds, S_dl, B_dl, alpha_dl_min, p_dl):
    """
    Derivative of:
        E_dl(p) = c * p / log(1 + alpha p)
    where c = K_rounds * S_dl * ln(2) / B_dl
    """
    z = 1.0 + alpha_dl_min * p_dl
    logz = np.log(z)
    c = K_rounds * S_dl * np.log(2.0) / B_dl
    return c * (logz - (alpha_dl_min * p_dl) / z) / max(logz**2, 1e-30)


def evaluate_true_metrics(
    D, c_bar, zeta, K_rounds, J, f_hz,
    u, bw,
    Omega, sum_omega_l,
    p_gpu,
    S_ul, B_ul, alpha_ul, p_ul,
    S_dl, B_dl, alpha_dl_min, p_dl,
    include_server_energy=True,
):
    """
    Compute true metrics at a given solution.
    """
    per_user_latency = (
        K_rounds * J * D * c_bar / f_hz
        + J * u * D / bw
        + 2.0 * Omega / bw
        + 2.0 * sum_omega_l / bw
    )

    per_user_energy = (
        K_rounds * zeta * J * D * c_bar * (f_hz ** 2)
        + J * u * D * p_gpu
        + 2.0 * Omega * p_gpu
        + 2.0 * sum_omega_l * p_gpu
    )

    ul_lat = ul_latency(K_rounds, S_ul, B_ul, alpha_ul, p_ul)
    ul_eng = ul_energy(K_rounds, S_ul, B_ul, alpha_ul, p_ul)

    dl_lat = dl_latency(K_rounds, S_dl, B_dl, alpha_dl_min, p_dl)
    dl_eng = dl_energy(K_rounds, S_dl, B_dl, alpha_dl_min, p_dl)

    per_user_latency = per_user_latency + ul_lat + dl_lat
    per_user_energy = per_user_energy + ul_eng

    max_latency = float(np.max(per_user_latency))
    total_user_energy = float(np.sum(per_user_energy))
    total_system_energy = total_user_energy + (dl_eng if include_server_energy else 0.0)

    return {
        "max_latency": max_latency,
        "total_user_energy": total_user_energy,
        "total_system_energy": total_system_energy,
        "per_user_latency": per_user_latency,
        "per_user_energy": per_user_energy,
        "ul_latency": ul_lat,
        "ul_energy": ul_eng,
        "dl_latency": dl_lat,
        "dl_energy": dl_eng,
    }


def reward_np(J, a, b, c):
    return a * (1.0 - np.exp(-b * J)) + c


def reward_cvx(J, a, b, c):
    return a * (1.0 - cp.exp(-b * J)) + c


def plot_tradeoff_results(best_by_a1, all_results, a1_values):
    x = np.array([r["a1"] for r in best_by_a1], dtype=float)
    lat = np.array([r["best_obj_latency"] for r in best_by_a1], dtype=float)
    eng = np.array([r["best_obj_energy"] for r in best_by_a1], dtype=float)
    rew = np.array([r["best_obj_reward"] for r in best_by_a1], dtype=float)
    lyr = np.array([r["best_obj_layer"] for r in best_by_a1], dtype=int)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(x, lat, marker="o", linewidth=2)
    axes[0].set_xlabel("a1")
    axes[0].set_ylabel("Latency")
    axes[0].set_title("Best-layer latency vs a1")
    axes[0].grid(True)

    axes[1].plot(x, eng, marker="o", linewidth=2)
    axes[1].set_xlabel("a1")
    axes[1].set_ylabel("Energy")
    axes[1].set_title("Best-layer energy vs a1")
    axes[1].grid(True)

    axes[2].plot(x, rew, marker="o", linewidth=2)
    axes[2].set_xlabel("a1")
    axes[2].set_ylabel("Reward")
    axes[2].set_title("Best-layer reward vs a1")
    axes[2].grid(True)

    for i in range(len(x)):
        axes[0].annotate(f"L{lyr[i]}", (x[i], lat[i]), textcoords="offset points", xytext=(0, 8), ha="center")
        axes[1].annotate(f"L{lyr[i]}", (x[i], eng[i]), textcoords="offset points", xytext=(0, 8), ha="center")
        axes[2].annotate(f"L{lyr[i]}", (x[i], rew[i]), textcoords="offset points", xytext=(0, 8), ha="center")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    for layer in range(1, 13):
        y = [r["final_latency"] for r in all_results if r["layer"] == layer]
        plt.plot(a1_values, y, marker="o", label=f"Layer {layer}")
    plt.xlabel("a1")
    plt.ylabel("Latency")
    plt.title("Latency vs a1 for all cumulative layer choices")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    for layer in range(1, 13):
        y = [r["final_energy"] for r in all_results if r["layer"] == layer]
        plt.plot(a1_values, y, marker="o", label=f"Layer {layer}")
    plt.xlabel("a1")
    plt.ylabel("Energy")
    plt.title("Energy vs a1 for all cumulative layer choices")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    for layer in range(1, 13):
        y = [r["final_reward"] for r in all_results if r["layer"] == layer]
        plt.plot(a1_values, y, marker="o", label=f"Layer {layer}")
    plt.xlabel("a1")
    plt.ylabel("Reward")
    plt.title("Reward vs a1 for all cumulative layer choices")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()


# ============================================================
# One optimization run for one cumulative layer choice
# ============================================================
def run_one_case(layer_idx, a1_raw, func_est, base_seed=12345):
    # ========================================================
    # Settings
    # ========================================================
    N = 10
    L = 12
    SEED = base_seed

    a2_raw = 4.20
    a3_raw = 0.20

    K_rounds = 50
    J_max = 10

    gamma = 1.0
    rho_prox = 1e-2
    slack_penalty = 50.0
    max_iters = 50
    tol = 1e-6
    verbose = False

    include_server_energy = True
    rng = np.random.default_rng(SEED)

    # ========================================================
    # Parameters
    # ========================================================
    D = rng.integers(50, 100, size=N).astype(float)
    E_batt = rng.uniform(1500.0, 3000.0, size=N).astype(float)

    f_min_hz = np.full(N, 4.0e9, dtype=float)
    f_max_hz = rng.uniform(1.10e10, 1.30e10, size=N).astype(float)

    # cumulative layer selection: choose 1..layer_idx
    s = np.zeros(L, dtype=float)
    s[:layer_idx] = 1.0
    print(f"Selected layers for l={layer_idx}: {s.astype(int)}")

    C_l = np.array([5.0e4] * L).astype(float)
    Omega_l = np.array([1.66e4] * L).astype(float)
    Omega = 1.7e8

    zeta = rng.uniform(1.5e-26, 2.5e-26, size=N).astype(float)
    u = rng.uniform(1.5e6, 3.0e6, size=N).astype(float)
    bw = rng.uniform(1.1e12, 1.6e12, size=N).astype(float)
    p_gpu = rng.uniform(2.0e-11, 6.0e-11, size=N).astype(float)

    # ========================================================
    # Uplink parameters
    # ========================================================
    B_ul = 10e6
    N0 = 1e-20
    h_ul_abs2 = rng.uniform(1e-10, 8e-10, size=N).astype(float)
    alpha_ul = h_ul_abs2 / (N0 * B_ul)

    S_ul = np.array([np.sum(s) * 8000 * 32] * N).astype(float)

    p_ul_min = np.full(N, 1e-4, dtype=float)
    p_ul_max = np.full(N, 5e-1, dtype=float)

    # ========================================================
    # Downlink broadcast parameters
    # ========================================================
    B_dl = 20e6
    S_dl = np.sum(s) * 8000 * 32
    p_dl_min = 1e-3
    p_dl_max = 5.0

    h_dl_abs2 = rng.uniform(1e-10, 8e-10, size=N).astype(float)
    alpha_dl = h_dl_abs2 / (N0 * B_dl)
    alpha_dl_min = float(np.min(alpha_dl))

    # ========================================================
    # Precompute constants
    # ========================================================
    sum_omega_l = np.sum(Omega_l * s)
    c_bar = np.sum(C_l * s)

    f_scale = 1e10
    x_min = f_min_hz / f_scale
    x_max = f_max_hz / f_scale

    lat_coeff = D * c_bar
    eng_coeff = D * zeta * c_bar

    lat_coeff_s = K_rounds * lat_coeff / f_scale
    eng_coeff_s = K_rounds * eng_coeff * (f_scale ** 2)

    weights = np.array([a1_raw, a2_raw, a3_raw], dtype=float)
    weights = weights / np.sum(weights)
    a1, a2, a3 = weights

    # reward function for this cumulative layer choice
    pars = func_est[str(layer_idx)]
    r_a = float(pars["a"])
    r_b = float(pars["b"])
    r_c = float(pars["c"])

    # ========================================================
    # Initialization
    # ========================================================
    x_i = 0.5 * (x_min + x_max)
    J_i = np.ones(N, dtype=float)
    p_ul_i = 0.5 * (p_ul_min + p_ul_max)
    p_dl_i = 0.5 * (p_dl_min + p_dl_max)

    history = []

    init_metrics = evaluate_true_metrics(
        D=D,
        c_bar=c_bar,
        zeta=zeta,
        K_rounds=K_rounds,
        J=J_i,
        f_hz=x_i * f_scale,
        u=u,
        bw=bw,
        Omega=Omega,
        sum_omega_l=sum_omega_l,
        p_gpu=p_gpu,
        S_ul=S_ul,
        B_ul=B_ul,
        alpha_ul=alpha_ul,
        p_ul=p_ul_i,
        S_dl=S_dl,
        B_dl=B_dl,
        alpha_dl_min=alpha_dl_min,
        p_dl=p_dl_i,
        include_server_energy=include_server_energy,
    )

    T_ref = max(init_metrics["max_latency"], 1e-9)
    E_ref = max(
        init_metrics["total_system_energy"] if include_server_energy
        else init_metrics["total_user_energy"],
        1e-9
    )
    R_ref = max(N * (r_a + r_c), 1e-9)
    slack_ref = max(float(np.mean(E_batt)), 1e-9)

    solver_chain = []
    if hasattr(cp, "CLARABEL"):
        solver_chain.append(cp.CLARABEL)
    solver_chain.append(cp.SCS)

    # ========================================================
    # SCA loop
    # ========================================================
    for it in range(max_iters):
        J = cp.Variable(N)
        x = cp.Variable(N)
        p_ul = cp.Variable(N)
        p_dl = cp.Variable()

        tau = cp.Variable()
        s_batt = cp.Variable(N, nonneg=True)

        constraints = [
            J >= 1.0,
            J <= J_max,
            x >= x_min,
            x <= x_max,
            p_ul >= p_ul_min,
            p_ul <= p_ul_max,
            p_dl >= p_dl_min,
            p_dl <= p_dl_max,
            tau >= 0.0,
        ]

        sur_energy_terms = []

        # Common broadcast DL terms
        c_dl = K_rounds * S_dl * np.log(2.0) / B_dl
        T_dl_sur = c_dl * cp.inv_pos(cp.log1p(alpha_dl_min * p_dl))

        E_dl_true_i = dl_energy(K_rounds, S_dl, B_dl, alpha_dl_min, p_dl_i)
        E_dl_grad_i = dl_energy_grad(K_rounds, S_dl, B_dl, alpha_dl_min, p_dl_i)
        E_dl_sur = E_dl_true_i + E_dl_grad_i * (p_dl - p_dl_i)

        for n in range(N):
            # SCA surrogate for J_n / x_n
            inv_x = cp.inv_pos(x[n])
            inv_x2 = cp.square(inv_x)

            coef_J2_over_x = 1.0 / (2.0 * max(J_i[n], 1e-9) * x_i[n])
            coef_invx2 = (max(J_i[n], 1e-9) * x_i[n]) / 2.0
            g_bar_n = coef_J2_over_x * cp.square(J[n]) + coef_invx2 * inv_x2

            # SCA surrogate for J_n * x_n^2
            y_i_n = x_i[n] ** 2
            coef_J2_y = y_i_n / (2.0 * max(J_i[n], 1e-9))
            coef_y2 = max(J_i[n], 1e-9) / (2.0 * max(y_i_n, 1e-12))

            y_n = cp.square(x[n])
            y2_n = cp.square(y_n)
            h_bar_n = coef_J2_y * cp.square(J[n]) + coef_y2 * y2_n

            T_sur_n_local = (
                lat_coeff_s[n] * g_bar_n
                + (J[n] * u[n] * D[n]) / bw[n]
                + 2.0 * Omega / bw[n]
                + 2.0 * sum_omega_l / bw[n]
            )

            E_sur_n_local = (
                eng_coeff_s[n] * h_bar_n
                + J[n] * u[n] * D[n] * p_gpu[n]
                + 2.0 * Omega * p_gpu[n]
                + 2.0 * sum_omega_l * p_gpu[n]
            )

            # uplink latency
            c_ul_n = K_rounds * S_ul[n] * np.log(2.0) / B_ul
            T_ul_n = c_ul_n * cp.inv_pos(cp.log1p(alpha_ul[n] * p_ul[n]))

            # uplink energy surrogate
            E_ul_true_i = ul_energy(K_rounds, S_ul[n], B_ul, alpha_ul[n], p_ul_i[n])
            E_ul_grad_i = ul_energy_grad(K_rounds, S_ul[n], B_ul, alpha_ul[n], p_ul_i[n])
            E_ul_sur_n = E_ul_true_i + E_ul_grad_i * (p_ul[n] - p_ul_i[n])

            T_sur_n_total = T_sur_n_local + T_ul_n + T_dl_sur
            E_sur_n_total = E_sur_n_local + E_ul_sur_n

            sur_energy_terms.append(E_sur_n_total)

            constraints += [E_sur_n_total <= E_batt[n] + s_batt[n]]
            constraints += [tau >= T_sur_n_total]

        sur_total_user_energy = cp.sum(cp.hstack(sur_energy_terms))
        sur_total_energy = sur_total_user_energy + (E_dl_sur if include_server_energy else 0.0)

        reward_true_expr = cp.sum(reward_cvx(J, r_a, r_b, r_c))

        # minimize latency + energy - reward
        obj = (
            a1 * (tau / T_ref)
            + a3 * (sur_total_energy / E_ref)
            - a2 * (reward_true_expr / R_ref)
            + slack_penalty * cp.sum(s_batt) / slack_ref
            + 0.5 * rho_prox * cp.sum_squares((J - J_i) / max(J_max, 1.0))
            + 0.5 * rho_prox * cp.sum_squares((x - x_i) / np.maximum(x_max, 1e-9))
            + 0.5 * rho_prox * cp.sum_squares((p_ul - p_ul_i) / np.maximum(p_ul_max, 1e-9))
            + 0.5 * rho_prox * cp.square((p_dl - p_dl_i) / max(p_dl_max, 1e-9))
        )

        prob = cp.Problem(cp.Minimize(obj), constraints)

        solved = False
        last_err = None
        for solver in solver_chain:
            try:
                if solver == cp.SCS:
                    prob.solve(
                        solver=solver,
                        verbose=verbose,
                        eps=1e-5,
                        max_iters=200000,
                        acceleration_lookback=20,
                    )
                else:
                    prob.solve(solver=solver, verbose=verbose)

                solved = prob.status in ("optimal", "optimal_inaccurate")
                if solved:
                    break
            except Exception as e:
                last_err = e
                continue

        if not solved:
            raise RuntimeError(
                f"SCA iteration {it}: failed for layer {layer_idx}, a1={a1_raw}. "
                f"status={prob.status}. Last solver error: {last_err}"
            )

        J_star = np.asarray(J.value, dtype=float).reshape(-1)
        x_star = np.asarray(x.value, dtype=float).reshape(-1)
        p_ul_star = np.asarray(p_ul.value, dtype=float).reshape(-1)
        p_dl_star = float(p_dl.value)

        true_metrics = evaluate_true_metrics(
            D=D,
            c_bar=c_bar,
            zeta=zeta,
            K_rounds=K_rounds,
            J=J_star,
            f_hz=x_star * f_scale,
            u=u,
            bw=bw,
            Omega=Omega,
            sum_omega_l=sum_omega_l,
            p_gpu=p_gpu,
            S_ul=S_ul,
            B_ul=B_ul,
            alpha_ul=alpha_ul,
            p_ul=p_ul_star,
            S_dl=S_dl,
            B_dl=B_dl,
            alpha_dl_min=alpha_dl_min,
            p_dl=p_dl_star,
            include_server_energy=include_server_energy,
        )

        slack_val = np.asarray(s_batt.value, dtype=float).reshape(-1)

        reward_true = np.sum(reward_np(J_star, r_a, r_b, r_c)) / N

        history.append({
            "iter": it,
            "J": J_star.copy(),
            "f_hz": x_star * f_scale,
            "p_ul": p_ul_star.copy(),
            "p_dl": p_dl_star,
            "sur_max_latency": float(tau.value),
            "true_max_latency": true_metrics["max_latency"],
            "sur_total_energy": float(sur_total_energy.value),
            "true_total_energy": (
                true_metrics["total_system_energy"]
                if include_server_energy else true_metrics["total_user_energy"]
            ),
            "slack": slack_val.copy(),
            "slack_sum": float(np.sum(slack_val)),
            "obj": float(prob.value),
            "status": prob.status,
            "reward": float(reward_true),
        })

        J_next = np.clip(J_i + gamma * (J_star - J_i), 1.0, J_max)
        x_next = np.clip(x_i + gamma * (x_star - x_i), x_min, x_max)
        p_ul_next = np.clip(p_ul_i + gamma * (p_ul_star - p_ul_i), p_ul_min, p_ul_max)
        p_dl_next = float(np.clip(p_dl_i + gamma * (p_dl_star - p_dl_i), p_dl_min, p_dl_max))

        if (
            np.max(np.abs(J_next - J_i)) <= tol * max(1.0, np.max(np.abs(J_i)))
            and np.max(np.abs(x_next - x_i)) <= tol * max(1.0, np.max(np.abs(x_i)))
            and np.max(np.abs(p_ul_next - p_ul_i)) <= tol * max(1.0, np.max(np.abs(p_ul_i)))
            and abs(p_dl_next - p_dl_i) <= tol * max(1.0, abs(p_dl_i))
        ):
            J_i, x_i, p_ul_i, p_dl_i = J_next, x_next, p_ul_next, p_dl_next
            break

        J_i, x_i, p_ul_i, p_dl_i = J_next, x_next, p_ul_next, p_dl_next

    # ========================================================
    # Final results
    # ========================================================
    f_opt_hz = x_i * f_scale
    J_cont = J_i.copy()
    J_int = np.clip(np.rint(J_i).astype(int), 1, int(J_max))
    p_ul_opt = p_ul_i.copy()
    p_dl_opt = p_dl_i

    final_metrics = evaluate_true_metrics(
        D=D,
        c_bar=c_bar,
        zeta=zeta,
        K_rounds=K_rounds,
        J=J_cont,
        f_hz=f_opt_hz,
        u=u,
        bw=bw,
        Omega=Omega,
        sum_omega_l=sum_omega_l,
        p_gpu=p_gpu,
        S_ul=S_ul,
        B_ul=B_ul,
        alpha_ul=alpha_ul,
        p_ul=p_ul_opt,
        S_dl=S_dl,
        B_dl=B_dl,
        alpha_dl_min=alpha_dl_min,
        p_dl=p_dl_opt,
        include_server_energy=include_server_energy,
    )

    final_reward = np.sum(reward_np(J_cont, r_a, r_b, r_c)) / N

    return {
        "layer": layer_idx,
        "a1_raw": a1_raw,
        "weights_normalized": (a1, a2, a3),
        "selected_layers": s.astype(int).tolist(),
        "J_cont": J_cont,
        "J_int": J_int,
        "f_opt_hz": f_opt_hz,
        "p_ul_opt": p_ul_opt,
        "p_dl_opt": p_dl_opt,
        "final_obj": history[-1]["obj"],
        "final_latency": final_metrics["max_latency"],
        "final_energy": (
            final_metrics["total_system_energy"]
            if include_server_energy else final_metrics["total_user_energy"]
        ),
        "final_reward": float(final_reward),
        "history": history,
    }


if __name__ == "__main__":
    # ========================================================
    # Fitted reward functions for cumulative layers 1..l
    # ========================================================
    func_est = {
        "1": {
            "adapter_layer": 1,
            "a": 11.91573452755539,
            "b": 0.7012101523142413,
            "c": 68.2193868512529,
            "source": "curve_fit"
        },
        "2": {
            "adapter_layer": 2,
            "a": 9.956981434199879,
            "b": 0.7187578774118424,
            "c": 71.01741486334103,
            "source": "curve_fit"
        },
        "3": {
            "adapter_layer": 3,
            "a": 11.37230682686563,
            "b": 0.9465162826670389,
            "c": 70.09615611956737,
            "source": "curve_fit"
        },
        "4": {
            "adapter_layer": 4,
            "a": 11.37230682686563,
            "b": 0.9465162826670389,
            "c": 70.30287445676427,
            "delta_from_layer_3": 0.20671833719690058,
            "source": "layer_3_shifted"
        },
        "5": {
            "adapter_layer": 5,
            "a": 11.37230682686563,
            "b": 0.9465162826670389,
            "c": 70.44854640691338,
            "delta_from_layer_3": 0.3523902873460133,
            "source": "layer_3_shifted"
        },
        "6": {
            "adapter_layer": 6,
            "a": 11.37230682686563,
            "b": 0.9465162826670389,
            "c": 70.55119969518955,
            "delta_from_layer_3": 0.4550435756221911,
            "source": "layer_3_shifted"
        },
        "7": {
            "adapter_layer": 7,
            "a": 11.37230682686563,
            "b": 0.9465162826670389,
            "c": 70.62353824480824,
            "delta_from_layer_3": 0.5273821252408755,
            "source": "layer_3_shifted"
        },
        "8": {
            "adapter_layer": 8,
            "a": 11.37230682686563,
            "b": 0.9465162826670389,
            "c": 70.67451435915206,
            "delta_from_layer_3": 0.5783582395846883,
            "source": "layer_3_shifted"
        },
        "9": {
            "adapter_layer": 9,
            "a": 11.37230682686563,
            "b": 0.9465162826670389,
            "c": 70.71043661979029,
            "delta_from_layer_3": 0.6142805002229126,
            "source": "layer_3_shifted"
        },
        "10": {
            "adapter_layer": 10,
            "a": 11.37230682686563,
            "b": 0.9465162826670389,
            "c": 70.73575060901781,
            "delta_from_layer_3": 0.6395944894504405,
            "source": "layer_3_shifted"
        },
        "11": {
            "adapter_layer": 11,
            "a": 11.37230682686563,
            "b": 0.9465162826670389,
            "c": 70.75358907572972,
            "delta_from_layer_3": 0.6574329561623474,
            "source": "layer_3_shifted"
        },
        "12": {
            "adapter_layer": 12,
            "a": 11.37230682686563,
            "b": 0.9465162826670389,
            "c": 70.76615963076044,
            "delta_from_layer_3": 0.6700035111930718,
            "source": "layer_3_shifted"
        }
    }

    # choose raw a1 values to sweep
    a1_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    all_results = []
    best_by_a1 = []

    for a1_val in a1_values:
        print("\n" + "=" * 70)
        print(f"Sweeping cumulative layers for a1 = {a1_val}")
        print("=" * 70)

        layer_results = []

        for layer in range(1, 13):
            res = run_one_case(
                layer_idx=layer,
                a1_raw=a1_val,
                func_est=func_est,
                base_seed=12345
            )
            layer_results.append(res)
            all_results.append(res)

            print(
                f"Layer {layer:2d} | "
                f"selected={res['selected_layers']} | "
                f"obj={res['final_obj']:.6f} | "
                f"lat={res['final_latency']:.6f} | "
                f"eng={res['final_energy']:.6f} | "
                f"rew={res['final_reward']:.6f}"
            )

        best_obj_res = min(layer_results, key=lambda x: x["final_obj"])
        best_lat_res = min(layer_results, key=lambda x: x["final_latency"])
        best_eng_res = min(layer_results, key=lambda x: x["final_energy"])
        best_rew_res = max(layer_results, key=lambda x: x["final_reward"])

        best_by_a1.append({
            "a1": a1_val,
            "best_obj_layer": best_obj_res["layer"],
            "best_obj": best_obj_res["final_obj"],
            "best_obj_latency": best_obj_res["final_latency"],
            "best_obj_energy": best_obj_res["final_energy"],
            "best_obj_reward": best_obj_res["final_reward"],

            "min_latency_layer": best_lat_res["layer"],
            "min_latency": best_lat_res["final_latency"],

            "min_energy_layer": best_eng_res["layer"],
            "min_energy": best_eng_res["final_energy"],

            "max_reward_layer": best_rew_res["layer"],
            "max_reward": best_rew_res["final_reward"],
        })

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for row in best_by_a1:
        print(
            f"a1={row['a1']:>5} | "
            f"best_obj_layer={row['best_obj_layer']:>2} | "
            f"lat={row['best_obj_latency']:.6f} | "
            f"eng={row['best_obj_energy']:.6f} | "
            f"rew={row['best_obj_reward']:.6f} | "
            f"min_lat_layer={row['min_latency_layer']:>2} | "
            f"min_eng_layer={row['min_energy_layer']:>2} | "
            f"max_rew_layer={row['max_reward_layer']:>2}"
        )
    # ---------------- save first ----------------
    SAVE_PATH = "tradeoff_results3.json"
    save_tradeoff_data(best_by_a1, all_results, a1_values, out_path=SAVE_PATH)

    plot_tradeoff_results(best_by_a1, all_results, a1_values)