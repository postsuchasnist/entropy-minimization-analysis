import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.optimize import minimize

# ============================================================
# Problem setup
# ============================================================

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

CSV_FILE = "grid_analysis_results_var_alpha_symmetry_reduced.csv"
INTERACTIVE_X11_FILE = "interactive_x11_heatmap.html"
DENSE_TOL = 0.02


def h_p(x, p):
    eps = 1e-12
    x = np.clip(x, eps, 1 - eps)
    p = np.clip(p, eps, 1 - eps)
    return x * np.log(x / p) + (1 - x) * np.log((1 - x) / (1 - p))


def objective_var_alpha(v, p):
    alpha, x11, x12, x22 = v
    beta = 1.0 - alpha

    return (
        alpha**2 * h_p(x11, p)
        + 2.0 * alpha * beta * h_p(x12, p)
        + beta**2 * h_p(x22, p)
    )


def triangle_density_polynomial_var_alpha(v):
    alpha, x11, x12, x22 = v
    beta = 1.0 - alpha

    return (
        alpha**3 * x11**3
        + 3.0 * alpha**2 * beta * x11 * x12**2
        + 3.0 * alpha * beta**2 * x12**2 * x22
        + beta**3 * x22**3
    )


def triangle_constraint_var_alpha(v, t):
    return triangle_density_polynomial_var_alpha(v) - t**3


# ============================================================
# Solver and classification functions
# ============================================================

def classify_minimizer(
    alpha,
    x11,
    x12,
    x22,
    p,
    t,
    const_tol=1e-4,
    dense_tol=DENSE_TOL,
):
    if max(abs(x11 - t), abs(x12 - t), abs(x22 - t)) <= const_tol:
        return "constant"

    dense11 = x11 > p + dense_tol
    dense12 = x12 > p + dense_tol
    dense22 = x22 > p + dense_tol

    num_dense = int(dense11) + int(dense12) + int(dense22)

    if num_dense == 1:
        if dense11:
            return "one_dense_x11"
        if dense12:
            return "one_dense_x12"
        if dense22:
            return "one_dense_x22"

    if num_dense == 2:
        if dense11 and dense22:
            return "two_dense_x11_x22"
        if dense11 and dense12:
            return "two_dense_x11_x12"
        if dense22 and dense12:
            return "two_dense_x22_x12"

    return "other"


def solve_one_pair_var_alpha(
    p,
    t,
    alpha_eps=1e-4,
    dense_tol=DENSE_TOL,
    rng_seed=12345,
):
    bounds = [
        (alpha_eps, 0.5),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
    ]

    constraints = [
        {
            "type": "ineq",
            "fun": lambda v, t=t: triangle_constraint_var_alpha(v, t),
        }
    ]

    starts = [
        np.array([0.50, t, t, t]),
        np.array([0.40, t, t, t]),
        np.array([0.30, t, t, t]),
        np.array([0.20, t, t, t]),
        np.array([0.10, t, t, t]),
        np.array([0.50, 1.0, t, 0.0]),
        np.array([0.50, 0.0, t, 1.0]),
        np.array([0.25, 1.0, t, 0.0]),
        np.array([0.25, 0.0, t, 1.0]),
        np.array([0.50, p, p, p]),
        np.array([0.25, p, p, p]),
        np.array([0.10, p, p, p]),
        np.array([0.50, 1.0, 1.0, 1.0]),
        np.array([0.25, 1.0, 1.0, 1.0]),
        np.array([0.50, t, min(1.0, t + 0.1), t]),
        np.array([0.50, t, max(0.0, t - 0.1), t]),
        np.array([0.25, min(1.0, t + 0.2), t, max(0.0, t - 0.2)]),
        np.array([0.25, max(0.0, t - 0.2), t, min(1.0, t + 0.2)]),
    ]

    rng = np.random.default_rng(rng_seed)

    for _ in range(25):
        starts.append(
            rng.uniform(
                low=[alpha_eps, 0.0, 0.0, 0.0],
                high=[0.5, 1.0, 1.0, 1.0],
            )
        )

    best = None

    for v0 in starts:
        res = minimize(
            objective_var_alpha,
            x0=v0,
            args=(p,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": 2000,
                "ftol": 1e-12,
                "disp": False,
            },
        )

        if not res.success:
            continue

        v = np.array(
            [
                np.clip(res.x[0], alpha_eps, 0.5),
                np.clip(res.x[1], 0.0, 1.0),
                np.clip(res.x[2], 0.0, 1.0),
                np.clip(res.x[3], 0.0, 1.0),
            ]
        )

        feasibility = triangle_constraint_var_alpha(v, t)

        if feasibility < -1e-6:
            continue

        value = objective_var_alpha(v, p)

        if best is None or value < best["value"]:
            best = {
                "v": v,
                "value": value,
                "success": True,
                "message": res.message,
            }

    if best is None:
        return {
            "p": p,
            "t": t,
            "success": False,
            "alpha": np.nan,
            "x11": np.nan,
            "x12": np.nan,
            "x22": np.nan,
            "value": np.nan,
            "const_value": np.nan,
            "dist_to_constant": np.nan,
            "category": "solver_failed",
            "is_constant_minimizer": False,
        }

    alpha_star, x11_star, x12_star, x22_star = best["v"]

    dist_to_constant = max(
        abs(x11_star - t),
        abs(x12_star - t),
        abs(x22_star - t),
    )

    const_value = objective_var_alpha(
        np.array([alpha_star, t, t, t]),
        p,
    )

    category = classify_minimizer(
        alpha_star,
        x11_star,
        x12_star,
        x22_star,
        p,
        t,
        const_tol=1e-4,
        dense_tol=dense_tol,
    )

    return {
        "p": p,
        "t": t,
        "success": True,
        "alpha": alpha_star,
        "x11": x11_star,
        "x12": x12_star,
        "x22": x22_star,
        "value": best["value"],
        "const_value": const_value,
        "dist_to_constant": dist_to_constant,
        "category": category,
        "is_constant_minimizer": category == "constant",
    }


def analyze_grid_var_alpha(
    p_values,
    t_values,
    verbose=True,
    dense_tol=DENSE_TOL,
):
    rows = []
    total = sum(1 for p in p_values for t in t_values if t >= p)
    count = 0

    for p in p_values:
        for t in t_values:
            if t < p:
                continue

            count += 1

            if verbose and count % 100 == 0:
                print(f"Processed {count}/{total}")

            result = solve_one_pair_var_alpha(
                float(p),
                float(t),
                dense_tol=dense_tol,
                rng_seed=1000 + count,
            )

            rows.append(result)

    return pd.DataFrame(rows)


def print_category_counts(df):
    print("\nCategory counts:")
    print(df["category"].value_counts(dropna=False))


def print_sample_minimizers(df, num_points=20):
    cols = ["p", "t", "alpha", "x11", "x12", "x22", "value", "category"]

    print(
        df[df["success"]]
        .sort_values(["p", "t"])
        .head(num_points)[cols]
        .to_string(index=False)
    )


# ============================================================
# Plotting setup
# ============================================================

category_order = [
    "constant",
    "one_dense_x11",
    "one_dense_x12",
    "one_dense_x22",
    "two_dense_x11_x22",
    "two_dense_x11_x12",
    "two_dense_x22_x12",
    "other",
]

category_colors = {
    "constant": "tab:blue",
    "one_dense_x11": "tab:red",
    "one_dense_x12": "tab:orange",
    "one_dense_x22": "tab:green",
    "two_dense_x11_x22": "tab:purple",
    "two_dense_x11_x12": "tab:brown",
    "two_dense_x22_x12": "tab:pink",
    "other": "tab:gray",
}

category_markers = {
    "constant": "s",
    "one_dense_x11": "o",
    "one_dense_x12": "^",
    "one_dense_x22": "v",
    "two_dense_x11_x22": "D",
    "two_dense_x11_x12": "P",
    "two_dense_x22_x12": "X",
    "other": ".",
}


def save_and_close(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def successful_data(df):
    return df[df["success"] == True].copy()


# ============================================================
# Needed plotting functions
# ============================================================

def plot_x11_heatmap(df):
    df = successful_data(df)

    pivot = df.pivot(index="t", columns="p", values="x11")
    pivot = pivot.sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=[
            pivot.columns.min(),
            pivot.columns.max(),
            pivot.index.min(),
            pivot.index.max(),
        ],
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("x11")

    ax.set_xlabel("p")
    ax.set_ylabel("t")
    ax.set_title("Optimal x11 heatmap over (p,t)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    save_and_close(fig, "plot_x11_heatmap_over_pt.png")

def plot_x11_heatmap_interactive(df):
    """
    Interactive x11 heatmap over (p,t).

    Hover shows:
    - p
    - t
    - x11
    - alpha
    - objective value
    - minimizer type
    """

    df = df[df["success"] == True].copy()
    df = df.sort_values(["t", "p"])

    x11_grid = df.pivot(index="t", columns="p", values="x11")
    alpha_grid = df.pivot(index="t", columns="p", values="alpha")
    value_grid = df.pivot(index="t", columns="p", values="value")
    category_grid = df.pivot(index="t", columns="p", values="category")

    p_values = x11_grid.columns.to_numpy()
    t_values = x11_grid.index.to_numpy()

    customdata = []

    for t in t_values:
        row = []
        for p in p_values:
            row.append([
                alpha_grid.loc[t, p],
                value_grid.loc[t, p],
                category_grid.loc[t, p],
            ])
        customdata.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            x=p_values,
            y=t_values,
            z=x11_grid.values,
            customdata=customdata,
            colorscale="Viridis",
            colorbar=dict(title="x11"),
            hovertemplate=
            "p = %{x:.2f}<br>"
            "t = %{y:.2f}<br>"
            "x11 = %{z:.4f}<br>"
            "alpha = %{customdata[0]:.4f}<br>"
            "value = %{customdata[1]:.3e}<br>"
            "type = %{customdata[2]}"
            "<extra></extra>",
        )
    )

    fig.update_layout(
        title="Interactive x11 heatmap over (p,t)",
        xaxis_title="p",
        yaxis_title="t",
    )

    fig.write_html(INTERACTIVE_X11_FILE)
    fig.show()

    print(f"Saved interactive heatmap to {INTERACTIVE_X11_FILE}")


def plot_minimizer_types(df):
    df = successful_data(df)

    fig, ax = plt.subplots(figsize=(10, 7))

    for cat in category_order:
        sub = df[df["category"] == cat]

        if len(sub) == 0:
            continue

        ax.scatter(
            sub["p"],
            sub["t"],
            s=18,
            c=category_colors[cat],
            marker=category_markers[cat],
            alpha=0.75,
            label=f"{cat} ({len(sub)})",
        )

    ax.set_xlabel("p")
    ax.set_ylabel("t")
    ax.set_title("Minimizer type in (p,t)-plane")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    save_and_close(fig, "plot_minimizer_types_pt.png")


def plot_alpha_vs_t_by_type_nonconstant(df):
    df = successful_data(df)
    df = df[df["category"] != "constant"].copy()

    fig, ax = plt.subplots(figsize=(10, 7))

    for cat in category_order:
        if cat == "constant":
            continue

        sub = df[df["category"] == cat]

        if len(sub) == 0:
            continue

        ax.scatter(
            sub["t"],
            sub["alpha"],
            s=18,
            c=category_colors[cat],
            marker=category_markers[cat],
            alpha=0.75,
            label=f"{cat} ({len(sub)})",
        )

    ax.set_xlabel("t")
    ax.set_ylabel("alpha")
    ax.set_title("alpha vs t by type, excluding constant")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    save_and_close(fig, "plot_alpha_vs_t_by_type_nonconstant.png")


def plot_alpha_heatmap(df):
    df = successful_data(df)

    pivot = df.pivot(index="t", columns="p", values="alpha")
    pivot = pivot.sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=[
            pivot.columns.min(),
            pivot.columns.max(),
            pivot.index.min(),
            pivot.index.max(),
        ],
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("alpha")

    ax.set_xlabel("p")
    ax.set_ylabel("t")
    ax.set_title("Optimal alpha heatmap over (p,t)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    save_and_close(fig, "plot_alpha_heatmap_over_pt.png")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    p_values = np.round(np.arange(0.01, 1.00, 0.01), 2)
    t_values = np.round(np.arange(0.01, 1.01, 0.01), 2)

    df = analyze_grid_var_alpha(
        p_values,
        t_values,
        verbose=True,
        dense_tol=DENSE_TOL,
    )

    df.to_csv(CSV_FILE, index=False)
    print(f"Saved results to {CSV_FILE}")

    print_category_counts(df)
    print_sample_minimizers(df, num_points=20)

    plot_x11_heatmap(df)
    plot_x11_heatmap_interactive(df)
    plot_minimizer_types(df)
    plot_alpha_vs_t_by_type_nonconstant(df)
    plot_alpha_heatmap(df)

    print("\nDone. Results CSV and all requested plots saved.")
