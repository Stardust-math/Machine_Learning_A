import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

# Gaussian basis functions
def phi_x(x):
    phi_row = [1.0]
    for j in range(1, 25):
        phi_row.append(math.e ** (-(x - 0.2 * (j - 12.5)) ** 2))
    return np.array(phi_row, dtype=float)

# Parameter estimator \hat{w} = (Φ^T Φ + λ I)^(-1) Φ^T y_l
def hat_w(phi, y_l, lam):
    A = phi.T @ phi + lam * np.eye(25)
    w = np.linalg.solve(A, phi.T @ y_l)  # More stable than direct inverse
    return w  # shape: (25,1)

def _resolve_path(file_name: str) -> Path:
    """
    Search in the following order:
    1) The input name as-is
    2) Automatically add ".txt" or ".dat"
    3) The above three relative to the directory where the script is located
    4) Case-insensitive matching (in the same directory)
    If all fail, throw a detailed error and print the current working directory and the list of suspicious files.
    """
    # Run directory and script directory
    cwd = Path.cwd()
    script_dir = Path(__file__).parent if '__file__' in globals() else cwd

    candidates = []
    names = [file_name, f"{file_name}.txt", f"{file_name}.dat"]

    for base in (cwd, script_dir):
        for name in names:
            candidates.append((base / name).resolve())

    # Direct hits
    for p in candidates:
        if p.is_file():
            return p

    # Try case-insensitive matching (in the same directory)
    for base in (cwd, script_dir):
        for p in base.iterdir():
            if p.is_file() and p.name.lower() in {n.lower() for n in names}:
                return p.resolve()

    # Not found: give a friendly reminder
    msg = [
        "Data file not found. The following candidate paths have been tried:",
        *[f"- {p}" for p in candidates],
        f"\nCurrent working directory: {cwd}",
        "Example files in this directory:"
    ]
    try:
        listing = sorted([f.name for f in cwd.iterdir()][:40])
        msg += [", ".join(listing)]
    except Exception:
        pass
    raise FileNotFoundError("\n".join(msg))

def _load_xy_from_file(path: Path):
    """Read N=25 lines, each containing two numbers x y; ignore empty lines and comments starting with #.
    Supports space or comma separation.
    """
    xs, ys = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # Support both space and comma as separators
            parts = s.replace(",", " ").split()
            if len(parts) < 2:
                continue
            x, y = float(parts[0]), float(parts[1])
            xs.append(x)
            ys.append(y)
    xs = np.array(xs, dtype=float).reshape(-1, 1)
    ys = np.array(ys, dtype=float).reshape(-1, 1)
    return xs, ys

# For file_name, compute and plot parameter estimators for each lambda in lam_list
def compute_parameter_estimator(file_name, lam_list):
    # Resolve path and load data
    file_path = _resolve_path(file_name)
    x_l, y_l = _load_xy_from_file(file_path)

    if x_l.shape[0] != 25:
        print(f"Warning: Read {x_l.shape[0]} points (not 25). Code will continue with plotting.")

    # Compute Φ and \hat{w}
    N = x_l.shape[0]
    phi = np.zeros((N, 25), dtype=float)
    for i in range(N):
        phi[i, :] = phi_x(float(x_l[i, 0]))

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(x_l, y_l, marker='o', label='data', zorder=3)

    x_range = np.arange(-1, 1, 0.01)
    # Precompute all φ(x) into a matrix for speedup
    Phi_grid = np.vstack([phi_x(x) for x in x_range])  # shape: (len(x_range), 25)

    for lam in lam_list:
        w = hat_w(phi, y_l, lam)          # (25,1)
        y_pred = (Phi_grid @ w).ravel()     # (len(x_range),)
        plt.plot(x_range, y_pred, label=rf"$\lambda$ = {lam}", linewidth=1.8)

    plt.xlabel("x")
    plt.ylabel("y / prediction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    lam_list = [10, 0.1, 1e-5, 1e-10]
    file = "Exercise-7-data/data_1"
    compute_parameter_estimator(file, lam_list)