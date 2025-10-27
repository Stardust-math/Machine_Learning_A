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

def compute_parameter_estimator_for_all_datasets(lam_list):
    # Create list of data file names
    file_list = [f"Exercise-7-data/data_{i}" for i in range(1, 26)]
    
    # For each lambda value, plot the fitting results for all datasets
    for lam in lam_list:
        plt.figure(figsize=(12, 8))

        # Fit each dataset and plot
        for file_idx, file_name in enumerate(file_list):
            try:
                # Load data
                file_path = _resolve_path(file_name)
                x_l, y_l = _load_xy_from_file(file_path)

                if x_l.shape[0] != 25:
                    print(f"Warning: {file_name} has {x_l.shape[0]} points (not 25).")

                # Compute Φ matrix
                N = x_l.shape[0]
                phi = np.zeros((N, 25), dtype=float)
                for i in range(N):
                    phi[i, :] = phi_x(float(x_l[i, 0]))

                # Compute fitting curve
                x_range = np.arange(-1, 1, 0.01)
                Phi_grid = np.vstack([phi_x(x) for x in x_range])
                
                w = hat_w(phi, y_l, lam)
                y_pred = (Phi_grid @ w).ravel()

                # Plot fitting curve with different colors for each dataset
                plt.plot(x_range, y_pred, label=f'Dataset {file_idx+1}', linewidth=1.5, alpha=0.7)

            except FileNotFoundError:
                print(f"Warning: File {file_name} not found, skipping.")
                continue
        
        plt.xlabel("x")
        plt.ylabel("y / prediction")
        plt.title(f"Fitting Results for All Datasets (λ = {lam})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend on the right
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    lam_list = [10, 0.1, 1e-5, 1e-10]
    compute_parameter_estimator_for_all_datasets(lam_list)
