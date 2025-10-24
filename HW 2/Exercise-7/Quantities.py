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

def compute_bias_variance():
    """
    Calculate bias-variance
    """
    
    # The real function h(x) = sin(πx)
    def h(x):
        return np.sin(np.pi * x)
    
    # The range of lambda (log_10^λ from -3 to 1)
    log_lambda_range = np.linspace(-3, 1, 50)
    lambda_range = 10 ** log_lambda_range
    
    # Store results
    bias_squared_list = []
    variance_list = []
    total_error_list = []

    # Read all data files
    all_data = []
    for i in range(1, 101):
        try:
            file_path = _resolve_path(f"Exercise-7-data/data_{i}")
            x_l, y_l = _load_xy_from_file(file_path)
            all_data.append((x_l, y_l))
        except FileNotFoundError:
            print(f"Warning: data_{i} not found, skipping...")
            continue
    
    if not all_data:
        raise FileNotFoundError("No data files found!")
    
    L = len(all_data)  # The actual number of datasets
    N = all_data[0][0].shape[0]  # The number of points in each dataset (25)

    print(f"Loaded {L} datasets, each with {N} points")

    # For each lambda value
    for lam in lambda_range:
        print(f"Processing λ = {lam:.2e}")

        # Store predictions for each dataset at training points
        y_predictions = []  # Shape: L × N

        # Compute parameter estimates and predictions for each dataset
        for x_l, y_l in all_data:
            # Compute design matrix Φ
            phi = np.zeros((N, 25), dtype=float)
            for i in range(N):
                phi[i, :] = phi_x(float(x_l[i, 0]))

            # Compute parameter estimates w
            w = hat_w(phi, y_l, lam)

            # Compute predictions at training points
            y_pred = np.zeros(N)
            for i in range(N):
                y_pred[i] = phi_x(float(x_l[i, 0])) @ w
            
            y_predictions.append(y_pred)

        y_predictions = np.array(y_predictions)  # Shape: L × N

        # Compute average prediction function \bar{y}(x_n)
        y_bar = np.mean(y_predictions, axis=0)  # Shape: N

        # Compute true values h(x_n)
        x_all = all_data[0][0]  # x_n is the same for all datasets (randomly and uniformly taken from [-1,1])
        h_true = h(x_all).flatten()

        # Compute bias squared
        bias_squared = np.mean((y_bar - h_true) ** 2)
        
        # Compute Variance
        variance = 0.0
        for l in range(L):
            variance += np.mean((y_predictions[l] - y_bar) ** 2)
        variance /= L
        
        # Total error = bias squared + Variance
        total_error = bias_squared + variance
        
        bias_squared_list.append(bias_squared)
        variance_list.append(variance)
        total_error_list.append(total_error)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.plot(log_lambda_range, bias_squared_list, 'b-', linewidth=2, label=r'$(bias)^2$')
    plt.plot(log_lambda_range, variance_list, 'r-', linewidth=2, label='variance')
    plt.plot(log_lambda_range, total_error_list, 'g-', linewidth=2, label=r'$(bias)^2 + variance$')
    
    plt.xlabel(r'$\log_{10}\lambda$', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.title('Bias-Variance Decomposition', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print some key values
    print("\nKey results:")
    min_total_idx = np.argmin(total_error_list)
    print(f"Minimum total error at log10λ = {log_lambda_range[min_total_idx]:.3f}")
    print(f"  (bias)^2 = {bias_squared_list[min_total_idx]:.6f}")
    print(f"  variance = {variance_list[min_total_idx]:.6f}")
    print(f"  total = {total_error_list[min_total_idx]:.6f}")

if __name__ == "__main__":
    compute_bias_variance()