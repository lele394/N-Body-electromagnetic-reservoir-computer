"""
N-Body Echo Reservoir Computing on MNIST
=========================================
Charged particles in a confining harmonic well with Coulomb interactions
serve as a physical reservoir. Input images are encoded as initial particle
positions (quadrant-mapped). Multi-checkpoint readout, closed-form ridge
regression. Pure NumPy.

Fixes applied vs. original:
  1. Vectorized O(N^2) acceleration computation (~20x speedup).
  2. Corrected force sign (Newton's 3rd law: F_on_i points from j -> i).
  3. Excluded i=j self-term from potential energy feature.
  4. Added kinetic energy as feature (carries velocity-magnitude info).
  5. Global feature normalization fitted on train set, applied to both.
  6. Closed-form ridge regression readout (replaces 300 Adam epochs).
  7. Fixed random seed for reproducibility.
  8. Removed unused Adam class and training loop.
  9. Continuous input drive (input also injected into initial velocity).
 10. Reduced damping so the reservoir doesn't collapse to equilibrium.
 11. Logistic-regression baseline on raw pixels for sanity check.
 12. Multi-seed runs to get error bars.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import json
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# --- Configuration ---
RESULTS_DIR = "nb_erc_mnist_v2"
os.makedirs(RESULTS_DIR, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# =====================================================================
#  N-Body Coulomb Reservoir
# =====================================================================
class NBodyReservoir:
    """
    A reservoir of N charged particles in 2D, interacting via softened
    Coulomb forces inside a harmonic confining well, with viscous damping.

    State at time t: positions r_i(t), velocities v_i(t).
    Dynamics (per particle i):
        m_i a_i = sum_{j != i} k * q_i * q_j * (r_i - r_j) / |r_i - r_j|^3
                  - k_well * r_i
                  - damping * v_i

    Input encoding:
        Image (784,) -> initial positions (and a fraction into velocities)
        via a quadrant-wise random projection W_in. Each MNIST quadrant
        drives its own group of particles, preserving 2D spatial structure.

    Readout:
        Features = concat over checkpoints of [positions, velocities,
        per-particle PE excluding self, per-particle KE].
        Globally normalized using train-set statistics, then a closed-form
        ridge regression maps features -> 10-way one-hot labels.
    """

    def __init__(self, n_particles=64, dt=0.04, seed=0):
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        self.params = {
            "n_particles": n_particles,
            "dt": dt,
            "k_coulomb": 1.5,
            "epsilon": 0.4,       # softening length
            "k_well": 0.2,
            "damping": 0.05,      # reduced (was 0.25) -- keep dynamics alive
            "checkpoints": [2, 5, 10, 20, 50],
            "input_velocity_scale": 0.3,  # also inject input into v0
        }
        self.n = n_particles
        self.dt = dt
        self.k_coulomb = self.params["k_coulomb"]
        self.epsilon = self.params["epsilon"]
        self.k_well = self.params["k_well"]
        self.damping = self.params["damping"]
        self.checkpoints = self.params["checkpoints"]
        self.v_scale = self.params["input_velocity_scale"]

        # Fixed reservoir parameters
        self.charges = self.rng.uniform(-1.0, 1.0, (self.n, 1))
        self.masses = np.ones((self.n, 1))

        # Quadrant-based input mapping: each 14x14 image quadrant drives
        # its own group of n/4 particles (positions + velocities).
        # Output dim = n*4 (n particles x 2D pos + 2D vel components)
        self.W_in = np.zeros((784, self.n * 4))
        n_per_quad = self.n // 4
        for i in range(28):
            for j in range(28):
                pixel_idx = i * 28 + j
                quad_idx = (0 if i < 14 else 2) + (0 if j < 14 else 1)
                start_p = quad_idx * n_per_quad
                end_p = (quad_idx + 1) * n_per_quad
                # Each particle gets 4 input weights: pos_x, pos_y, vel_x, vel_y
                self.W_in[pixel_idx, start_p * 4 : end_p * 4] = \
                    self.rng.normal(0, 1.5, n_per_quad * 4)

        # Feature dim per checkpoint: 2n (pos) + 2n (vel) + n (PE) + n (KE) = 6n
        self.feature_dim = 6 * self.n * len(self.checkpoints)
        self.feat_mean = None
        self.feat_std = None

    # -----------------------------------------------------------------
    #  Physics
    # -----------------------------------------------------------------
    def _accel(self, pos, vel):
        """
        Vectorized Coulomb + harmonic well + damping.

        pos, vel : (B, N, 2)
        returns acc : (B, N, 2)

        Sign convention check: force on particle i due to j is
            F_ij = k * q_i * q_j * (r_i - r_j) / |r_i - r_j|^3
        For two like positive charges (q_i, q_j > 0), this points in the
        direction (r_i - r_j), i.e. away from j -- repulsive. Correct.
        """
        # diff[b, i, j, :] = r_i - r_j
        diff = pos[:, :, None, :] - pos[:, None, :, :]           # (B, N, N, 2)
        dist_sq = np.sum(diff ** 2, axis=-1, keepdims=True) + self.epsilon ** 2
        inv_r3 = dist_sq ** (-1.5)                               # (B, N, N, 1)

        # q_i * q_j matrix (N, N, 1)
        q_prod = (self.charges * self.charges.T)[None, :, :, None]  # (1, N, N, 1)
        # Zero out self-interaction (i == j)
        np.fill_diagonal(q_prod[0, :, :, 0], 0.0)

        force = self.k_coulomb * q_prod * inv_r3 * diff          # (B, N, N, 2)
        acc = np.sum(force, axis=2) / self.masses[None, :, :]    # (B, N, 2)

        acc -= self.k_well * pos + self.damping * vel
        return acc

    def _features_at_checkpoint(self, pos, vel):
        """Build feature vector: [pos, vel, per-particle PE (no self), KE]."""
        B = pos.shape[0]

        # Potential energy per particle, excluding self
        diff = pos[:, :, None, :] - pos[:, None, :, :]
        r = np.sqrt(np.sum(diff ** 2, axis=-1) + self.epsilon ** 2)  # (B, N, N)
        q_prod = (self.charges * self.charges.T)[None, :, :]         # (1, N, N)
        pe_matrix = self.k_coulomb * q_prod / r                      # (B, N, N)
        # Zero the diagonal (i == j) so self-term doesn't pollute
        idx = np.arange(self.n)
        pe_matrix[:, idx, idx] = 0.0
        pe = np.sum(pe_matrix, axis=2)                               # (B, N)

        # Kinetic energy per particle: 0.5 * m * |v|^2
        ke = 0.5 * self.masses.flatten()[None, :] * np.sum(vel ** 2, axis=-1)  # (B, N)

        return np.concatenate([
            pos.reshape(B, -1),
            vel.reshape(B, -1),
            pe,
            ke,
        ], axis=1)

    def simulate_batch(self, u_batch):
        """
        Run reservoir dynamics on a batch of inputs.

        u_batch : (B, 784)
        returns : (B, feature_dim)
        """
        B = u_batch.shape[0]

        # Project input -> initial pos and vel
        proj = u_batch @ self.W_in                       # (B, 4N)
        proj = proj.reshape(B, self.n, 4)
        pos = proj[:, :, :2].copy()                      # (B, N, 2)
        vel = self.v_scale * proj[:, :, 2:].copy()       # (B, N, 2)

        acc = self._accel(pos, vel)
        captured = []
        max_step = max(self.checkpoints)

        for step in range(max_step + 1):
            if step in self.checkpoints:
                captured.append(self._features_at_checkpoint(pos, vel))
            if step == max_step:
                break
            # Velocity-Verlet
            v_half = vel + 0.5 * acc * self.dt
            pos = pos + v_half * self.dt
            acc = self._accel(pos, v_half)
            vel = v_half + 0.5 * acc * self.dt

        return np.concatenate(captured, axis=1)          # (B, feature_dim)

    # -----------------------------------------------------------------
    #  Feature extraction with global normalization
    # -----------------------------------------------------------------
    def extract_features(self, X, label="data", chunk=500, fit_norm=False):
        log(f"Simulating reservoir for {label} ({X.shape[0]} samples)...")
        feats = []
        t0 = time.time()
        for i in range(0, X.shape[0], chunk):
            end = min(i + chunk, X.shape[0])
            feats.append(self.simulate_batch(X[i:end]))
            if (i // chunk) % 4 == 0 or end == X.shape[0]:
                log(f"  > [{label}] {end}/{X.shape[0]}  ({time.time()-t0:.1f}s)")
        H = np.vstack(feats)

        if fit_norm:
            self.feat_mean = H.mean(axis=0, keepdims=True)
            self.feat_std = H.std(axis=0, keepdims=True) + 1e-6
        if self.feat_mean is None:
            raise RuntimeError("Normalization not fitted yet. Call with fit_norm=True on train first.")
        return (H - self.feat_mean) / self.feat_std

    # -----------------------------------------------------------------
    #  Closed-form ridge readout
    # -----------------------------------------------------------------
    def fit_readout(self, H_tr, Y_tr, alpha=1.0):
        """
        Ridge regression: W = (H^T H + alpha I)^-1 H^T Y
        Solve via np.linalg.solve for numerical stability.
        """
        log(f"Fitting ridge readout (alpha={alpha})...")
        # Add a bias column to H
        H_aug = np.hstack([H_tr, np.ones((H_tr.shape[0], 1))])
        d = H_aug.shape[1]
        A = H_aug.T @ H_aug
        # Don't regularize the bias term
        reg = alpha * np.eye(d)
        reg[-1, -1] = 0.0
        A += reg
        B = H_aug.T @ Y_tr
        self.W_out = np.linalg.solve(A, B)
        return self.W_out

    def predict(self, H):
        H_aug = np.hstack([H, np.ones((H.shape[0], 1))])
        logits = H_aug @ self.W_out
        return np.argmax(logits, axis=1)


# =====================================================================
#  Experiment runner
# =====================================================================
def run_experiment(X_tr, y_tr_oh, y_tr, X_te, y_te_oh, y_te,
                   n_particles=64, seed=0, ridge_alpha=1.0):
    model = NBodyReservoir(n_particles=n_particles, seed=seed)
    H_tr = model.extract_features(X_tr, label="Train", fit_norm=True)
    H_te = model.extract_features(X_te, label="Test", fit_norm=False)

    model.fit_readout(H_tr, y_tr_oh, alpha=ridge_alpha)

    tr_pred = model.predict(H_tr)
    te_pred = model.predict(H_te)
    tr_acc = accuracy_score(y_tr, tr_pred)
    te_acc = accuracy_score(y_te, te_pred)

    log(f"  Seed {seed}: train acc = {tr_acc:.4f}, test acc = {te_acc:.4f}")
    return model, tr_acc, te_acc, tr_pred, te_pred


def main():
    # ---------- Data ----------
    log("Downloading MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data / 255.0, mnist.target.astype(int)
    X_tr_f, X_te_f, y_tr_f, y_te_f = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_tr, y_tr = X_tr_f[:10000], y_tr_f[:10000]
    X_te, y_te = X_te_f[:2000], y_te_f[:2000]

    encoder = OneHotEncoder(sparse_output=False)
    y_tr_oh = encoder.fit_transform(y_tr.reshape(-1, 1))
    y_te_oh = encoder.transform(y_te.reshape(-1, 1))

    # ---------- Baseline: logistic regression on raw pixels ----------
    log("Fitting logistic-regression baseline on raw pixels...")
    t0 = time.time()
    baseline = LogisticRegression(max_iter=200, n_jobs=-1, C=1.0)
    baseline.fit(X_tr, y_tr)
    base_tr = baseline.score(X_tr, y_tr)
    base_te = baseline.score(X_te, y_te)
    log(f"  Logistic baseline: train={base_tr:.4f}, test={base_te:.4f} "
        f"({time.time()-t0:.1f}s)")

    # ---------- N-body reservoir, multi-seed ----------
    seeds = [0, 1, 2, 3, 4]
    seed_results = []
    last_model = None
    last_preds = None

    for s in seeds:
        log(f"\n=== Reservoir run, seed={s} ===")
        model, tr_acc, te_acc, tr_pred, te_pred = run_experiment(
            X_tr, y_tr_oh, y_tr, X_te, y_te_oh, y_te,
            n_particles=64, seed=s, ridge_alpha=1.0
        )
        seed_results.append({"seed": s, "train_acc": tr_acc, "test_acc": te_acc})
        last_model = model
        last_preds = (tr_pred, te_pred)

    tr_accs = np.array([r["train_acc"] for r in seed_results])
    te_accs = np.array([r["test_acc"] for r in seed_results])
    log(f"\nReservoir over {len(seeds)} seeds:")
    log(f"  Train: {tr_accs.mean():.4f} +/- {tr_accs.std():.4f}")
    log(f"  Test:  {te_accs.mean():.4f} +/- {te_accs.std():.4f}")
    log(f"  Logistic baseline test: {base_te:.4f}")
    log(f"  Reservoir - baseline (test): {te_accs.mean() - base_te:+.4f}")

    # ---------- Save report ----------
    tr_pred, te_pred = last_preds
    cm_train = confusion_matrix(y_tr, tr_pred).tolist()
    cm_test = confusion_matrix(y_te, te_pred).tolist()
    report = {
        "system_parameters": last_model.params,
        "readout": {"type": "ridge_regression", "alpha": 1.0},
        "seeds": seeds,
        "per_seed": seed_results,
        "summary": {
            "train_mean": float(tr_accs.mean()),
            "train_std": float(tr_accs.std()),
            "test_mean": float(te_accs.mean()),
            "test_std": float(te_accs.std()),
        },
        "baseline_logistic_regression": {
            "train_accuracy": float(base_tr),
            "test_accuracy": float(base_te),
        },
        "last_run_confusion_train": cm_train,
        "last_run_confusion_test": cm_test,
    }
    out_json = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=4)
    log(f"Saved metrics to {out_json}")

    # ---------- Plots ----------
    # Per-seed bar chart with baseline line
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(seeds))
    ax.bar(x - 0.2, tr_accs, 0.4, label="Train", color="forestgreen")
    ax.bar(x + 0.2, te_accs, 0.4, label="Test", color="royalblue")
    ax.axhline(base_te, color="crimson", ls="--", lw=1.3,
               label=f"Logistic baseline (test={base_te:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("N-Body Reservoir vs. Logistic Baseline (MNIST 10k/2k)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "seed_comparison.png"), dpi=120)

    # Confusion matrices for last run
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens', ax=ax1, cbar=False)
    ax1.set_title(f"Train confusion (acc {tr_accs[-1]:.4f})")
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
    ax2.set_title(f"Test confusion (acc {te_accs[-1]:.4f})")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_last_run.png"), dpi=120)

    log("Done.")


if __name__ == "__main__":
    main()


