import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import json
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --- Configuration & Logging ---
RESULTS_DIR = "nb_erc_mnist_production"
os.makedirs(RESULTS_DIR, exist_ok=True)

def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

# --- Optimizer ---
class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr, self.beta1, self.beta2, self.epsilon = lr, beta1, beta2, epsilon
        self.m, self.v, self.t = {}, {}, 0

    def step(self, param, grad, param_name):
        if param_name not in self.m:
            self.m[param_name] = np.zeros_like(param)
            self.v[param_name] = np.zeros_like(param)
        self.t += 1
        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad**2)
        m_hat = self.m[param_name] / (1 - self.beta1**self.t)
        v_hat = self.v[param_name] / (1 - self.beta2**self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return param

# --- Quadrant-Based N-Body Reservoir ---
class NB_ERC_MNIST:
    def __init__(self, n_particles=64, dt=0.04):
        self.params = {
            "n_particles": n_particles,
            "dt": dt,
            "k_coulomb": 1.5,
            "epsilon": 0.4,
            "k_well": 0.2,
            "damping": 0.25,
            "checkpoints": [2, 5, 10, 20, 50],
            "train_size": 10000,
            "test_size": 2000
        }
        
        self.train_params = {}

        self.n = n_particles
        self.dt = dt
        self.k_coulomb = self.params["k_coulomb"]
        self.epsilon = self.params["epsilon"]
        self.k_well = self.params["k_well"]
        self.damping = self.params["damping"]
        self.checkpoints = self.params["checkpoints"]
        
        self.charges = np.random.uniform(-1, 1, (self.n, 1))
        self.masses = np.ones((self.n, 1))
        
        # --- Quadrant Spatial Mapping ---
        self.W_in = np.zeros((784, self.n * 2))
        n_per_quad = self.n // 4
        for i in range(28):
            for j in range(28):
                pixel_idx = i * 28 + j
                quad_idx = (0 if i < 14 else 2) + (0 if j < 14 else 1)
                start_p = quad_idx * n_per_quad
                end_p = (quad_idx + 1) * n_per_quad
                self.W_in[pixel_idx, start_p*2 : end_p*2] = np.random.normal(0, 1.5, n_per_quad * 2)

        self.feature_dim = (self.n * 4 + self.n) * len(self.checkpoints)
        self.W_out = np.random.normal(0, 0.01, (self.feature_dim, 10))
        self.bias = np.zeros((1, 10))

    def _compute_acceleration(self, pos, vel):
        batch_size = pos.shape[0]
        acc = np.zeros_like(pos)
        for i in range(self.n):
            diff = pos - pos[:, i:i+1, :] 
            dist_sq = np.sum(diff**2, axis=-1, keepdims=True) + self.epsilon**2
            f_mag = (self.k_coulomb * self.charges[i] * self.charges) / (dist_sq**1.5)
            acc[:, i, :] = np.sum(diff * f_mag, axis=1)
        acc -= (self.k_well * pos + self.damping * vel)
        return acc

    def simulate_batch(self, u_batch):
        batch_size = u_batch.shape[0]
        pos = (u_batch @ self.W_in).reshape(batch_size, self.n, 2)
        vel = np.zeros((batch_size, self.n, 2))
        captured_features = []
        acc = self._compute_acceleration(pos, vel)
        
        for step in range(max(self.checkpoints) + 1):
            if step in self.checkpoints:
                dyn = np.concatenate([pos.reshape(batch_size, -1), vel.reshape(batch_size, -1)], axis=1)
                energy = np.zeros((batch_size, self.n))
                for i in range(self.n):
                    diff = pos - pos[:, i:i+1, :]
                    r = np.sqrt(np.sum(diff**2, axis=-1) + self.epsilon**2)
                    energy[:, i] = np.sum(self.k_coulomb * self.charges[i] * self.charges.T / r, axis=1)
                captured_features.append(np.concatenate([dyn, energy], axis=1))
            
            v_half = vel + 0.5 * acc * self.dt
            pos = pos + v_half * self.dt
            acc = self._compute_acceleration(pos, v_half)
            vel = v_half + 0.5 * acc * self.dt
            
        full = np.concatenate(captured_features, axis=1)
        return (full - np.mean(full)) / (np.std(full) + 1e-6)

    def train(self, X_tr, y_tr, X_te, y_te, epochs=300, batch_size=32, lr=0.005):

        self.train_params = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr
        }

        num_tr = X_tr.shape[0]
        num_te = X_te.shape[0]

        def get_all_features(data, label="Dataset"):
            log(f"Starting Physics Simulation for {label} ({data.shape[0]} samples)...")
            feats = []
            chunk = 500
            start_time = time.time()
            for i in range(0, data.shape[0], chunk):
                end = min(i + chunk, data.shape[0])
                feats.append(self.simulate_batch(data[i:end]))
                elapsed = time.time() - start_time
                log(f"  > [{label}] Processed {end}/{data.shape[0]} | Elapsed: {elapsed:.1f}s")
            return np.vstack(feats)

        H_tr = get_all_features(X_tr, "Train")
        H_te = get_all_features(X_te, "Test")
        
        log("Optimizing Readout Layer with Adam...")
        optimizer = AdamOptimizer(lr=lr)
        tr_acc_h, te_acc_h, loss_h = [], [], []

        for epoch in range(epochs):
            indices = np.random.permutation(num_tr)
            epoch_loss = 0
            for i in range(0, num_tr, batch_size):
                idx = indices[i:i+batch_size]
                logits = H_tr[idx] @ self.W_out + self.bias
                preds = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
                
                loss = -np.mean(np.sum(y_tr[idx] * np.log(preds + 1e-10), axis=1))
                epoch_loss += loss
                
                error = (preds - y_tr[idx]) / batch_size
                self.W_out = optimizer.step(self.W_out, H_tr[idx].T @ error + 0.001*self.W_out, "w")
                self.bias = optimizer.step(self.bias, np.sum(error, axis=0, keepdims=True), "b")
            
            t_acc = accuracy_score(np.argmax(y_tr, 1), np.argmax(H_tr @ self.W_out + self.bias, 1))
            v_acc = accuracy_score(np.argmax(y_te, 1), np.argmax(H_te @ self.W_out + self.bias, 1))
            tr_acc_h.append(float(t_acc))
            te_acc_h.append(float(v_acc))
            loss_h.append(float(epoch_loss / (num_tr // batch_size)))
            
            if (epoch + 1) % 50 == 0 or epoch == 0:
                log(f"  Ep {epoch+1:3d}/{epochs} | Loss: {loss_h[-1]:.4f} | Tr Acc: {t_acc:.4f} | Te Acc: {v_acc:.4f}")
        
        return loss_h, tr_acc_h, te_acc_h, H_tr, H_te

# --- Execution ---
log("Downloading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data / 255.0, mnist.target.astype(int)
X_tr_f, X_te_f, y_tr_f, y_te_f = train_test_split(X, y, test_size=0.2, random_state=42)

# Subsets
X_tr, y_tr = X_tr_f[:10000], y_tr_f[:10000]
X_te, y_te = X_te_f[:2000], y_te_f[:2000]

encoder = OneHotEncoder(sparse_output=False)
y_tr_oh = encoder.fit_transform(y_tr.reshape(-1,1))
y_te_oh = encoder.transform(y_te.reshape(-1,1))

model = NB_ERC_MNIST(n_particles=64)
loss_h, tr_acc_h, te_acc_h, H_tr, H_te = model.train(X_tr, y_tr_oh, X_te, y_te_oh, epochs=300, batch_size=128, lr=0.001)

# --- Compute Final Results for JSON ---
tr_preds = np.argmax(H_tr @ model.W_out + model.bias, 1)
te_preds = np.argmax(H_te @ model.W_out + model.bias, 1)
cm_train = confusion_matrix(y_tr, tr_preds).tolist()
cm_test = confusion_matrix(y_te, te_preds).tolist()

# --- Build and Save JSON ---
report_data = {
    "system_parameters": model.params,
    "training_parameters": model.train_params,
    "training_history": {
        "loss": loss_h,
        "train_accuracy": tr_acc_h,
        "test_accuracy": te_acc_h
    },
    "final_results": {
        "train_accuracy": tr_acc_h[-1],
        "test_accuracy": te_acc_h[-1],
        "confusion_matrix_train": cm_train,
        "confusion_matrix_test": cm_test
    }
}

with open(f"{RESULTS_DIR}/experiment_results.json", "w") as f:
    json.dump(report_data, f, indent=4)
log(f"Exported metrics to {RESULTS_DIR}/experiment_results.json")

# --- Plotting ---
log("Generating Figures...")
plt.style.use('default')

# 1. Accuracy & Log Loss
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
ax1.plot(tr_acc_h, label='Train Acc', color='forestgreen', lw=1.5)
ax1.plot(te_acc_h, label='Test Acc', color='royalblue', lw=1.5)
ax2.plot(loss_h, label='Log Loss', color='crimson', ls='--', alpha=0.6)
ax2.set_yscale('log')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax2.set_ylabel('Loss (Log)')
ax1.legend(loc='lower left'); ax2.legend(loc='lower right')
plt.title("NB-ERC MNIST Ballistic Reservoir History")
plt.savefig(f"{RESULTS_DIR}/training_history.png")

# 2. Confusion Matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens', ax=ax1, cbar=False)
ax1.set_title(f"Train Confusion (Acc: {tr_acc_h[-1]:.4f})")
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
ax2.set_title(f"Test Confusion (Acc: {te_acc_h[-1]:.4f})")
plt.savefig(f"{RESULTS_DIR}/confusion_comparison.png")

log("All tasks complete.")
plt.show()