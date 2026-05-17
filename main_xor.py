import numpy as np
import matplotlib.pyplot as plt

class NBodyReservoir:
    def __init__(self, n_particles=16, dt=0.05, steps=50):
        self.n = n_particles
        self.dt = dt
        self.steps = steps
        
        # 1. Physical Parameters
        self.k_coulomb = 2.0      # Interaction strength
        self.epsilon = 0.5        # Softening (prevents singularities)
        self.k_well = 1.2         # Confinement pull
        self.damping = 0.2        # Fading memory
        
        # 2. Particle Properties
        # Charges (N, 1), Masses (N, 1)
        self.charges = np.random.uniform(-1, 1, (self.n, 1))
        self.masses = np.random.uniform(0.8, 1.2, (self.n, 1))
        
        # 3. Input Mapping (2 inputs -> forces for N particles)
        # We map 2 inputs to a force vector (x, y) for each particle
        self.W_in = np.random.uniform(-1, 1, (2, self.n * 2))
        
        # 4. Readout Weights
        self.feature_dim = self.n * 4 # pos_x, pos_y, vel_x, vel_y
        self.W_out = np.random.normal(0, 0.1, (self.feature_dim, 1))
        self.bias = 0.0

    def _compute_acceleration(self, pos, vel, u):
        batch_size = pos.shape[0]
        # acc will store the total force per particle
        acc = np.zeros_like(pos)
        
        # A. Coulomb Forces
        # For each particle i, calculate sum of forces from all other particles j
        for i in range(self.n):
            # Distance vector from i to all others: (Batch, N, 2)
            diff = pos - pos[:, i:i+1, :] 
            
            # Distance squared: (Batch, N, 1)
            dist_sq = np.sum(diff**2, axis=-1, keepdims=True) + self.epsilon**2
            
            # Force magnitude: (Batch, N, 1)
            # F = k * q_i * q_j / r^2
            # charges[i] is scalar, charges is (N, 1)
            f_mag = (self.k_coulomb * self.charges[i] * self.charges) / (dist_sq**1.5)
            
            # Sum forces from all j acting on i: (Batch, 2)
            total_f_on_i = np.sum(diff * f_mag, axis=1)
            acc[:, i, :] = total_f_on_i

        # B. Well Confinement (Harmonic oscillator towards origin)
        acc -= self.k_well * pos
        
        # C. Damping (Friction)
        acc -= self.damping * vel
        
        # D. External Driving Force (The Input)
        # u is (B, 2), W_in is (2, N*2)
        input_force = (u @ self.W_in).reshape(batch_size, self.n, 2)
        acc += input_force
        
        return acc / self.masses.reshape(1, self.n, 1)

    def simulate(self, u_batch):
        batch_size = u_batch.shape[0]
        
        # FIX 1: Deterministic Initial State (Always start from the same place)
        # We use a fixed seed or just zeros so that for input X, output H is ALWAYS the same.
        state_rng = np.random.RandomState(42) 
        pos = state_rng.normal(0, 0.1, (batch_size, self.n, 2))
        vel = np.zeros((batch_size, self.n, 2))
        
        acc = self._compute_acceleration(pos, vel, u_batch)
        
        for _ in range(self.steps):
            v_half = vel + 0.5 * acc * self.dt
            pos = pos + v_half * self.dt
            acc = self._compute_acceleration(pos, v_half, u_batch)
            vel = v_half + 0.5 * acc * self.dt
            
        # FIX 2: Feature Normalization
        # Combine and scale so features are roughly Mean 0, Std 1
        features = np.concatenate([pos.reshape(batch_size, -1), 
                                   vel.reshape(batch_size, -1)], axis=1)
        
        return features / (np.std(features) + 1e-6)

    def train(self, inputs, targets, epochs=1200, lr=0.01):
        # FIX 3: Pre-calculate the Reservoir States
        # Since the reservoir is FIXED, we don't need to re-simulate it every epoch!
        # This makes training 1000x faster and removes the noise.
        print("Pre-simulating reservoir states...")
        H = self.simulate(inputs) 
        
        losses = []
        for epoch in range(epochs):
            logits = H @ self.W_out + self.bias
            preds = 1 / (1 + np.exp(-np.clip(logits, -10, 10)))
            
            error = preds - targets
            # BCE Loss
            loss = -np.mean(targets * np.log(preds + 1e-10) + (1-targets) * np.log(1-preds + 1e-10))
            losses.append(loss)
            
            # Gradient Descent
            grad_w = H.T @ (preds - targets)
            grad_b = np.sum(preds - targets)
            
            self.W_out -= lr * grad_w
            self.bias -= lr * grad_b
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss: {loss:.6f}")
        return losses

# Data setup
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)

# Initialize and Train
model = NBodyReservoir(n_particles=32, dt=0.04, steps=200)
model.k_coulomb = 5.0  # Stronger interaction
model.k_well = 0.2     # Weaker well
model.damping = 0.3    # Higher damping to kill chaos

history = model.train(X, Y, epochs=1200, lr=0.01)

# Results
final_h = model.simulate(X)
final_preds = 1 / (1 + np.exp(-(final_h @ model.W_out + model.bias)))

print("\nFinal Results (XOR Task):")
for i in range(4):
    print(f"In: {X[i]} | Target: {Y[i][0]} | Predicted: {final_preds[i][0]:.4f}")

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history)
plt.title("Learning Curve")
plt.xlabel("Epoch")
plt.yscale("log")
plt.ylabel("BCE Loss")

plt.subplot(1, 2, 2)
# Show the "Trajectory" spread for the 4 classes
for i in range(4):
    h = model.simulate(X[i:i+1]).flatten()
    plt.scatter([i]*len(h), h, alpha=0.3, label=f"Input {X[i]}")
plt.title("Reservoir State Distribution")
plt.xlabel("Input Class")
plt.ylabel("State Value (Pos/Vel)")
plt.tight_layout()
plt.show()


















# def get_trajectories(model, u_single):
#     """Runs a single input and returns the (Steps, N, 2) position history"""
#     # Deterministic start (must match training)
#     state_rng = np.random.RandomState(42)
#     pos = state_rng.normal(0, 0.1, (1, model.n, 2))
#     vel = np.zeros((1, model.n, 2))
    
#     history = [pos.copy()]
#     acc = model._compute_acceleration(pos, vel, u_single.reshape(1, 2))
    
#     for _ in range(model.steps):
#         v_half = vel + 0.5 * acc * model.dt
#         pos = pos + v_half * model.dt
#         acc = model._compute_acceleration(pos, vel, u_single.reshape(1, 2))
#         vel = v_half + 0.5 * acc * model.dt
#         history.append(pos.copy())
        
#     return np.array(history).squeeze(1) # (Steps, N, 2)

# # Create the visualization
# fig, axes = plt.subplots(2, 2, figsize=(12, 12))
# fig.suptitle(f"NB-ERC Particle Trajectories (N={model.n})", fontsize=16)

# xor_inputs = [
#     ([0, 0], "Target: 0"),
#     ([0, 1], "Target: 1"),
#     ([1, 0], "Target: 1"),
#     ([1, 1], "Target: 0")
# ]

# for i, (u, label) in enumerate(xor_inputs):
#     ax = axes[i//2, i%2]
#     traj = get_trajectories(model, np.array(u)) # (Steps, N, 2)
    
#     # Plot each particle's path
#     for p in range(model.n):
#         path = traj[:, p, :]
#         # Fade color from light to dark to show time
#         ax.plot(path[:, 0], path[:, 1], alpha=0.3, linewidth=1)
#         # Mark final position
#         ax.scatter(path[-1, 0], path[-1, 1], s=20, edgecolors='black', alpha=0.7)
#         # Mark start position
#         ax.scatter(path[0, 0], path[0, 1], s=5, c='red', alpha=0.5)

#     ax.set_title(f"Input: {u} | {label}")
#     ax.set_xlim(-2.5, 2.5)
#     ax.set_ylim(-2.5, 2.5)
#     ax.set_aspect('equal')
#     ax.grid(True, linestyle='--', alpha=0.6)

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()