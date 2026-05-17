
### 1. Initialization & Spatial Mapping (`__init__`)
*   **Purpose:** To define the "physical world" constants and establish the interface between the high-dimensional image (784 pixels) and the low-dimensional reservoir (64 particles).
*   **Initial Implementation:** A purely random projection matrix ($W_{in}$) that mapped pixels to particles without regard for their location in the image.
*   **Issue:** **"Global Chaos."** Because pixels from opposite corners of the image were influencing the same particles, the physical interaction became a "meaningless soup" of noise that erased the digit's shape.
*   **Fix:** **Quadrant-Locked Mapping.** The image was divided into four quadrants, with specific particle groups (16 each) assigned to each. This preserved **spatial locality**, allowing the physics to extract local features like loops or lines within specific regions.

### 2. The Physics Engine (`_compute_acceleration`)
*   **Purpose:** To calculate the forces acting on every particle.
*   **Initial Implementation:** A standard Newtonian $1/r^2$ force law.
*   **Issue:** **"Numerical Explosions."** When two particles got too close, the force approached infinity, causing particles to be ejected from the simulation at near-infinite speeds (producing `NaN` errors).
*   **Fix:** **Softened Coulomb Potential.** Added an epsilon ($\epsilon$) parameter to the denominator ($r^2 + \epsilon^2$). This capped the maximum force, ensuring numerical stability even during high-density "collisions."

### 3. The Temporal Integrator (`simulate_batch` loop)
*   **Purpose:** To advance the physical state (Positions and Velocities) through time while maintaining energy stability.
*   **Initial Implementation:** Simple Euler integration ($x = x + v \cdot dt$).
*   **Issue:** **Energy Drift.** Euler integration is non-symplectic; it adds artificial energy to the system every step, causing the reservoir to "heat up" and explode regardless of the input.
*   **Fix:** **Velocity Verlet Integration.** A two-stage update for velocity and position that is "time-reversible" and conserves the system's geometric properties, keeping the "explosion" controlled and repeatable.

### 4. Input Encoding (The "Ballistic" Launch)
*   **Purpose:** To translate the 0–255 pixel values into the reservoir's initial energy.
*   **Initial Implementation:** Applying a constant electric field (driving force) to the particles throughout the entire simulation.
*   **Issue:** **Signal Masking.** The external "shaking" force was so dominant that the internal particle interactions (the actual computation) were drowned out. The physics couldn't "speak" over the input noise.
*   **Fix:** **Transient Ballistic Encoding.** The input is used only to set the **initial positions** of the particles. Once the simulation starts, the external force is zeroed. The system is left to evolve freely, meaning every movement is a pure result of internal physical "negotiation."

### 5. Physical Feature Extraction (Checkpoints & Energy)
*   **Purpose:** To convert the evolution of the particles into a format the Readout Layer can classify.
*   **Initial Implementation:** Taking only the $(x, y)$ coordinates of the particles at the very last step of the simulation.
*   **Issue:** **Fading Memory.** By the time the simulation reached the end, the damping had often "erased" the interesting interactions, leaving a static, boring state.
*   **Fix:** **Trajectory Snapshots & Potential Energy.** 
    *   **Snapshots:** Capturing the state at steps `[2, 5, 10, 20, 50]` to see the digit "unfold" over time.
    *   **Energy:** Calculating the $1/r$ Potential Energy. This is a purely nonlinear feature that only exists when particles interact ($k > 0$), providing the "Physical Advantage" over the linear baseline.

### 6. Readout Optimization (`AdamOptimizer`)
*   **Purpose:** To find the optimal weights ($W_{out}$) that map physical patterns to digit labels (0–9).
*   **Initial Implementation:** Standard Stochastic Gradient Descent (SGD).
*   **Issue:** **Stall & Jitter.** Because physical features (like potential energy) have very different numerical scales than positions, SGD struggled to converge, leading to jagged accuracy plots and slow learning.
*   **Fix:** **Adam.** Used a per-parameter learning rate with momentum. This normalized the updates, allowing the model to learn from the subtle energy signals and the large coordinate shifts simultaneously.

### 7. Evaluation & The "Scientific" Baseline
*   **Purpose:** To prove that the N-body physics actually helps.
*   **Initial Implementation:** Running the simulation with $k=1.5$ and assuming a high score meant success.
*   **Issue:** **The "Hidden Linear" Trap.** Many datasets (like MNIST) can be solved fairly well with a random linear projection. High accuracy doesn't prove the *physics* is working; it might just prove the *randomness* is working.
*   **Fix:** **The $k=0$ Control Group & Multi-Run Averaging.**
    *   Every experiment is compared against a "Ghost Reservoir" where $k=0$ (no interactions).
    *   Results are averaged over 10 runs to calculate the **"Net Stable Gain."** This proved a consistent **+0.6% to +1.5%** boost attributable solely to electromagnetic interaction.