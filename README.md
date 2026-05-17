# NB-ERC: N-Body Electromagnetic Reservoir Computer

> [Article](https://www.nebuleo.org/blog/electromag_reservoir_computer/)

## What It Does
The NB-ERC is a **Physical Reservoir Computing (PRC)** experiment that replaces the traditional recurrent neural network (RNN) hidden layer with a 2D physics simulation of interacting charged particles. It uses classical Newtonian and electromagnetic dynamics to perform nonlinear feature extraction, successfully classifying MNIST digits by letting data "evolve" through simulated physics.

## How It Does It
1. **Ballistic Input Encoding:** Input data is treated as initial conditions. The 784 pixels of an MNIST digit are linearly projected to set the starting $(x,y)$ coordinates of 64 particles. 
2. **Spatial Quadrant Mapping:** To preserve image topology, the 28x28 image is split into four quadrants. Pixels in a specific quadrant only dictate the starting positions of a dedicated sub-group of 16 particles.
3. **Physics Evolution:** Once initial positions are set, external inputs are turned off. Particles evolve via Velocity Verlet integration, governed by pairwise **Coulomb interactions** ($1/r^2$ with softening), a harmonic confinement well, and damping (friction).
4. **Feature Extraction:** The system's state is sampled at specific time checkpoints (e.g., $t=2, 5, 10, 20, 50$). The features extracted are the particles' positions, velocities, and their **Inter-particle Potential Energy** (a purely physical, nonlinear metric of spatial density).
5. **Readout Training:** A simple linear readout layer is trained on these extracted physical features using the Adam optimizer to classify the digits 0-9.

## Results
To prove the physical interactions perform useful computation, the interacting reservoir ($k_c = 1.5$) was tested against a "Ghost" linear baseline where particles do not interact ($k_c = 0$). 

Averaged over 10 independent runs on a 10,000-sample MNIST subset:
* **Linear Baseline ($k=0$):** 87.87% (± 0.67%)
* **Physics Reservoir ($k=1.5$):** 88.49% (± 0.39%)
* **Net Stable Gain:** **+0.63%**

### Key Takeaways:
* **Physical Regularization:** The Coulomb interactions nearly halved the standard deviation of the accuracy (from $\pm 0.67\%$ to $\pm 0.39\%$), making the system more stable and robust to random weight initialization.
* **Topological Feature Extraction:** Differential analysis shows the physics reservoir vastly outperforms the linear baseline on digits with loops and curves (e.g., **3, 5, and 8**).