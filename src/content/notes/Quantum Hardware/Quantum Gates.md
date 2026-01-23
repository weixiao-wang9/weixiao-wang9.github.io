---
id: 202601172132
type: source
subtype: lecture_note
course: Quantum Hardware
module:
title: Quantum Gates
status: Finished
tags: []
created: 2026-01-17
---

## 1. The Fundamental Unit: The Qubit

In classical systems, information is stored as bits (0 or 1). Quantum systems use the **qubit** (quantum bit).

### Mathematical Representation

A qubit exists in a state vector or wave function denoted as $|\psi \rangle$. It is expressed as:

$$\mid\psi\rangle = \alpha\mid0\rangle + \beta\mid1\rangle$$

- **Basis States**: $\mid0\rangle$ and $\mid1\rangle$ are the fundamental discrete states
- **Amplitudes**: $\alpha$ and $\beta$ are **complex numbers** representing probability amplitudes.

### Measurement and Duality

A qubit possesses a dual nature: it acts as an **analog** variable(any real number) before measurement (as $\alpha$ and $\beta$ vary continuously) but becomes **digital**(0/1) after measurement6.

- **Superposition**: Before being observed, the qubit exists in both states simultaneously This is illustrated by **Schrödinger’s Cat**, which is considered both "dead" and "alive" until the box is opened.
- **Probability**: The probability of collapsing to $\mid0\rangle$ is $|\alpha|^2$, and for $\mid1\rangle$ is $|\beta|^2$.
- **Normalization**: The total probability must always sum to 1 ($|\alpha|^2 + |\beta|^2 = 1$)10.
    

---

## 2. Visualizing States: The Bloch Sphere

The **Bloch Sphere** is a 3D geometric tool where any point on the surface represents a pure quantum state11.

- **Z-Axis (Vertical)**: The North Pole is $\mid0\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and the South Pole is $\mid1\rangle = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$.

- **X-Axis (Horizontal Equator)**: Represents the **Hadamard Basis**. $\mid+\rangle = \frac{\mid0\rangle + \mid1\rangle}{\sqrt{2}}$ is at the positive x-axis, and $\mid-\rangle = \frac{\mid0\rangle - \mid1\rangle}{\sqrt{2}}$ is at the negative x-axis.
    
- **Y-Axis (Depth Equator)**: Includes complex phases. $\mid+i\rangle = \frac{\mid0\rangle + i\mid1\rangle}{\sqrt{2}}$ (positive y) and $\mid-i\rangle = \frac{\mid0\rangle - i\mid1\rangle}{\sqrt{2}}$ (negative y).

    
- General State: Defined by angles $\theta$ (latitude) and $\phi$ (longitude)15:
    
    $$\mid\psi\rangle = \cos\frac{\theta}{2}\mid0\rangle + e^{i\phi}\sin\frac{\theta}{2}\mid1\rangle$$
## 3. Quantum Gates

Quantum operations are represented by **unitary matrices** that rotate the state vector on the Bloch sphere.


### Single-Qubit Gates

- **X-Gate (Pauli-X)**: Acts as a **bit-flip** (the quantum NOT gate), mapping $\mid0\rangle \to \mid1\rangle$. It rotates the sphere 180° around the X-axis. e.g. If the input state is $\mid A \rangle = \alpha\mid0\rangle + \beta\mid1\rangle$, the output becomes $\mid B \rangle = \beta\mid0\rangle + \alpha\mid1\rangle$
>$X= \begin{bmatrix} 0 &1 \\ 1 & 0 \end{bmatrix}$
    
- **Z-Gate (Pauli-Z)**: A **phase-flip** gate. It leaves $\mid0\rangle$ unchanged but flips the sign of $\mid1\rangle$ ($\mid1\rangle \to -\mid1\rangle$). It rotates the sphere 180° around the Z-axis.
>$Z= \begin{bmatrix} 1 &0 \\ 0 & -1 \end{bmatrix}$

- **Hadamard (H) Gate**: The most critical gate for **superposition**. it transforms basis states ($\mid0\rangle, \mid1\rangle$) into equatorial states ($\mid+\rangle, \mid-\rangle$). 
>$H= \frac{1}{\sqrt{ 2 }}\begin{bmatrix} 1 &1 \\ 1 & -1 \end{bmatrix}$


- **S and T Gates**: Phase-shifting gates. The **S-gate** rotates 90° ($\pi/2$) maps $\mid +\rangle$ to $\mid i \rangle$and the **T-gate** rotates 45° ($\pi/4$) around the Z-axis.
>$S= \begin{bmatrix} 1 &0 \\ 0 & i \end{bmatrix}$
>$T= \begin{bmatrix} 1 &0 \\ 0 & \exp\left( \frac{i\pi}{4} \right) \end{bmatrix}$



### Two Qubit Gates

- **CNOT (Controlled NOT)**: Flips the second qubit only if the first qubit is $\mid1\rangle$. This is the primary tool for creating **entanglement**.
![[Screenshot 2026-01-17 at 10.52.11 PM.png]]
> $CNOT = \begin{bmatrix} 1& 0 & 0 & 0\\0&1&0&0\\0&0&0&1\\0&0&1&0\end{bmatrix}$
- **SWAP Gate**: Exchanges the states of two qubits, often implemented using three CNOT gates. It used to move qubits under ***restricted connectivity***.
![[Screenshot 2026-01-17 at 10.56.13 PM.png]]
>$SWAP = \begin{bmatrix} 1& 0 & 0 & 0\\0&0&1&0\\0&1&0&0\\0&0&0&1\end{bmatrix}$
- **Toffoli (CCNOT)**: A three-qubit gate where the third qubit flips only if the first two are both $\mid1\rangle$.
![[Screenshot 2026-01-17 at 10.58.00 PM.png]]
    
- **Fredkin (CSWAP)**: Swaps the second and the thrid bit iff the control bit is $\mid1\rangle$
 ![[Screenshot 2026-01-17 at 10.58.27 PM.png]]   

---

## 4. Governing Principles

Quantum computing is restricted by laws that do not apply to classical systems.
### Reversibility
All quantum operations must be **reversible**; information cannot be destroyed. If you have the output state, you must be able to mathematically reconstruct the exact input state.

### No-Cloning Theorem
It is physically impossible to create an identical, independent copy of an unknown quantum state.

- **Entanglement vs. Copying**: Using a CNOT gate on a qubit and a blank $\mid0\rangle$ qubit results in entanglement, not a clone.
- **Teleportation**: You can transfer a state to another qubit, but the original state is destroyed in the process