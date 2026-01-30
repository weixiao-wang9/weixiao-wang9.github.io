---
id: 202601180210
type: source
subtype: lecture_note
course: Quantum Hardware
module:
title: Quantum Mechanics
status: Finished
tags: []
created: 2026-01-18
---



## 1. Wave-Particle Duality and History

The core of quantum mechanics is the realization that fundamental units of nature do not behave exclusively as solid "bullets" or continuous waves.

- **Particle Theories**: Dating back to Democritus in the 4th century BC, the idea was that matter is made of indivisible units. Historically, these were called atoms, but today we use the term "particles" for truly indivisible units like electrons.
    
- **Wave Theories**: After Young’s double-slit experiment, "wave optics" became the dominant view of light because particles were not expected to diffract or interfere.
    
- **The Quantum Shift**:
    
    - **1900**: Max Planck proposed light is "quantized" into discrete packets now called photons.
    - **1905**: Einstein confirmed this by using quantized light to explain the photoelectric effect.
        
- **Duality**: We now accept that light and matter have both particle-like properties (they are indivisible units) and wave-like properties (they exhibit diffraction).
    

---

## 2. Matter Waves and the Schrödinger Equation

In 1924, Louis de Broglie proposed that matter, like light, obeys a wave equation. Two years later, Erwin Schrödinger formulated the famous wave equation for matter:


$$i\hbar\frac{\partial\psi}{\partial t}=-\frac{\hbar^{2}}{2m}\frac{\partial^{2}\psi}{\partial x^{2}}+V(x)\psi$$


- **Structure**: It is a complex equation (having real and imaginary parts).
    
- **Differences from Classical Waves**: Unlike light or sound equations, it has only a single time derivative and includes a **potential energy term** ($V(x)$) describing how matter interacts with its environment.
    
- **Wavelength-Momentum Relation**: Using Planck's constant ($h \approx 6.626 \times 10^{-34} J \cdot s$), we find the de Broglie wavelength: $\lambda = \frac{h}{p} = \frac{h}{mv}$.
    
    
    

---

## 3. Quantum States and Confinement

When a quantum wave is "confined" (trapped in a small space), it can only exist at specific frequencies where it constructively interferes with itself9. This leads to discrete **energy levels**.

- The Cubic Box: In a box of side length $a$, the wave must have "nodes" (zero amplitude) at the walls. This requires the number of waves along each axis to be a half-integer:
    

    
    $$\frac{a}{\lambda} = \frac{n}{2}$$
    
    This results in quantized energy levels: $E = \frac{h^{2}}{8ma^{2}}(n_{x}^{2}+n_{y}^{2}+n_{z}^{2})$.
    The momentum must be quantized then: $[p_{x},p_{y},p_{z}]=\frac{h}{2}[n_{x},n_{y},n_{z}]$
    
    
- **Harmonic Oscillator**: For a system with a quadratic potential (like a mass on a spring), energy levels are evenly spaced: $E_{n} = \hbar\omega(n + \frac{1}{2})$. $V(x)=\frac{1}{2}kx^{2}=\frac{1}{2}m \omega^{2}x^2$
    
- **Hydrogen Atom**: In an atomic potential, the energy levels are **anharmonic** (not evenly spaced), proportional to $\frac{1}{n^{2}}$.$ $V(x) = -\frac{q^2}{4\pi\epsilon_0 x}$

### Transitions and Frequency

When an electron jumps between these levels (from $E_{n}$ to $E_{m}$), it must absorb or emit a photon with a very specific frequency, calculated by7:

$$v = \frac{E_{n} - E_{m}}{h}$$

This is why different atoms emit specific colors of light; those colors correspond to the unique "fingerprint" of their energy level transitions8.

---

## 4. Superposition and Interference

**Superposition** is the ability of a system to exist in multiple states at once.

- **Double-Slit Interferometer**: If you fire single particles at two slits, they form an interference pattern over time. The particle acts as a wave going through _both_ slits simultaneously.

**The "Wave" Part of the Duality**: Superposition is essentially the "wave-like" behavior of matter. Just as two water waves can overlap to create a new pattern, the quantum particle's "path states" overlap.
    
- Mathematical Expression: We represent the particle not as being at a single point, but as a sum of states:
    
    $$\psi = \frac{1}{\sqrt{2}}|Slit\ A\rangle + \frac{1}{\sqrt{2}}|Slit\ B\rangle$$
    
    This is the mathematical way of saying it is in both places at once.
    

- **Destroying Interference**: If a detector measures which slit the particle went through, the superposition collapses. In qubit language, the detector performs a **CNOT gate** on the particle, creating entanglement and preventing the waves from being added together to show interference.
    

---

## 5. Quantum Entanglement and Bell States

**Entanglement** occurs when two or more particles are linked such that they cannot be described individually, only as a single system

- **Bell Basis**: A standard set of four highly entangled two-qubit states. For example: $\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$.
- **Manipulation**: Single-qubit gates (like $Z$ or $H$) applied to one half of an entangled pair can move the system between different Bell states
    
All entangled states involve superposition, but not all superpositions are entangled.
![](</images/Screenshot 2026-01-18 at 1.55.26 AM.png>)
- **Z-Gate on a Bell State:** Applying a $Z$ gate (a phase flip) to one qubit of the $\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$ state changes it to $\frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)$.
- **Hadamard on both:** Following that with Hadamard gates on both qubits moves the system into the $\frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$ state.

---

## 6. The Bloch Sphere and Bell’s Inequality

The **Bloch Sphere** is a geometric representation of a single qubit.

- **Rotations**: Single-qubit operations are visualized as rotations:
    - **Pauli X, Y, Z**: 180° rotations around the respective axes.
    - **Hadamard**: 180° rotation halfway between X and Z.
        
### Bell Inequalities
imagine starting with a **Bell State**—specifically the entangled pair $\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$.

**Separation**: The two qubits are sent to two different laboratories located far apart from each othe
**The Switch**: Each researcher has a switch with three settings: Left ($-\pi/4$), Middle ($0$), and Right ($\pi/4$).
    
**The Action**: Depending on the switch setting, the laboratory performs a rotation on the Y-axis of the Bloch sphere before measuring the qubit.
This experiment proves quantum mechanics is not a "hidden variables" theory.
    
Classical logic suggests that if two distant parties measure entangled qubits, their maximum agreement in certain settings should be **75%**.

Quantum mechanics predicts—and experiments confirm—an agreement of **85%**, proving that the information is not stored locally in the particles but exists in the shared quantum state.