---
id: 202601181358
type: source
subtype: lecture_note
course: Quantum Hardware
module:
title: Superconducting Qubit Technology
status: Finished
tags: []
created: 2026-01-18
---
## 1. Quantized Energy Levels and Systems

Small physical systems (atomic, electronic, or optical) exist in discrete energy levels known as a **spectrum**. The arrangement and spacing of these levels reveal the fundamental nature of the system.

In qubits, which only use states $|0\rangle$ and $|1\rangle$, many physical systems have a much larger range of possible states
### Types of Energy Spacings

- **Harmonic (Evenly Spaced):** The energy difference between any two adjacent levels is identical.
    
- **Anharmonic (Unevenly Spaced):** The energy gaps vary between levels. This is critical for isolating specific states.
    
- **Degenerate:** Multiple distinct quantum states exist at the exact same energy level.
    
- **Chaotic:** Complex systems where levels follow specific mathematical relationships.
![[Screenshot 2026-01-18 at 2.57.00 PM.png]]

---

## 2. Harmonic Systems

Harmonic systems are common in optical and electronic domains (e.g., photons).

- **Data Analogy:** They are the quantum version of an "unsigned int" because they can occupy infinitely many possible states ($|0\rangle, |1\rangle, |2\rangle, \dots$).

    
- **Operators:** These systems are manipulated using **Raising ($\hat{a}^*$)** and **Lowering ($\hat{a}$)** operators.
    
    - The lowering operator reduces the state: $\hat{a}|n\rangle = \sqrt{n}|n-1\rangle$.
    - The raising operator increases the state: $\hat{a}^*|n\rangle = \sqrt{n+1}|n+1\rangle$.
>**The Operator $\hat{a}^*\hat{a}$**: This is defined as the **number operator** ($\hat{n}$). When it acts on a state $|n\rangle$, it "measures" the energy level and returns the number $n$.
$$\hat{a}\hat{a}^*|n\rangle = (n+1)|n\rangle = |n\rangle + n|n\rangle = (I + \hat{n})|n\rangle$$
if you raise a state and then immediately lower it, you get a term proportional to the energy level plus one

- Position ($\hat{x}$): Represented as the sum of the raising and lowering operators multiplied by a scaling factor involving Planck's constant ($\hbar$), mass ($m$), and frequency ($\omega$).
    
    $$\hat{x}=\sqrt{\frac{\hbar}{2m\omega}}(\hat{a}+\hat{a}^*)$$
    
- Momentum ($\hat{p}$): Represented as the difference between the operators10. Notice the $i$ (imaginary unit), which is a hallmark of quantum momentum.
    
    $$\hat{p}=-i\sqrt{\frac{\hbar m\omega}{2}}(\hat{a}-\hat{a}^*)$$
- **Linearity:** Because the levels are evenly spaced ($E_n = n\hbar\omega$), Energy of each state is:
$$\hat{H}|n\rangle = \hbar\omega\hat{a}^*\hat{a}|n\rangle = n\hbar\omega|n\rangle$$

- a frequency that drives a system from $0 \rightarrow 1$ will also drive it from $1 \rightarrow 2$, causing it to "climb the ladder" rather than staying in two specific states.

    

---

## 3. Anharmonic Systems and Qubits

To create a **qubit** (a "bool" data type), a system must be anharmonic.

- **Isolation:** In an anharmonic system, the frequency required to move from $0 \rightarrow 1$ is different from $1 \rightarrow 2$. This allows researchers to isolate the bottom two levels to function as a qubit.
    
- Anharmonicity Formula: It is measured by the difference in energy gaps:
    
    $$\text{Anharmonicity} = (E_2 - E_1) - (E_1 - E_0)$$
    

---

## 4. The Josephson Junction

The Josephson junction is the "key element" used to create artificial atoms called **transmon qubits**. It consists of two superconducting reservoirs separated by a thin insulating barrier.

- **Superconductivity:** Electrons form **charge-pairs** (Cooper pairs) with a charge of $-2e$. Superconductors are used because they allow electrons to move across the junction without dissipating energy.
    
- **Tunneling:** Charge pairs can tunnel across the barrier, changing the distribution of charge between the two sides.
    
- **The Phase ($\phi$):** The quantum state is often expressed using a phase variable $\phi$, which is the Fourier Transform of the charge states.
    

### The Quantum Hamiltonian

The energy of the junction is described by an equation resembling the Schrödinger equation:

$$\hat{H}\psi(\phi) = -\frac{(2e)^2}{2C}\frac{\partial^2\psi(\phi)}{\partial\phi^2} - E_j \cos\phi \, \psi(\phi)$$


- **Kinetic Energy:** The junction's **capacitance** ($C$) acts like mass in this analogy.
    
- **Potential Energy:** The tunneling acts like a **cosine potential**.


---

## 5. The RCSJ Model (Classical Limit)

When a junction is large, it can be treated classically using the **Resistively and Capacitively Shunted Junction (RCSJ)** model.

### The "Tilted Washboard" Analogy

The RCSJ model is visualized as a particle in a **tilted washboard potential**:

- **The Washboard:** The cosine term creates the "bumps" (minima)
    
- **The Tilt:** An external current bias ($I_{ext}$) tilts the potential.
    
- **Damping:** A resistive shunt ($R$) acts like viscous damping (friction).

### Behavior

1. **Stable (Zero Voltage):** If the tilt is small, the "particle" (phase) sits in a minimum. There is a supercurrent but no average voltage.
    
2. **Running State (Voltage):** If the tilt is too steep (high current), the particle rolls down the hills forever. This represents a constant average voltage with small AC oscillations as it bumps over each crest.