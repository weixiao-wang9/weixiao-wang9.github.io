---
title: "Morse Theory — An Introduction"
description: "Critical points, the Morse Lemma, and handle decompositions of surfaces."
date: 2026-04-01
---

## § 1.1 — Critical Points of Functions

> [!note] Definition
> A point $x_0$ satisfying $f'(x_0) = 0$ is called a **critical point** of the function $f$. Critical points fall into two categories according to the value of $f''(x_0)$:
>
> - $x_0$ is a **non-degenerate** critical point if $f''(x_0) \neq 0$.
> - $x_0$ is a **degenerate** critical point if $f''(x_0) = 0$.

> **Observation:** Non-degenerate critical points are "stable," while degenerate critical points are "unstable."

---

## § 1.2 — The Hessian Matrix

We say that a point $p_0 = (x_0, y_0)$ in the $xy$-plane is a **critical point** of a function $z = f(x,y)$ if:

$$\frac{\partial f}{\partial x}(p_0) = 0, \qquad \frac{\partial f}{\partial y}(p_0) = 0$$

We assume $f(x,y)$ is of class $C^\infty$. We further assume $\frac{\partial^2 f}{\partial x^2}(p_0) \neq 0$ and $\frac{\partial^2 f}{\partial y^2}(p_0) \neq 0$, though after some coordinate changes this won't necessarily hold.

> [!note] Definition — Hessian
> Suppose $p_0 = (x_0, y_0)$ is a critical point of $z = f(x,y)$. The **Hessian of $f$** at $p_0$, denoted $H_f(p_0)$, is the matrix of second derivatives evaluated at $p_0$:
>
> $$H_f(p_0) = \begin{pmatrix} \dfrac{\partial^2 f}{\partial x^2}(p_0) & \dfrac{\partial^2 f}{\partial x \, \partial y}(p_0) \\[10pt] \dfrac{\partial^2 f}{\partial y \, \partial x}(p_0) & \dfrac{\partial^2 f}{\partial y^2}(p_0) \end{pmatrix}$$

A critical point $p_0$ is **non-degenerate** if the determinant of the Hessian is nonzero:

$$\det H_f(p_0) = \frac{\partial^2 f}{\partial x^2}(p_0) \cdot \frac{\partial^2 f}{\partial y^2}(p_0) - \left(\frac{\partial^2 f}{\partial x \, \partial y}(p_0)\right)^2 \neq 0$$

Note that $H_f(p_0)$ is symmetric, since $\dfrac{\partial^2 f}{\partial x \, \partial y} = \dfrac{\partial^2 f}{\partial y \, \partial x}$.

### Hessian Under Coordinate Change

> [!tip] Lemma
> Let $p_0$ be a critical point of $z = f(x,y)$. Denote by $H_f(p_0)$ the Hessian in coordinates $(x,y)$ and by $\overline{H}_f(p_0)$ the Hessian in coordinates $(X,Y)$. Then:
>
> $$\overline{H}_f(p_0) = {}^t\!J(p_0) \cdot H_f(p_0) \cdot J(p_0)$$
>
> where $J(p_0)$ is the Jacobian matrix:
>
> $$J(p_0) = \begin{pmatrix} \dfrac{\partial x}{\partial X}(p_0) & \dfrac{\partial x}{\partial Y}(p_0) \\[8pt] \dfrac{\partial y}{\partial X}(p_0) & \dfrac{\partial y}{\partial Y}(p_0) \end{pmatrix}$$

> [!info] Corollary
> The property that $p_0$ is a non-degenerate critical point **does not depend on the choice of coordinates**. The same is true for degenerate critical points.

---

## § 1.3 — The Morse Lemma

> [!abstract] Morse Lemma
> Let $p_0$ be a non-degenerate critical point of a function $f$ of two variables. Then we can choose appropriate local coordinates $(X, Y)$ such that $f$ takes one of the following three standard forms:
>
> $$\begin{aligned}
> \text{(i)} \quad & f = X^2 + Y^2 + C \\[4pt]
> \text{(ii)} \quad & f = X^2 - Y^2 + C \\[4pt]
> \text{(iii)} \quad & f = -X^2 - Y^2 + C
> \end{aligned}$$
>
> where $C$ is a constant and $p_0$ is the origin.

The theorem states that a function looks extremely simple near a non-degenerate critical point — up to a coordinate change, it is purely quadratic.

> [!info] Corollary
> A non-degenerate critical point of a function of two variables is **isolated**.

<div style="text-align:center;margin:2rem 0;">
<svg viewBox="0 0 600 210" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;">
  <g transform="translate(150,20)">
    <text x="0" y="0" font-family="Georgia,serif" font-size="14" fill="#8b4513" text-anchor="middle" font-style="italic">Non-degenerate</text>
    <path d="M-60,150 Q-50,60 0,50 Q50,60 60,150" fill="none" stroke="#3d2e1a" stroke-width="2"/>
    <circle cx="0" cy="50" r="4" fill="#8b4513"/>
    <text x="14" y="46" font-family="Georgia,serif" font-size="13" fill="#6b5d4f" font-style="italic">p&#x2080;</text>
    <text x="0" y="185" font-family="monospace" font-size="10" fill="#6b5d4f" text-anchor="middle" letter-spacing="2">ISOLATED</text>
  </g>
  <g transform="translate(440,20)">
    <text x="0" y="0" font-family="Georgia,serif" font-size="14" fill="#8b4513" text-anchor="middle" font-style="italic">Degenerate</text>
    <path d="M-80,150 Q-60,70 -20,55 Q-5,50 0,55 Q5,50 20,55 Q60,70 80,150" fill="none" stroke="#3d2e1a" stroke-width="2"/>
    <circle cx="-20" cy="55" r="3.5" fill="#8b4513"/>
    <circle cx="20" cy="55" r="3.5" fill="#8b4513"/>
    <text x="0" y="185" font-family="monospace" font-size="10" fill="#6b5d4f" text-anchor="middle" letter-spacing="2">NOT ISOLATED</text>
  </g>
</svg>
<p style="font-size:0.85rem;color:#6b5d4f;font-style:italic;margin-top:0.5rem;">Figure 1 — Non-degenerate critical points are isolated; degenerate ones need not be.</p>
</div>

---

## § 1.4 — Index of a Non-Degenerate Critical Point

> [!note] Definition
> Let $p_0$ be a non-degenerate critical point of $f$. Using the coordinate system given by the Morse Lemma, we define the **index** of $p_0$ as:
>
> $$\operatorname{index}(p_0) = \begin{cases} 0 & \text{if } f = x^2 + y^2 + C \quad \text{(local minimum)} \\ 1 & \text{if } f = x^2 - y^2 + C \quad \text{(saddle point)} \\ 2 & \text{if } f = -x^2 - y^2 + C \quad \text{(local maximum)} \end{cases}$$
>
> In other words, the index is the number of minus signs in the standard form.

The index of a non-degenerate critical point $p_0$ is determined by the *behaviour of $f$ near $p_0$*.

<div style="text-align:center;margin:2rem 0;">
<svg viewBox="0 0 660 190" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;">
  <g transform="translate(110,15)">
    <text x="0" y="0" font-family="Georgia,serif" font-size="13" fill="#8b4513" text-anchor="middle" font-style="italic">Index 0 — Minimum</text>
    <ellipse cx="0" cy="130" rx="65" ry="20" fill="none" stroke="#3d2e1a" stroke-width="1.5" stroke-dasharray="4,3"/>
    <path d="M-65,130 Q-55,55 0,40 Q55,55 65,130" fill="none" stroke="#3d2e1a" stroke-width="2"/>
    <circle cx="0" cy="40" r="4" fill="#8b4513"/>
    <text x="0" y="170" font-family="monospace" font-size="11" fill="#3d2e1a" text-anchor="middle">x&#xB2;+y&#xB2;+C</text>
  </g>
  <g transform="translate(330,15)">
    <text x="0" y="0" font-family="Georgia,serif" font-size="13" fill="#8b4513" text-anchor="middle" font-style="italic">Index 1 — Saddle</text>
    <path d="M-60,50 Q-30,90 0,85 Q30,80 60,120" fill="none" stroke="#3d2e1a" stroke-width="2"/>
    <path d="M-60,120 Q-30,80 0,85 Q30,90 60,50" fill="none" stroke="#3d2e1a" stroke-width="2"/>
    <circle cx="0" cy="85" r="4" fill="#8b4513"/>
    <text x="0" y="170" font-family="monospace" font-size="11" fill="#3d2e1a" text-anchor="middle">x&#xB2;&#x2212;y&#xB2;+C</text>
  </g>
  <g transform="translate(550,15)">
    <text x="0" y="0" font-family="Georgia,serif" font-size="13" fill="#8b4513" text-anchor="middle" font-style="italic">Index 2 — Maximum</text>
    <ellipse cx="0" cy="50" rx="65" ry="20" fill="none" stroke="#3d2e1a" stroke-width="1.5" stroke-dasharray="4,3"/>
    <path d="M-65,50 Q-55,125 0,140 Q55,125 65,50" fill="none" stroke="#3d2e1a" stroke-width="2"/>
    <circle cx="0" cy="140" r="4" fill="#8b4513"/>
    <text x="0" y="170" font-family="monospace" font-size="11" fill="#3d2e1a" text-anchor="middle">&#x2212;x&#xB2;&#x2212;y&#xB2;+C</text>
  </g>
</svg>
<p style="font-size:0.85rem;color:#6b5d4f;font-style:italic;margin-top:0.5rem;">Figure 2 — The three types of non-degenerate critical points in two variables, classified by index.</p>
</div>

---

## § 2.1 — Morse Functions on Surfaces

We now move from functions on $\mathbb{R}^2$ to functions on surfaces. Recall that a **closed surface** is a compact surface without boundary. The **genus** of a closed surface is the number of "holes" in it.

<div style="text-align:center;margin:2rem 0;">
<svg viewBox="0 0 520 150" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;">
  <g transform="translate(90,70)">
    <ellipse cx="0" cy="0" rx="45" ry="42" fill="none" stroke="#3d2e1a" stroke-width="2"/>
    <path d="M-15,-30 Q-5,-38 8,-32" fill="none" stroke="#3d2e1a" stroke-width="1" opacity="0.4"/>
    <text x="0" y="62" font-family="Georgia,serif" font-size="13" fill="#6b5d4f" text-anchor="middle" font-style="italic">genus 0</text>
  </g>
  <g transform="translate(260,70)">
    <ellipse cx="0" cy="0" rx="55" ry="38" fill="none" stroke="#3d2e1a" stroke-width="2"/>
    <ellipse cx="0" cy="2" rx="20" ry="11" fill="none" stroke="#3d2e1a" stroke-width="1.5"/>
    <text x="0" y="62" font-family="Georgia,serif" font-size="13" fill="#6b5d4f" text-anchor="middle" font-style="italic">genus 1</text>
  </g>
  <g transform="translate(430,70)">
    <path d="M-55,20 Q-55,-30 -25,-32 Q-5,-33 0,-15 Q5,-33 25,-32 Q55,-30 55,20 Q55,40 25,38 Q5,37 0,25 Q-5,37 -25,38 Q-55,40 -55,20Z" fill="none" stroke="#3d2e1a" stroke-width="2"/>
    <ellipse cx="-25" cy="5" rx="13" ry="9" fill="none" stroke="#3d2e1a" stroke-width="1.5"/>
    <ellipse cx="25" cy="5" rx="13" ry="9" fill="none" stroke="#3d2e1a" stroke-width="1.5"/>
    <text x="0" y="62" font-family="Georgia,serif" font-size="13" fill="#6b5d4f" text-anchor="middle" font-style="italic">genus 2</text>
  </g>
</svg>
<p style="font-size:0.85rem;color:#6b5d4f;font-style:italic;margin-top:0.5rem;">Figure 3 — Closed surfaces of genus 0, 1, and 2.</p>
</div>

Let $M$ be a surface. A function $f\colon M \to \mathbb{R}$ assigns a real number to each point $p \in M$. We say $f$ is of class $C^\infty$ if it is smooth with respect to any smooth local coordinates at each point of $M$.

> [!note] Definition — Morse Function
> Suppose that every critical point of $f\colon M \to \mathbb{R}$ is non-degenerate. Then we say that $f$ is a **Morse function**.

> [!example] Example
> Consider the sphere $S^2 \subset \mathbb{R}^3$ defined by $x^2 + y^2 + z^2 = 1$. Let $f\colon S^2 \to \mathbb{R}$ be the height function $f(x,y,z) = z$. Then $f$ has exactly two critical points — the north pole $p_0$ (a maximum) and the south pole $q_0$ (a minimum) — and both are non-degenerate. Hence $f$ is a Morse function.

<div style="text-align:center;margin:2rem 0;">
<svg viewBox="0 0 400 240" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;">
  <defs>
    <marker id="arr1" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0,8 3,0 6" fill="#3d2e1a"/>
    </marker>
  </defs>
  <g transform="translate(110,120)">
    <ellipse cx="0" cy="0" rx="55" ry="55" fill="none" stroke="#3d2e1a" stroke-width="2"/>
    <ellipse cx="0" cy="0" rx="55" ry="16" fill="none" stroke="#3d2e1a" stroke-width="1" stroke-dasharray="4,3"/>
    <line x1="0" y1="-75" x2="0" y2="75" stroke="#3d2e1a" stroke-width="0.8" stroke-dasharray="3,3" opacity="0.4"/>
    <text x="8" y="-62" font-family="Georgia,serif" font-size="13" fill="#6b5d4f" font-style="italic">z</text>
    <circle cx="0" cy="-55" r="4" fill="#8b4513"/>
    <text x="12" y="-51" font-family="Georgia,serif" font-size="13" fill="#8b4513" font-style="italic">p&#x2080;</text>
    <circle cx="0" cy="55" r="4" fill="#8b4513"/>
    <text x="12" y="59" font-family="Georgia,serif" font-size="13" fill="#8b4513" font-style="italic">q&#x2080;</text>
    <text x="0" y="95" font-family="Georgia,serif" font-size="13" fill="#6b5d4f" text-anchor="middle">S&#xB2;</text>
  </g>
  <g transform="translate(195,120)">
    <line x1="0" y1="0" x2="55" y2="0" stroke="#3d2e1a" stroke-width="1.5" marker-end="url(#arr1)"/>
    <text x="27" y="-10" font-family="Georgia,serif" font-size="13" fill="#6b5d4f" text-anchor="middle" font-style="italic">f</text>
  </g>
  <g transform="translate(300,120)">
    <line x1="0" y1="-70" x2="0" y2="70" stroke="#3d2e1a" stroke-width="1.5"/>
    <text x="14" y="-60" font-family="Georgia,serif" font-size="13" fill="#6b5d4f" font-style="italic">&#x211D;</text>
    <circle cx="0" cy="-50" r="3" fill="#8b4513"/>
    <text x="12" y="-46" font-family="Georgia,serif" font-size="11" fill="#6b5d4f" font-style="italic">max</text>
    <circle cx="0" cy="50" r="3" fill="#8b4513"/>
    <text x="12" y="54" font-family="Georgia,serif" font-size="11" fill="#6b5d4f" font-style="italic">min</text>
  </g>
</svg>
<p style="font-size:0.85rem;color:#6b5d4f;font-style:italic;margin-top:0.5rem;">Figure 4 — The height function on $S^2$ as a Morse function with two critical points.</p>
</div>

> [!tip] Lemma
> A Morse function $f\colon M \to \mathbb{R}$ defined on a closed surface $M$ has only a **finite number** of critical points.

---

## § 2.2 — Diffeomorphism and the Reeb Theorem

> [!note] Definition — Homeomorphism
> Suppose there is a one-to-one and onto map $h\colon X \to Y$ between two topological spaces $X$ and $Y$ with inverse $h^{-1}\colon Y \to X$. If both maps are continuous, we say $X$ and $Y$ are **homeomorphic** and $h$ is a homeomorphism — intuitively, they have the *same shape*.

> [!note] Definition — Diffeomorphism
> A homeomorphism $h\colon M \to N$ between surfaces is a **diffeomorphism** if both $h$ and $h^{-1}$ are of class $C^\infty$. Two surfaces are called **diffeomorphic** if there is a diffeomorphism between them.

> [!abstract] Theorem — Maximum-Value Theorem
> Let $f\colon X \to \mathbb{R}$ be a continuous function on a compact space $X$. Then $f$ takes its maximum value at some point $p_0$ and its minimum value at some point $q_0$.

> [!abstract] Theorem (Reeb)
> Let $M$ be a closed surface. Suppose there exists a Morse function $f\colon M \to \mathbb{R}$ with exactly two non-degenerate critical points. Then $M$ is **diffeomorphic to the sphere $S^2$**.

<div style="text-align:center;margin:2rem 0;">
<svg viewBox="0 0 480 180" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;">
  <g transform="translate(100,90)">
    <path d="M-40,-60 Q-55,-40 -50,-10 Q-55,20 -40,50 Q-20,70 10,65 Q40,60 45,30 Q50,0 45,-20 Q40,-50 20,-62 Q0,-68 -40,-60Z" fill="none" stroke="#3d2e1a" stroke-width="2"/>
    <circle cx="-10" cy="-60" r="3.5" fill="#8b4513"/>
    <text x="-2" y="-65" font-family="Georgia,serif" font-size="12" fill="#8b4513" font-style="italic">p&#x2080;</text>
    <circle cx="5" cy="60" r="3.5" fill="#8b4513"/>
    <text x="14" y="63" font-family="Georgia,serif" font-size="12" fill="#8b4513" font-style="italic">q&#x2080;</text>
    <text x="0" y="90" font-family="Georgia,serif" font-size="14" fill="#6b5d4f" text-anchor="middle" font-style="italic">M</text>
  </g>
  <text x="215" y="95" font-family="Georgia,serif" font-size="28" fill="#3d2e1a" text-anchor="middle">&#x2245;</text>
  <g transform="translate(340,90)">
    <ellipse cx="0" cy="0" rx="50" ry="50" fill="none" stroke="#3d2e1a" stroke-width="2"/>
    <ellipse cx="0" cy="0" rx="50" ry="14" fill="none" stroke="#3d2e1a" stroke-width="1" stroke-dasharray="4,3"/>
    <circle cx="0" cy="-50" r="3.5" fill="#8b4513"/>
    <circle cx="0" cy="50" r="3.5" fill="#8b4513"/>
    <text x="0" y="78" font-family="Georgia,serif" font-size="14" fill="#6b5d4f" text-anchor="middle" font-style="italic">S&#xB2;</text>
  </g>
</svg>
<p style="font-size:0.85rem;color:#6b5d4f;font-style:italic;margin-top:0.5rem;">Figure 5 — A closed surface admitting a Morse function with exactly two critical points is diffeomorphic to $S^2$.</p>
</div>

### Proof Sketch

By the maximum-value theorem, $f\colon M \to \mathbb{R}$ takes its maximum at $p_0$ and its minimum at $q_0$. Since $f$ is a Morse function, both are non-degenerate critical points. By the Morse Lemma, $f$ takes standard form near each:

$$f = \begin{cases} -x^2 - y^2 + A & \text{near } p_0 \\ X^2 + Y^2 + a & \text{near } q_0 \end{cases}$$

Here $A$ and $a$ are the maximum and minimum values. For small $\varepsilon > 0$, the set $D(p_0) = \{A - \varepsilon \leq f(p) \leq A\}$ satisfies $x^2 + y^2 \leq \varepsilon$, making it diffeomorphic to a 2-disk. Similarly for $D(q_0)$.

<div style="text-align:center;margin:2rem 0;">
<svg viewBox="0 0 460 160" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;">
  <g transform="translate(120,80)">
    <path d="M-50,30 Q-50,-20 0,-35 Q50,-20 50,30" fill="none" stroke="#3d2e1a" stroke-width="2"/>
    <ellipse cx="0" cy="30" rx="50" ry="12" fill="none" stroke="#3d2e1a" stroke-width="1.5" stroke-dasharray="4,3"/>
    <circle cx="0" cy="-35" r="3.5" fill="#8b4513"/>
    <text x="10" y="-32" font-family="Georgia,serif" font-size="12" fill="#8b4513" font-style="italic">p&#x2080;</text>
    <text x="0" y="68" font-family="Georgia,serif" font-size="12" fill="#6b5d4f" text-anchor="middle" font-style="italic">f near p&#x2080;</text>
  </g>
  <g transform="translate(340,80)">
    <path d="M-50,-30 Q-50,20 0,35 Q50,20 50,-30" fill="none" stroke="#3d2e1a" stroke-width="2"/>
    <ellipse cx="0" cy="-30" rx="50" ry="12" fill="none" stroke="#3d2e1a" stroke-width="1.5" stroke-dasharray="4,3"/>
    <circle cx="0" cy="35" r="3.5" fill="#8b4513"/>
    <text x="10" y="38" font-family="Georgia,serif" font-size="12" fill="#8b4513" font-style="italic">q&#x2080;</text>
    <text x="0" y="68" font-family="Georgia,serif" font-size="12" fill="#6b5d4f" text-anchor="middle" font-style="italic">f near q&#x2080;</text>
  </g>
</svg>
<p style="font-size:0.85rem;color:#6b5d4f;font-style:italic;margin-top:0.5rem;">Figure 6 — Disk neighbourhoods around the maximum and minimum.</p>
</div>

Removing the interiors of $D(p_0)$ and $D(q_0)$ from $M$ yields a surface $M_0$ with boundary $\partial M_0 = C(p_0) \cup C(q_0)$.

> [!tip] Lemma
> Let $f\colon M_0 \to \mathbb{R}$ be a $C^\infty$ function taking constant values on the boundary circles $C(p_0)$ and $C(q_0)$. If $f$ has no critical points on $M_0$, then $M_0$ is diffeomorphic to $C(q_0) \times [0,1]$.

Since $C(q_0) \cong S^1$, we conclude $M_0 \cong S^1 \times [0,1]$ — an **annulus**. Gluing $D(p_0)$ and $D(q_0)$ back along the boundary circles reconstructs $M$ and shows it is diffeomorphic to $S^2$.

<div style="text-align:center;margin:2rem 0;">
<svg viewBox="0 0 360 200" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;">
  <g transform="translate(180,100)">
    <ellipse cx="0" cy="-60" rx="55" ry="14" fill="none" stroke="#3d2e1a" stroke-width="2"/>
    <line x1="-55" y1="-60" x2="-55" y2="60" stroke="#3d2e1a" stroke-width="2"/>
    <line x1="55" y1="-60" x2="55" y2="60" stroke="#3d2e1a" stroke-width="2"/>
    <ellipse cx="0" cy="60" rx="55" ry="14" fill="none" stroke="#3d2e1a" stroke-width="2"/>
    <text x="72" y="-57" font-family="Georgia,serif" font-size="12" fill="#8b4513" font-style="italic">C(p&#x2080;)</text>
    <text x="72" y="63" font-family="Georgia,serif" font-size="12" fill="#8b4513" font-style="italic">C(q&#x2080;)</text>
    <text x="-80" y="5" font-family="Georgia,serif" font-size="14" fill="#6b5d4f" font-style="italic">M&#x2080;</text>
  </g>
</svg>
<p style="font-size:0.85rem;color:#6b5d4f;font-style:italic;margin-top:0.5rem;">Figure 7 — $M_0 \cong S^1 \times [0,1]$, the annulus between the two boundary circles.</p>
</div>

---

## § 1.5 — Handle Decomposition

Start with a Morse function $f\colon M \to \mathbb{R}$ on a closed, connected surface. Define the **sublevel set**:

$$M_t = \{p \in M \mid f(p) \leq t\}$$

Let $L_t$ be the level curve $\{f = t\}$. Then $M_t$ is everything below $L_t$, and $L_t = \partial M_t$.

<div style="text-align:center;margin:2rem 0;">
<svg viewBox="0 0 460 240" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;">
  <defs>
    <marker id="arr2" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0,8 3,0 6" fill="#3d2e1a"/>
    </marker>
  </defs>
  <g transform="translate(175,120)">
    <ellipse cx="0" cy="0" rx="80" ry="70" fill="none" stroke="#3d2e1a" stroke-width="2"/>
    <ellipse cx="0" cy="-5" rx="28" ry="16" fill="none" stroke="#3d2e1a" stroke-width="1.5"/>
    <line x1="-90" y1="15" x2="90" y2="15" stroke="#8b4513" stroke-width="1.5" stroke-dasharray="5,3"/>
    <text x="96" y="19" font-family="Georgia,serif" font-size="12" fill="#8b4513" font-style="italic">L_t</text>
    <path d="M-80,15 Q-80,70 0,70 Q80,70 80,15" fill="rgba(139,69,19,0.08)" stroke="none"/>
    <text x="0" y="52" font-family="Georgia,serif" font-size="13" fill="#8b4513" text-anchor="middle" font-style="italic">M_t</text>
  </g>
  <g transform="translate(360,120)">
    <line x1="0" y1="-90" x2="0" y2="90" stroke="#3d2e1a" stroke-width="1.5"/>
    <text x="14" y="-78" font-family="Georgia,serif" font-size="12" fill="#6b5d4f" font-style="italic">A</text>
    <text x="14" y="85" font-family="Georgia,serif" font-size="12" fill="#6b5d4f" font-style="italic">a</text>
    <line x1="-5" y1="15" x2="5" y2="15" stroke="#8b4513" stroke-width="2"/>
    <text x="14" y="19" font-family="Georgia,serif" font-size="12" fill="#8b4513" font-style="italic">t</text>
    <line x1="-55" y1="0" x2="-15" y2="0" stroke="#3d2e1a" stroke-width="1" marker-end="url(#arr2)"/>
    <text x="-35" y="-10" font-family="Georgia,serif" font-size="12" fill="#6b5d4f" font-style="italic">f</text>
  </g>
</svg>
<p style="font-size:0.85rem;color:#6b5d4f;font-style:italic;margin-top:0.5rem;">Figure 8 — The sublevel set $M_t$ and level curve $L_t$ of a Morse function on a surface.</p>
</div>

Since $f$ has maximum $A$ and minimum $a$: $M_t = \varnothing$ for $t < a$, and $M_t = M$ for $t \geq A$.

**The fundamental idea of Morse Theory is to trace the change of shape of $M_t$ as $t$ increases from below $a$ to above $A$.**

> [!note] Definition — Critical Values
> A real number $c_0$ is a **critical value** if $f(p_0) = c_0$ for some critical point $p_0$.

> [!tip] Lemma
> Let $b < c$ be real numbers such that $f$ has no critical values in $[b,c]$. Then $M_b$ and $M_c$ are **diffeomorphic**.

The topology of $M_t$ only changes when $t$ passes through a critical value. The change depends on the **index** of the critical point.

### Index 0 — Attaching a 0-handle

When the index of $p_0$ is zero, $f = x^2 + y^2 + c_0$ near $p_0$. The sublevel set gains a new connected component — a disk $D^2$:

$$M_{c_0 + \varepsilon} \cong M_{c_0 - \varepsilon} \;\sqcup\; D^2$$

<div style="text-align:center;margin:2rem 0;">
<svg viewBox="0 0 420 140" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;">
  <g transform="translate(95,70)">
    <rect x="-65" y="-45" width="130" height="90" rx="6" fill="none" stroke="#c8bfb0" stroke-width="1.5" stroke-dasharray="6,4"/>
    <text x="0" y="-4" font-family="Georgia,serif" font-size="18" fill="#c8bfb0" text-anchor="middle">&#x2205;</text>
    <text x="0" y="55" font-family="Georgia,serif" font-size="11" fill="#9a8a7a" text-anchor="middle" font-style="italic">M_{c&#x2080;&#x2212;&#x03B5;} = &#x2205;</text>
  </g>
  <text x="208" y="74" font-family="Georgia,serif" font-size="22" fill="#3d2e1a" text-anchor="middle">&#x27F6;</text>
  <g transform="translate(322,70)">
    <ellipse cx="0" cy="0" rx="48" ry="48" fill="rgba(139,69,19,0.10)" stroke="#8b4513" stroke-width="2"/>
    <circle cx="0" cy="0" r="4" fill="#8b4513"/>
    <text x="14" y="-8" font-family="Georgia,serif" font-size="12" fill="#8b4513" font-style="italic">p&#x2080;</text>
    <text x="0" y="18" font-family="Georgia,serif" font-size="13" fill="#8b4513" text-anchor="middle" font-style="italic">D&#xB2;</text>
    <text x="0" y="66" font-family="Georgia,serif" font-size="11" fill="#9a8a7a" text-anchor="middle" font-style="italic">M_{c&#x2080;+&#x03B5;} &#x2245; D&#xB2;</text>
  </g>
  <text x="210" y="128" font-family="monospace" font-size="10" fill="#6b5d4f" text-anchor="middle" letter-spacing="2">0-HANDLE</text>
</svg>
<p style="font-size:0.85rem;color:#6b5d4f;font-style:italic;margin-top:0.5rem;">Figure 9 — Passing the minimum (index 0): the sublevel set jumps from empty to a single disk.</p>
</div>

### Index 1 — Attaching a 1-handle

When the index of $p_0$ is one, $f = x^2 - y^2 + c_0$ near $p_0$. The sublevel set gains a **bridge** $D^1 \times D^1$:

$$M_{c_0 + \varepsilon} \cong M_{c_0 - \varepsilon} \;\cup\; (D^1 \times D^1)$$

<div style="text-align:center;margin:2rem 0;">
<svg viewBox="0 0 500 170" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;">
  <g transform="translate(115,85)">
    <text x="0" y="-62" font-family="Georgia,serif" font-size="12" fill="#6b5d4f" text-anchor="middle" font-style="italic">before</text>
    <ellipse cx="-38" cy="0" rx="28" ry="45" fill="rgba(139,69,19,0.06)" stroke="#3d2e1a" stroke-width="2"/>
    <ellipse cx="38" cy="0" rx="28" ry="45" fill="rgba(139,69,19,0.06)" stroke="#3d2e1a" stroke-width="2"/>
    <text x="0" y="5" font-family="Georgia,serif" font-size="11" fill="#9a8a7a" text-anchor="middle" font-style="italic">gap</text>
  </g>
  <text x="247" y="89" font-family="Georgia,serif" font-size="22" fill="#3d2e1a" text-anchor="middle">&#x27F6;</text>
  <g transform="translate(383,85)">
    <text x="0" y="-62" font-family="Georgia,serif" font-size="12" fill="#6b5d4f" text-anchor="middle" font-style="italic">after</text>
    <ellipse cx="-38" cy="0" rx="28" ry="45" fill="rgba(139,69,19,0.06)" stroke="#3d2e1a" stroke-width="2"/>
    <ellipse cx="38" cy="0" rx="28" ry="45" fill="rgba(139,69,19,0.06)" stroke="#3d2e1a" stroke-width="2"/>
    <rect x="-10" y="-14" width="20" height="28" rx="2" fill="rgba(139,69,19,0.18)" stroke="#8b4513" stroke-width="2"/>
    <circle cx="0" cy="0" r="3.5" fill="#8b4513"/>
    <text x="14" y="-5" font-family="Georgia,serif" font-size="11" fill="#8b4513" font-style="italic">p&#x2080;</text>
  </g>
  <text x="250" y="157" font-family="monospace" font-size="10" fill="#6b5d4f" text-anchor="middle" letter-spacing="2">1-HANDLE (BRIDGE)</text>
</svg>
<p style="font-size:0.85rem;color:#6b5d4f;font-style:italic;margin-top:0.5rem;">Figure 10 — Passing a critical point of index 1 connects two components via a bridge (1-handle = $D^1 \times D^1$).</p>
</div>

### Index 2 — Attaching a 2-handle

When the index of $p_0$ is two, $f = -x^2 - y^2 + c_0$ near $p_0$. The sublevel set is capped off by a disk $D^2$:

$$M_{c_0 + \varepsilon} \cong M_{c_0 - \varepsilon} \;\cup\; D^2$$

> [!abstract] Theorem — Handle Decomposition
> If a closed surface $M$ admits a Morse function $f\colon M \to \mathbb{R}$, then $M$ can be described as a union of finitely many **0-handles, 1-handles, and 2-handles**.

This is the fundamental result of Morse theory on surfaces: any closed surface equipped with a Morse function can be built up, piece by piece, by attaching handles of index 0, 1, and 2 as we sweep through the critical values from the minimum to the maximum.

---

*Notes on Morse Theory — transcribed from handwritten lecture notes.*
