---
title: "Manifold Viz: Topological Dynamics of Active Learning"
type: "source"
course: "Research / Machine Learning"
created: "2026-01-23"
tags: ["FastAPI", "PyTorch", "TDA"]
github: "https://github.com/weixiao-wang9/manifold_viz"
---

## Executive Summary
Manifold Viz is an interactive dashboard designed to open the "black box" of **Active Learning**. Instead of simple metrics, it tracks how a model learns the **geometry (latent manifold)** and **topology (connectivity)** of 20D synthetic datasets.

## The Architecture


### The Backend (FastAPI + PyTorch)
- **TDA Engine:** Computes persistent homology ($H_0, H_1$) via **Ripser**.
- **Acquisition:** Compares Uncertainty Sampling vs. Coreset Diversity.
- **Drift Calculation:** Tracks latent movement using the vector:
  $$v_i = Z_t^{(i)} - Z_{t-1}^{(i)}$$

### The Frontend (React Three Fiber)
- **InstancedMesh:** Renders 10,000+ particles at 60fps.
- **Time Travel:** A snapshot system that allows scrubbing through the model's history to pinpoint structural failures like mode collapse.

## Stress-Testing the "Bifurcation"
The system simulates a "Y"-shaped manifold embedded in 20D space. 
- **The Challenge:** 18 dimensions of noise and a 15% rare branch.
- **The Result:** Visualizes whether the model's "uncertainty" correctly identifies the bifurcation point before the rare branch is lost to noise.
