---
title: "Neural Manifolds"
description: "On the geometry and topology of learned representations."
date: 2025-03-01
status: "In Progress"
demo: "https://manifold-visualization.onrender.com"
tags: ["manifold learning", "representation geometry", "neural networks"]
---

## Overview

This project investigates the geometric and topological properties of representations learned by deep neural networks. We study how the structure of data manifolds evolves across layers and training epochs.

## Key Questions

- How does the intrinsic dimensionality of learned representations change across network depth?
- What topological invariants (Betti numbers, persistent homology) characterize well-trained vs. poorly-trained representations?
- Can we use manifold geometry to diagnose failure modes in deep learning?

## Approach

### Representation Geometry

We analyze the geometry of intermediate representations using:
- **Intrinsic dimensionality estimation** via maximum likelihood methods
- **Curvature analysis** of the representation manifold
- **Geodesic distance** comparisons between Euclidean and manifold metrics

### Topological Analysis

Using persistent homology, we track the birth and death of topological features as data flows through the network layers, revealing how neural networks progressively simplify the topology of the input data.

## Interactive Visualization

The companion web application provides real-time 3D visualization of neural manifolds using Three.js, allowing users to explore how different architectures shape the geometry of learned representations.
