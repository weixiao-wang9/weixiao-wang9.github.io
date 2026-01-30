---
title: "Bayesian Rips Active Learning (BRAL)"
description: "Topology-aware active learning for rare lineage discovery."
date: 2025-01-15
status: "In Progress"
github: "https://github.com/weixiao-wang9/BRAL"
tags: ["topological data analysis", "active learning", "bayesian inference"]
---

## Overview

Bayesian Rips Active Learning (BRAL) is a framework that integrates topological data analysis with active learning strategies to discover rare lineages in high-dimensional biological datasets.

## Motivation

Traditional active learning methods rely on geometric or density-based heuristics that struggle with manifold-structured data. BRAL leverages persistent homology from the Rips complex to identify topologically significant regions where rare lineages are likely to reside.

## Method

### Rips Complex Construction

We construct a Vietoris-Rips complex from the current labeled set, computing persistent homology to identify topological features (connected components, loops, voids) that persist across multiple scales.

### Bayesian Acquisition Function

The acquisition function combines:
- **Topological uncertainty**: regions where the Rips complex exhibits unstable homological features
- **Posterior predictive variance**: standard Bayesian uncertainty from the classification model
- **Diversity score**: ensures spatial coverage across the manifold

### Active Learning Loop

At each iteration:
1. Compute the Rips complex on labeled data
2. Extract persistent homology features
3. Score unlabeled points using the topology-aware acquisition function
4. Query the oracle for labels on the top-k candidates
5. Update the model and repeat

## Results

BRAL demonstrates improved discovery rates for rare lineages compared to standard active learning baselines, particularly in settings where the rare class occupies a topologically distinct region of the data manifold.
