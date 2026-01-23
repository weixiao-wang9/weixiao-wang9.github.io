---
date: 202512142254
tags:
  - atomic
  - concept
source: "[[OS CPU Scheduling]]"
---

# Multilevel Feedback Queue (MLFQ)

## Definition
### Multilevel Feedback Queue (MLFQ)
Give different tasks different timeslices and priorities automatically, based on their behavior.

Core idea: Maintain multiple queue, each with:
	A different priority
	A different timeslice length
Top queue
	Highest priority, shortest timeslice (good for I/O bound/interactive)
Bottom queue:
	Lowest priority, longest timeslice (good for CPU-bound)
Heuristic rules:
1. New tasks start in the topmost queue
2. When a task runs:
	1. If it yields before its timeslice ends -> likely I/O-bound or interactive -> keep it in a high-priority queue
	2. If it must be preempted because it used up its timeslice -> likely CPU-bound -> demote it to a lower queue with a longer quantum.
	3. If a task in a lower queue starts yielding often, you can boost it back up

	>The scheduler learns from how tasks behave and adjusts their queue accordingly. It adapts priorities dynamically.

## Context
## Reference
Source: [[OS CPU Scheduling]]