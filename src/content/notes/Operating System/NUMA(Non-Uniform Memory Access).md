---
date: 202512142254
tags:
  - atomic
  - concept
source: "[[OS CPU Scheduling]]"
---

# NUMA(Non-Uniform Memory Access)

## Definition
### NUMA(Non-Uniform Memory Access)
Some systems have multiple memory nodes; each CPU/socket is closer (physically) to some memory banks.
	Local memory access: fast
	Remote memory access: slower
Goal: NUMA-aware scheduling
- Keep threads on CPUs close to the memory where their data resides
- Avoid migrating tasks across NUMA nodes unless necessary

## Context
## Reference
Source: [[OS CPU Scheduling]]