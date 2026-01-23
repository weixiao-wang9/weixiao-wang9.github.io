---
date: 202512142254
tags:
  - atomic
  - concept
source: "[[OS CPU Scheduling]]"
---

# Multiprocessor Scheduling

## Definition
### Multiprocessor Scheduling
We have multiple CPUs/cores

Architectures:
1. Shared memory multiprocessor
	Multiple CPUs
	Each CPU: private L1/L2; maybe shared last-level cache.
	Shared main memory (DRAM)
2. Multicore CPU
	One socket with multiple cores inside
	Each core: private L1/L2
	Shared last-level cache on chip
	DRAM shared
From OS viewpoint: sees multiple logical CPUs (each core or hardware thread is something)

***Cache Affinity***
Reasoning:
	CPU caches matter a lot for performance
	If a thread runs on CPU 0, its working set ends up in CPU 0's cache.
	If later we run it on CPU 1, we lose that cache state -> more cache misses.
Hence:

Try to keep tasks on the same CPU they run on last time --> ***cache affinity***

## Context
## Reference
Source: [[OS CPU Scheduling]]