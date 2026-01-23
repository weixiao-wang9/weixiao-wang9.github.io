---
date: 202512142254
tags:
  - atomic
  - concept
source: "[[OS CPU Scheduling]]"
---

# Priority Scheduling

## Definition
### Priority Scheduling
We assign each task a priority
Rules:
* Always run the highest-priority runnable task
* Must support preemption: If a higher priority task becomes ready, it preempts any lower-priority task
Real systems:
	Kernel task -> high priority
	User apps -> highest
Implementation:
* Pre-priority runqueues: one queue per priority level
Scheduler:
* Find highest non-empty priority queue
* Dequeue head of that queue
***Starvation and priority Aging***
Problem: Low-priority tasks can starve if high-priority tasks keep arriving.
Solution:Priority aging
Effective priority = function(actual priority, waiting time)
The longer a task waits, the more its effective priority is boosted.
Eventually it bubbles up high enough to get CPU time

## Context
## Reference
Source: [[OS CPU Scheduling]]