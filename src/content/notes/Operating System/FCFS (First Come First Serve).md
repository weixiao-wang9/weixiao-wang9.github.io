---
date: 202512142254
tags:
  - atomic
  - concept
source: "[[OS CPU Scheduling]]"
---

# FCFS (First Come First Serve)

## Definition
### FCFS (First Come First Serve)
Tasks run in order of arrival; no preemption.
>Example:
T1: 1s, T2: 10s, T3: 1s; arrival order T1, T2, T3.
Timeline:
- 0–1: T1 (finishes at 1)
- 1–11: T2 (finishes at 11)
- 11–12: T3 (finishes at 12)
>Metrics:
* Throughput = 3 / 12 = 0.25 tasks/s
- Completion times = [1, 11, 12], average = (1 + 11 + 12) / 3 = 8
- Wait times =
    - T1: 0
    - T2: waited 1s before starting
    - T3: waited 11s
        → average = (0 + 1 + 11) / 3 = 4
Runqueue implementation: simple FIFO queue
***Problem***: a big job arriving early can make shorter jobs wait a lot -> poor average completion/wait times.

***SJF (Shortest Job First)***
Idea: Always run the task with the shortest remaining execution time( for run-to-completion)
* Improves average completion and wait times compared with FCFS
* Needs a runqueue where you can efficiently find the shortest job
>Data structures: sorted list, min-heap or balanced tree

SJF assumes you know how long jobs will take, which is unrealistic. In practice, we approximate based on history

Mechanics:
* A task arrives -> we insert it into the runqueue
* The scheduler checks:" is the new job shorter than the remaining time of the current job?"
	* if yes -> preempt current job and run new one
	* if no -> continue running current job
Problem: need to estimate job length and possible starvation of long jobs if short ones keep arriving.

## Context
## Reference
Source: [[OS CPU Scheduling]]