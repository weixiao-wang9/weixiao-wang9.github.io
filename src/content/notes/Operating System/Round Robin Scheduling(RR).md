---
date: 202512142254
tags:
  - atomic
  - concept
source: "[[OS CPU Scheduling]]"
---

# Round Robin Scheduling(RR)

## Definition
### Round Robin Scheduling(RR)
Use when tasks have ***equal priority***
Basic (non-preemptive) roll:
* Put all ready tasks in a queue
* Run T1 until completion, then T2, then T3, etc
Combine with timeslicing:
* Each task gets a fixed time quantum
* After its quantum expires, it is preempted and moved to the back of the queue.
Benefits: Good responsiveness for interactive tasks. No starvation

## Context
## Reference
Source: [[OS CPU Scheduling]]