---
date: 202512142254
tags:
  - atomic
  - concept
source: "[[OS CPU Scheduling]]"
---

# Linux O(1) Scheduler

## Definition
### Linux O(1) Scheduler
Selecting next task
Adding a task to runqueue
This happens in constant time, independent of number of tasks

***Priority levels***
140 priority levels total.
0-99: real-time tasks
100-139: normal timesharing tasks
Default user process priority roughly 120
"nice values" you are "Nice," you let others go first. If you are "Not Nice" (negative), you are selfish and want the CPU now.
- **The Range:** -20 (Most Selfish) to +19 (Most Generous).
Timeslice and feedback
Eaach priority has its own timeslice length
Uses behavior feedback:
	Tasks that sleep a lot(requires I/O) -> treated as interactive -> priority boosted
	Tasks that run continuously(don't sleep) -> CPU-bound -> priority lowered

Runqueue Structure:
Two arrays per CPU:
* active array
* expired array
Each array: 
has one linked list (queue) per priority level; 
has a bitmask with bits set for non-empty lists.
***Behavior:***
1. Scheduler always chooses from active:
		Use "find first set bit" CPU instruction on the bitmask to find highest-priority non-empty queue in O(1)
		Take first task from that list
2. When a running task:
	Blocks or is preempted before using full timeslice
	-> put back in active
	Uses entire timeslice -> move it to expired
3. When active is empty:
	Swap active and expired.
Why small timeslices for low-priority tasks?
* Low-priority tasks only run once all high-priority tasks are done
* If they had huge timeslices, they'd hog CPU when they finally get it, making the system feel sluggish
* With smaller quanta, they get short bursts of CPU and then give it back quickly

## Context
## Reference
Source: [[OS CPU Scheduling]]