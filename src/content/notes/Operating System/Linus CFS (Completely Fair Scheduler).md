---
date: 202512142254
tags:
  - atomic
  - concept
source: "[[OS CPU Scheduling]]"
---

# Linus CFS (Completely Fair Scheduler)

## Definition
### Linus CFS (Completely Fair Scheduler)
Goal: Provide ***fair sharing*** of CPU time among tasks, proportional to their priority (weight).

Key concept: virtual runtime (vruntime)
* For each tasks, CFS tracks *how much CPU time it has effectively consumed.*
* "Virtual" because it's scaled by priority:
	* High-priority tasks' vruntime increases more slowly
	* Low-priority tasks' vruntime increases more quickly
	>High-priority tasks can run longer before being considered "fair"
>	Low-priority tasks are more easily preempted

Runqueue data structure:
* Runqueue is a red-black tree
	* Balanced BST -> insert/search/remove in O(log n)
* Tree is ordered by vruntime.
	 Leftmost node -> smallest vruntime -> task that has gotten the *least* CPU so far.
* Scheduler always chooses the **leftmost** node

How it runs:
	1. Pick leftmost node (min vruntime)
	2. Run that task
	3. While it runs:
		Its vruntime increases (fast or slow, depending on priority)
	4. Periodically compare its vruntime to leftmost node's vruntime:
		If still smallest -> keep running it
		If it has "caught up" or exceeded another task -> preempt and put it back in the tree
Effects:
- Task selection: O(1)
- Insertion/removal: O(log n)
- Smoother fairness: tasks share CPU in proportion to priority
- Better interactivity: interactive tasks that frequently sleep tend to have lower vruntime, so when they wake up, they appear on the left and get CPU quickly

Downside: For huge numbers of tasks, the log n factor of insert/remove might become a bottleneck

## Context
## Reference
Source: [[OS CPU Scheduling]]