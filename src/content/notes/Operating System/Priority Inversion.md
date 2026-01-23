---
date: 202512142254
tags:
  - atomic
  - concept
source: "[[OS CPU Scheduling]]"
---

# Priority Inversion

## Definition
### Priority Inversion
Classic subtle bug.
Setup:
- Priority: P1 (high), P2 (medium), P3 (low) where P3 < P2 < P1.
- T3 (low) is running and locks a mutex (e.g. shared data).
- T2 (medium) arrives → preempts T3 (because P2 > P3).
- T1 (high) arrives → preempts T2.
- T1 now tries to lock the mutex held by T3 → **T1 blocks**, waiting for T3.
result: A lower-priority thread is effectively blocking a higher-priority one -> priority inversion
>(fix) Priority Inheritance
>When a high-priority task is blocked on a lock held by a low-priority task.
>Temporarily Boost T3's priority to that of T1
>The scheduler now sees T3 as high priority and runs it.
>T3 runs and releases the mutex, then its priority is dropped back down

## Context
## Reference
Source: [[OS CPU Scheduling]]