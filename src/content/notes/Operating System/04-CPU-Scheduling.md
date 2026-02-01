---
type: source
course: "[[Operating System]]"
---

# CPU Scheduling

## Foundation & Mechanism

> [!NOTE] Core Definition
> **The Scheduler** decides which task (thread/process) runs on the CPU and when.
>
> **The Loop**:
> 1. **Ready Queue**: Set of tasks ready to run.
> 2. **Trigger**: CPU becomes idle (task finishes, blocks on I/O, or timeslice expires).
> 3. **Context Switch**: Scheduler picks a new task and swaps context.

### Scheduling Metrics

| Metric | Definition | Goal |
| :--- | :--- | :--- |
| **Throughput** | Jobs completed per unit time | Maximize |
| **Wait Time** | Time spent waiting in the ready queue | Minimize |
| **Completion Time** | Total time from arrival to finish | Minimize |
| **CPU Utilization** | Fraction of time CPU is doing work (not idle) | Maximize |
| **Responsiveness** | How quickly the system responds to user input | Maximize |

---

## Scheduling Algorithms

### Non-Preemptive Scheduling

*Assumption: Once a task starts, it runs until it finishes or blocks voluntarily.*

#### FCFS (First Come First Serve)

Tasks run in order of arrival; no preemption.

**Example**:
* T1: 1s, T2: 10s, T3: 1s
* Arrival order: T1, T2, T3

**Timeline**:
* 0–1: T1 (finishes at 1)
* 1–11: T2 (finishes at 11)
* 11–12: T3 (finishes at 12)

**Metrics**:
* Throughput = 3 / 12 = 0.25 tasks/s
* Completion times = [1, 11, 12], average = (1 + 11 + 12) / 3 = 8
* Wait times:
  - T1: 0
  - T2: waited 1s before starting
  - T3: waited 11s
  - Average = (0 + 1 + 11) / 3 = 4

**Runqueue implementation**: Simple FIFO queue

**Problem**: A big job arriving early can make shorter jobs wait a lot → poor average completion/wait times. This is called the **convoy effect**.

#### SJF (Shortest Job First)

**Idea**: Always run the task with the shortest remaining execution time (for run-to-completion).

* Improves average completion and wait times compared with FCFS
* Needs a runqueue where you can efficiently find the shortest job

> Data structures: sorted list, min-heap, or balanced tree

**Problems**:
* SJF assumes you know how long jobs will take, which is unrealistic. In practice, we approximate based on history.
* Possible **starvation** of long jobs if short ones keep arriving.

**Mechanics** (Preemptive variant - SRTF):
* A task arrives → insert it into the runqueue
* The scheduler checks: "Is the new job shorter than the remaining time of the current job?"
  - If yes → preempt current job and run new one
  - If no → continue running current job

---

### Preemptive Scheduling

*Modern Standard: Tasks are forced to stop when their time is up.*

#### Round Robin Scheduling (RR)

Use when tasks have **equal priority**.

**Basic (non-preemptive) rule**:
* Put all ready tasks in a queue
* Run T1 until completion, then T2, then T3, etc.

**Combine with timeslicing**:
* Each task gets a fixed **time quantum** (timeslice)
* After its quantum expires, it is preempted and moved to the back of the queue

**Benefits**:
* Good responsiveness for interactive tasks
* No starvation
* Fair sharing of CPU

**Trade-offs**:
* Too short timeslice → high context switch overhead
* Too long timeslice → poor responsiveness

---

#### Priority Scheduling

We assign each task a priority.

**Rules**:
* Always run the highest-priority runnable task
* Must support preemption: If a higher priority task becomes ready, it preempts any lower-priority task

**Real systems**:
* Kernel tasks → high priority
* User apps → lower priority

**Implementation**:
* Per-priority runqueues: one queue per priority level

**Scheduler**:
* Find highest non-empty priority queue
* Dequeue head of that queue

**Starvation and Priority Aging**:

**Problem**: Low-priority tasks can starve if high-priority tasks keep arriving.

**Solution**: Priority aging

```
Effective priority = function(actual priority, waiting time)
```

The longer a task waits, the more its effective priority is boosted. Eventually it bubbles up high enough to get CPU time.

---

#### Priority Inversion

**Classic subtle bug**.

**Setup**:
* Priority: P1 (high), P2 (medium), P3 (low) where P3 < P2 < P1
* T3 (low) is running and locks a mutex (e.g., shared data)
* T2 (medium) arrives → preempts T3 (because P2 > P3)
* T1 (high) arrives → preempts T2
* T1 now tries to lock the mutex held by T3 → **T1 blocks**, waiting for T3

**Result**: A lower-priority thread is effectively blocking a higher-priority one → priority inversion

**Fix: Priority Inheritance**

When a high-priority task is blocked on a lock held by a low-priority task:
1. Temporarily boost T3's priority to that of T1
2. The scheduler now sees T3 as high priority and runs it
3. T3 runs and releases the mutex
4. T3's priority is dropped back down

---

#### Multilevel Feedback Queue (MLFQ)

Give different tasks different timeslices and priorities **automatically**, based on their behavior.

**Core idea**: Maintain multiple queues, each with:
* A different priority
* A different timeslice length

**Top queue**:
* Highest priority, shortest timeslice (good for I/O bound/interactive)

**Bottom queue**:
* Lowest priority, longest timeslice (good for CPU-bound)

**Heuristic rules**:
1. New tasks start in the topmost queue
2. When a task runs:
   - If it yields before its timeslice ends → likely I/O-bound or interactive → keep it in a high-priority queue
   - If it must be preempted because it used up its timeslice → likely CPU-bound → demote it to a lower queue with a longer quantum
   - If a task in a lower queue starts yielding often, you can boost it back up

> The scheduler learns from how tasks behave and adjusts their queue accordingly. It adapts priorities dynamically.

**Benefits**:
* Automatically balances interactive and CPU-bound tasks
* No need for users to manually set priorities

**Challenges**:
* Gaming the scheduler: tasks could intentionally yield just before timeslice ends to stay in high priority
* Starvation of CPU-bound tasks if too many interactive tasks arrive

---

## Real-World Linux Schedulers

### Linux O(1) Scheduler

**Selecting next task** and **adding a task to runqueue** happens in constant time, independent of number of tasks.

#### Priority Levels

**140 priority levels total**:
* 0-99: real-time tasks
* 100-139: normal timesharing tasks
* Default user process priority roughly 120

**Nice values**: "Nice" means you let others go first. "Not Nice" (negative) means you are selfish and want the CPU now.
* **The Range**: -20 (Most Selfish) to +19 (Most Generous)

#### Timeslice and Feedback

* Each priority has its own timeslice length
* Uses behavior feedback:
  - Tasks that sleep a lot (I/O-bound) → treated as interactive → priority boosted
  - Tasks that run continuously (don't sleep) → CPU-bound → priority lowered

#### Runqueue Structure

**Two arrays per CPU**:
* active array
* expired array

**Each array**:
* Has one linked list (queue) per priority level
* Has a bitmask with bits set for non-empty lists

**Behavior**:
1. Scheduler always chooses from active:
   - Use "find first set bit" CPU instruction on the bitmask to find highest-priority non-empty queue in O(1)
   - Take first task from that list

2. When a running task:
   - Blocks or is preempted before using full timeslice → put back in active
   - Uses entire timeslice → move it to expired

3. When active is empty:
   - Swap active and expired

**Why small timeslices for low-priority tasks?**
* Low-priority tasks only run once all high-priority tasks are done
* If they had huge timeslices, they'd hog CPU when they finally get it, making the system feel sluggish
* With smaller quanta, they get short bursts of CPU and then give it back quickly

---

### Linux CFS (Completely Fair Scheduler)

**Goal**: Provide **fair sharing** of CPU time among tasks, proportional to their priority (weight).

#### Key Concept: Virtual Runtime (vruntime)

* For each task, CFS tracks *how much CPU time it has effectively consumed*
* "Virtual" because it's scaled by priority:
  - High-priority tasks' vruntime increases more slowly
  - Low-priority tasks' vruntime increases more quickly

> High-priority tasks can run longer before being considered "unfair"
> Low-priority tasks are more easily preempted

#### Runqueue Data Structure

* Runqueue is a **red-black tree**
  - Balanced BST → insert/search/remove in O(log n)
* Tree is ordered by vruntime
  - Leftmost node → smallest vruntime → task that has gotten the *least* CPU so far
* Scheduler always chooses the **leftmost** node

#### How It Runs

1. Pick leftmost node (min vruntime)
2. Run that task
3. While it runs:
   - Its vruntime increases (fast or slow, depending on priority)
4. Periodically compare its vruntime to leftmost node's vruntime:
   - If still smallest → keep running it
   - If it has "caught up" or exceeded another task → preempt and put it back in the tree

#### Effects

* **Task selection**: O(1) (cached leftmost pointer)
* **Insertion/removal**: O(log n)
* **Smoother fairness**: tasks share CPU in proportion to priority
* **Better interactivity**: interactive tasks that frequently sleep tend to have lower vruntime, so when they wake up, they appear on the left and get CPU quickly

**Downside**: For huge numbers of tasks, the log n factor of insert/remove might become a bottleneck.

---

## Multiprocessor Scheduling

We have multiple CPUs/cores.

### Architectures

1. **Shared memory multiprocessor**:
   - Multiple CPUs
   - Each CPU: private L1/L2; maybe shared last-level cache
   - Shared main memory (DRAM)

2. **Multicore CPU**:
   - One socket with multiple cores inside
   - Each core: private L1/L2
   - Shared last-level cache on chip
   - DRAM shared

From OS viewpoint: sees multiple logical CPUs (each core or hardware thread is something schedulable).

### Cache Affinity

**Reasoning**:
* CPU caches matter a lot for performance
* If a thread runs on CPU 0, its working set ends up in CPU 0's cache
* If later we run it on CPU 1, we lose that cache state → more cache misses

**Hence**:
* Try to keep tasks on the same CPU they ran on last time → **cache affinity**

### Load Balancing

We usually give each CPU its own runqueue and scheduler, plus a **load balancer**:
* Each CPU mostly schedules from its own queue
* Periodically:
  - Check if some CPUs are overloaded vs. others
  - Move tasks from busy CPU's runqueue to idle/less busy CPU's runqueue

**Trade-off**:
* Load balancing improves fairness and utilization
* But it conflicts with cache affinity (moving tasks loses cache)
* Schedulers must balance these competing goals

---

## Hardware Considerations

### Hyperthreading/SMT

**Hyperthreading (Simultaneous Multithreading)** treats one physical core as two logical CPUs.

**Key Concept**: Hiding memory stalls by treating one core as two.

**Strategy**: Mix CPU-bound and memory-bound threads.
* When one thread stalls waiting for memory, the other can use the execution units

**From OS perspective**:
* Sees two CPUs per core
* Must be aware that they share resources (execution units, caches)
* Scheduling decisions should consider this sharing

---

### NUMA (Non-Uniform Memory Access)

Some systems have multiple memory nodes; each CPU/socket is closer (physically) to some memory banks.

* **Local memory access**: fast
* **Remote memory access**: slower

**Goal**: NUMA-aware scheduling
* Keep threads on CPUs close to the memory where their data resides
* Avoid migrating tasks across NUMA nodes unless necessary

**Implementation**:
* OS tracks which memory pages belong to which NUMA node
* Scheduler tries to keep threads on the node where their memory is allocated
* Page migration policies can move pages closer to where they're accessed

---

## Summary

### Scheduling Algorithm Evolution

1. **FCFS**: Simple but suffers from convoy effect
2. **SJF/SRTF**: Optimal average wait time, but needs job length prediction
3. **Round Robin**: Fair timesharing, good for interactive systems
4. **Priority**: Supports different classes of work, needs anti-starvation mechanisms
5. **MLFQ**: Adaptive, learns task behavior automatically
6. **O(1) Scheduler**: Constant-time operations, priority-based with feedback
7. **CFS**: Proportional fairness, better for modern workloads

### Key Trade-offs

* **Throughput vs. Responsiveness**: Long timeslices improve throughput but hurt responsiveness
* **Fairness vs. Efficiency**: Perfect fairness requires frequent context switches
* **Cache Affinity vs. Load Balancing**: Keeping tasks on same CPU helps cache, but may create imbalance
* **Complexity vs. Performance**: More sophisticated algorithms can perform better but are harder to implement and debug

### Modern Trends

* Move toward fair scheduling (CFS)
* NUMA awareness for large systems
* Energy-aware scheduling for mobile/embedded
* Real-time guarantees for critical tasks
* Container and cgroup-aware scheduling
