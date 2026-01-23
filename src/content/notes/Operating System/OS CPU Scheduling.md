
## 1. Foundation & Mechanism
> [!NOTE] Core Definition
> **The Scheduler** decides which task (thread/process) runs on the CPU and when.
>
> **The Loop**:
> 1. **Ready Queue**: Set of tasks ready to run.
> 2. **Trigger**: CPU becomes idle (Task finishes, blocks on I/O, or Timeslice expires).
> 3. **Context Switch**: Scheduler picks a new task and swaps context.

### Metrics
| Metric | Definition | Goal |
| :--- | :--- | :--- |
| **Throughput** | Jobs completed per unit time. | Maximize |
| **Wait Time** | Time spent waiting in the ready queue. | Minimize |
| **CPU Utilization** | Fraction of time CPU is doing work (not idle). | Maximize |

---

## 2. Scheduling Algorithms

### Non-preemptive (Run-to-Completion)
*Assumption: Once a task starts, it runs until it finishes or blocks voluntary.*
- [[Run-to-Completion scheduling]] (Base assumptions)
- [[FCFS (First Come First Serve)]]: Simple, but suffers from convoy effect.

### Preemptive
*Modern Standard: Tasks are forced to stop when their time is up.*
- [[Round Robin Scheduling(RR)]]: Based on **Timeslices**. Fair but context-switch heavy.
- [[Priority Scheduling]]: High priority runs first (Watch out for [[Priority Inversion]]).
- [[Multilevel Feedback Queue (MLFQ)]]: Dynamic priority adjustment based on behavior.
- [[Linux O(1) Scheduler]] & [[Linus CFS (Completely Fair Scheduler)]]

---

## 3. Hardware & Multicore
- [[Multiprocessor Scheduling]]: Complexity of scheduling across multiple chips.
    - Key Issues: [[Load Balancing]] and [[NUMA(Non-Uniform Memory Access)]].
- [[HyperthreadingSMT]]: **(Key Concept)** Hiding memory stalls by treating one core as two.
    - *Strategy: Mix CPU-bound and Memory-bound threads.*
    