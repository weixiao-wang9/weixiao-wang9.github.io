---
type: source
course: "[[Operating System]]"
---

# Processes and Process Management

## What is a Process?

An **application** is just a program stored on disk. At this stage, it's static (not running).

* When you **launch the application**, the operating system loads it into memory and starts executing it
* At that moment, the application becomes a **process**

A **process** is an active entity.

---

## What a Process Looks Like in Memory

### Encapsulation of Process State

A process contains all the information needed for a running application:
* **Code**: The program instructions that are executed
* **Data**: Variables and constants that are initialized when the program first loads
* **Heap**: Memory ***dynamically*** allocated during execution
* **Stack**: Memory used for function calls, local variables, and return addresses

All of this together forms the **state of the process**.

### Address Space

The operating system gives each process its own **address space** - a continuous range of virtual memory addresses from $V_0$ to $V_{max}$.

This isolates processes from one another.

Inside this space, different sections are laid out in different regions:

* **Text (Code)** at the bottom: compiled instructions of the program
* **Data** above it: static/global variables initialized at load time
* **Heap** grows upwards: used for dynamic memory allocation (malloc, calloc, etc.)
* **Stack** grows downwards: keeps track of function calls (stack frames), parameters, local variables, and return addresses
  - The stack grows and shrinks dynamically as functions are called and return (LIFO behavior)

### Virtual Addresses

The potential range of addresses in process address space go from V_0 to V_max. They are **virtual** because they don't have to correspond to actual locations in physical memory.

* The operating system (with help from the Memory Management Unit, MMU) maps each virtual address to a physical address
* This mapping allows the layout in virtual memory to be independent of the layout in physical memory (OS can move the data around in RAM without the process ever knowing)
* The mapping between virtual address and physical address is stored in a data structure called a **page table** (each process has its own page table)

This enables: **Isolation, Flexibility, Security**

---

## Execution State of a Process

The operating system must track what a process is doing at any moment.

If the OS stops a process, it needs to remember exactly where it left off so the process can resume seamlessly.

### Key Components of Process Execution

**Program Counter (PC)**:
* A special CPU register that stores the address of the next instruction to execute
* This tells the CPU exactly where the process is in its code

**CPU Registers**:
* Store temporary data
* These must be saved/restored when switching between processes

**Stack and Stack Pointer (SP)**:
* Each process has its own stack
* The stack pointer register keeps track of the top of this stack

---

## Process Control Block (PCB)

The OS keeps all of this information (PC, registers, stack pointer, memory info, state, etc.) in a data structure called the **Process Control Block (PCB)**.

The PCB is crucial for **context switching**.

### PCB Contains:
* Process state
* Process number
* Program counter
* CPU registers
* Memory limits
* CPU scheduling info
* And more

The PCB is created when the process is initially created and is also initialized at that time. Certain fields of the PCB may be changed when process state changes (some fields can change often).

### How Execution Works with PCB

PCB is stored in memory on behalf of each process. It holds all the critical execution state:

* **When a process is scheduled to run**:
  - The OS **loads the PCB** of that process into the CPU registers
  - The CPU then executes instructions from the state saved in the PCB

* **When the process is interrupted** (e.g., time slice is over, or I/O event occurs):
  - The OS **saves the process's state back into its PCB** in memory
  - That way, the process can resume later exactly where it left off

---

## Context Switching

A **context switch** happens when the CPU switches from one process (say P1) to another (say P2).

This operation is **EXPENSIVE**:

**Direct cost**: Number of CPU cycles required to load and store a new PCB to and from memory

**Indirect cost**: Even though the OS can switch processes quickly by saving/restoring the PCB, the **loss of cache data** makes the resumed process slower initially. Too many context switches reduce overall system performance.

> **Hot cache**: If the process is currently running and its needed data is already in cache, execution is very efficient

> **Cold cache**: If the process was swapped out (due to context switch), its cache data gets cleared. When it resumes later, cache is empty and the CPU has to refill it from memory.

---

## Process Life Cycle: States

1. **New**: When a process is first created, it is in the new state. The OS does admission control (allocates a PCB and initializes resources). The process is not yet ready to run.

2. **Ready**: Once initialized, the process moves to the ready state. It is prepared to run but is waiting for the CPU to become available.

3. **Running**: When the scheduler picks a process from the ready queue, it goes into the **running** state. The process instructions are now being executed on the CPU (the active state of execution).

4. **Interrupt or I/O**:
   - a. **Interrupt / Time Slice Ends** → The process is moved back to **ready** (so another process can use the CPU)
   - b. **I/O request or event wait** → The process moves to the **waiting** state

5. **Wait**: The process is waiting for some event (e.g., I/O completion, network packet arrival). It is not using the CPU, but is still active in the system.

6. **Terminated**: When a process finishes execution (normally or due to an error), it moves into the **terminated** state. The OS cleans up its resources and removes its PCB.

---

## Process Creation

A process can create child processes. All processes form a tree-like hierarchy.

When the OS boots, it creates **root processes** with special (privileged) access. These root processes then create other processes, forming the process hierarchy.

### Two Main Mechanisms of Process Creation

**fork()**:
* The OS makes a new **PCB** for the child
* It copies all the values from the parent's PCB (program counter, registers, stack, etc.)
* As a result, the child process is an almost identical copy of the parent
* Both parent and child resume execution **right after the fork call**, with the same program counter (but they can tell which is which because fork() returns different values: 0 for child, PID for parent)

**exec()**:
* Replaces the process's memory with a **new program**
* The child still has its PCB (from fork), but its contents are overwritten with a new program's values
* Now the program counter points to the first instruction of the new program
* Used when you want to run a different program (e.g., ls, bash, etc.) instead of just copying the parent

---

## Role of the CPU Scheduler

The CPU scheduler manages how processes use the CPU resources. It determines which process will use the CPU next, and how long that process has to run.

### OS Must:
* **Preempt**: Interrupt and save current context
* **Schedule**: Run scheduler to choose next process
* **Dispatch**: Dispatch process and switch into its context

### CPU Efficiency

The CPU's efficiency is defined as:

$$\text{Efficiency} = \frac{\text{Time spent executing processes}}{\text{Total computation time}}$$

* **Numerator (useful work)**: How long the CPU actually spends running processes
* **Denominator (total time)**: Includes both **execution time** and **scheduling overhead** (time spent switching between processes)

> Every time the OS does a **context switch** (saving PCB, restoring another), it uses CPU cycles that don't directly help execute a process. This time is called **scheduling overhead**. If context switches happen too often, a lot of CPU time is wasted.

### Timeslice

The **timeslice** is how long a process gets to run before the scheduler interrupts it.

* If the timeslice is **too short**:
  - Many context switches → high overhead → poor efficiency

* If the timeslice is **too long**:
  - Efficiency is high, but responsiveness is poor (other processes wait too long)

> **Design Question**: What are appropriate timeslice values? What criteria is used to decide which process is the next to run?

For **I/O** requests, the process is sent back to the ready queue.

---

# Threads

## Process vs. Thread

A single-threaded process is represented by two components:
* **Address space** (virtual ↔ physical memory mapping)
* **Execution context** (CPU registers, stack)

All of this information is represented by the OS in a process control block.

A **thread** is a smaller unit of execution within a process:
* Multiple threads run inside the same address space
* They share code, data, files/resources
* But each thread has its own **execution context**: its own registers, program counter, stack

Each thread has its own data structure to represent information specific to its execution context.

---

## Multithreading

### Benefits

1. **Parallelism**: While each thread is executing the same code, each thread may be executing a different instruction (in the sense of different line or function) → parallel execution → speed up the program's execution

2. **Specialization**: We can give higher priority to tasks that handle more important tasks or service higher paying customers → each thread may only need a small amount of data/code → the entire state might fit into the cache (fast)

> Memory requirements for a multiprocess application are greater than those of a multithreaded application.

### Even on a Single CPU

* Context switching between **processes** is expensive because:
  - The OS must reload **virtual-to-physical address mappings**

* Context switching between **threads in the same process** is cheaper because:
  - They share the **same address space and mappings**
  - $t_{ctx\_switch\_thread} < t_{ctx\_switch\_process}$

Even though only **one thread runs at a time**, threads let the CPU stay busy by switching when one is waiting.

---

## Synchronization Mechanisms

Since threads share the same virtual to physical address mappings, and they share the same address space, we could see **data race** problems: One thread can try to read the data while another modifies it, leading to inconsistencies.

### Mutual Exclusion (Mutex)

**Mutual exclusion** (through mutex): Only one thread at a time is granted access to some data. The remaining threads must wait their turn.

### Condition Variable

A thread can wait on another thread, and be able to exactly specify what condition the thread is waiting on.

---

## Thread Creation

We need some data structure to represent a thread:
* Thread ID
* Program counter
* Stack pointer
* Register values
* Stack
* Other attributes (priority attributes)

Creating a thread is conceptually like a **fork** (not the UNIX fork(), but similar idea):
* You specify:
  - The **procedure (proc)** the thread should run
  - The **arguments (args)** to pass to it
* The new thread is created with its **own program counter pointing to the start of proc**, and its own registers + stack
* Now the process has multiple threads, all sharing the same address space, but each thread runs independently

### Thread Join

Once a child thread finishes, we need a way for it to return results or signal the parent. We also need to ensure the parent doesn't exit before its child threads (otherwise the process would die and kill the threads).

The parent thread can call **join** on a child thread:
* This means:
  - The parent is **blocked** until the child finishes
  - Once done, the child's result is returned
  - The child thread then exits, and its resources are deallocated

This ensures proper synchronization — parent waits for child before moving on if necessary.

### Example

```c
Thread thread1;
Shared_list list;
thread1 = fork(safe_insert, 4);
safe_insert(6);
join(thread1);
```

The issue is that the order of parent thread and child is unpredictable unless we synchronize (with join or locks).

---

## Multithreading Models

### 1. One-to-One Model

**Mapping**: Each user-level thread = 1 kernel-level thread

When you create a thread in your program, the kernel creates a real kernel thread too.

**Pros**:
* Kernel fully understands threading (better scheduling, true parallelism on multi-CPU systems)
* Each thread can be managed independently

**Cons**:
* Expensive
* Limited by OS policies and different OS have different rules

### 2. Many-to-One Model

**Mapping**: Many User threads → 1 Kernel thread

The user-level thread library decides which user thread runs, but the kernel only sees **one** thread.

**Pros**:
* Portable (not tied to OS kernel support)
* Fast thread operations (no system calls needed)

**Cons**:
* Kernel doesn't even know it's multithreaded
* If one user thread blocks (e.g., I/O), the **whole process blocks**
* No true parallelism (only 1 kernel thread exists)

### 3. Many-to-Many Model

**Mapping**: Many User threads ↔ Many Kernel threads

Kernel is aware of multithreading, but not every user thread must have a dedicated kernel thread.

Supports:
* **Bound threads** → fixed mapping (user thread always tied to same kernel thread)
* **Unbound threads** → flexible mapping (user thread can run on any available kernel thread)

**Pros**:
* Best of both worlds → concurrency + flexibility
* If one kernel thread blocks, others can continue
* Good scalability for large apps

**Cons**:
* More complex (needs coordination between user-level library and kernel-level scheduler)

---

## Scope of Multithreading

### Process Scope (User-Level)
* Threads managed by the **user-level library** inside one process
* The kernel only sees the process, not the individual threads
* Example: Process A has 4 threads, Process B has 2 threads → kernel might still give A and B the _same CPU share_, so each thread in A gets **less time** than each thread in B

### System Scope (Kernel-Level)
* Threads are visible to the **kernel scheduler**
* Kernel allocates CPU to _all threads across the system_
* Example: A has 4 threads, B has 2 threads → kernel may give A twice the CPU share as B, since A has more threads

**Difference**: With **process scope**, fairness is at the _process level_. With **system scope**, fairness is at the _thread level_.

---

## Multithreading Patterns

### Boss/Workers Pattern

**Setup**:
* 1 boss thread → hands out work
* Multiple worker threads → actually do the work

**Variants**:
* **Direct assignment** → boss tracks idle workers and assigns tasks
  - Simpler worker logic, but boss is overloaded
* **Work queue** → boss just drops tasks into a queue, workers pull tasks
  - Reduces boss overhead, but requires worker synchronization

**Scaling**:
* Use a **thread pool** to avoid creating/destroying threads each time
* Can dynamically grow/shrink pool size depending on load

**Pros**: Simple, easy to understand

**Cons**: Bottleneck at boss thread; load balancing tricky

### Pipeline Pattern

**Setup**: Break work into **stages** (like an assembly line)

Example: A 6-step process → 6 thread types, each for one stage

**Execution**:
* Tasks flow stage by stage
* At any time, different tasks are in different pipeline stages

**Scaling**: If one stage is slow (the **bottleneck**), add more threads to that stage

**Pros**: Specialization, good cache locality, natural concurrency

**Cons**: Hard to keep balanced; bursts of input can overwhelm a stage

### Layered Pattern

**Setup**: Group related subtasks into **layers**

* Each layer has multiple threads that can perform _any subtask_ in that layer
* Work flows through all layers in order

**Example**:
* Layer 1: Input handling
* Layer 2: Processing
* Layer 3: Output/storage

**Pros**: Some specialization without being as rigid as a pipeline

**Cons**: Synchronization between layers can be complex
