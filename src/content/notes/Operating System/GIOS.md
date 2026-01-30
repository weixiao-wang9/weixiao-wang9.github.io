**Operating System:**
An operating system is a layer of systems software that:
* directly has privileged access to the underlying hardware;
* manages hardware on behalf of one or more applications according to some predefined policies.
* ensures that applications are isolated and protected from one another

**The computing system consists of :**
1. Central Processing Unit(CPU); 
2. Physical Memory; 
3. Network interfaces(Ethernet/Wifi Card); 
4. GPU; 
5. Storage Device(Disk, Flash drives);
In addition, a computing system can have higher level applications -- programs

**Functions of an Operation System**
1. Operating system hide hardware complexity: provides a higher level abstraction.
2. Operating systems manage underlying hardware resources: it allocates memory for applications, schedules them for execution on the CPU, controls access to various network devices and so on.
3. Provides isolation and protection: When applications are running concurrently, the operating system has to ensure that they can do what they need to without hurting one another.

Example:
For desktop operating systems we have:
	Microsoft Windows
	Unix-based systems OS X Linux
For embedded operating systems:
	Android
	IOS


***OS Elements***
An operating system provides a number of high level abstractions, as well as a number of mechanisms to operate on these abstractions.

Abstractions: 
process, thread(application abstraction)
file, socket, memory page(hardware abstractions)

Corresponding mechanisms:
create/schedule
open/write/allocate

Operating systems may also integrate specific policies that determine exactly how the mechanisms will be used to manage the underlying hardware. (e.g. a policy could determine the maximum number of sockets that a process has access to)

Memory management example:
The main abstraction is memory page, which corresponding to some addressable region of memory of some fixed size.
The operating system integrates mechanisms to operate on the page like `allocate`, which allocates the memory in DRAM. It can also map that page into the address space of the process, so that the process can interact with the underlying DRAM.
The page may be moved to different spaces of memory later on use policy(LRU)

### OS Design Principle
1. Separation of mechanism and policy: we want to incorporate flexible mechanisms that can support a number of policies
2. Optimize for the common case: Where will the OS be used? What will the user want to execute on that machine? What are the workload requirements?
### OS Protection Boundary
computer systems distinguish between at least two modes of execution:
* user-level(unprivileged)
* kernel-level(privileged)
OS must have direct access to hardware, it must operate in kernel mode.
Hardware access can only be utilized in the kernel mode from the OS directly.

Crossing from user-level to kernel-level is supported by most modern operating systems
Applications usually operate in user-mode：When privileged instructions are encountered during a non-privileged execution, the application will be trapped. This means the application's execution will be interrupted, and the control will be handed back to the OS.
1. The OS can decided whether to grant the access or potentially terminate the process.
2. The OS also exposes an interface of system calls, which the application can invoke, which allows privileged access of hardware resources for the application.
3. The OS also support signals. which is a way for the operating system to send notification to the application

***System Call Flow:***
Running a process
- You start with a **process** (a program that’s currently running).
- Sometimes, that process needs to use **hardware resources** (like disk, network, keyboard, etc.).
- But! Processes run in **user mode**, where they _can’t_ directly access hardware — only the **OS kernel** can.
---
Making a System Call

- The process asks the OS for help via a **system call**.
- This is basically the process saying: _“Hey OS, can you do this privileged thing for me?”_
- To do that, control **switches** from user mode → kernel mode.
- The OS executes the requested operation (maybe accessing hardware).
- Once finished, the OS passes control (and results/data) back to the process (kernel mode → user mode).
---
Cost of Context Switching

- Switching from user mode ↔ kernel mode is called a **context switch**.
- This is **not free** — it takes CPU cycles to save state, switch privileges, and restore state.
- That’s why the text says: _“Not necessarily a cheap operation to make a system call!”_
---
Passing Arguments to the System Call

- When making a system call, the process may need to pass arguments (e.g., file name, buffer, size).
- This can be done in two ways:
    1. **Directly**: arguments copied into registers/stack and passed into the kernel.
    2. **Indirectly**: pass a **pointer** to where the data is stored in memory, and the OS reads from there.
---
Synchronous vs. Asynchronous Mode

- **Synchronous mode**:
    - The process **waits** until the system call finishes.
    - Example: reading from a file — you can’t continue until the data arrives
- **Asynchronous mode**:
    - The process can continue doing other work while the OS finishes the request in the background.
    - Example: non-blocking I/O.
    
### Crossing the OS Boundary

User/Kernel transitions are common and useful throughout the course of application execution.

The hardware supports user/kernel transitions: the hardware will cause a trap on illegal executions that require special privilege. Then it initiates transfer of control from process to operating system when a trap occurs.
User/Kernel transition requires instructions to execute, which can take 100ns on a 2Ghz Linus box.
In addition, the OS may bring some data into the hardware cache, which will bounce out some other memory used by another application.

### OS Service
An operating system provides applications with the access to the underlying hardware. It's the layer between applications and hardware. The OS exposes certain services that map almost directly to hardware components:
***Service directly linked to hardware***
* Scheduling component (CPU) : decides which process gets CPU time and when.
* Memory manager(physical memory): keeps track of what part of RAM is used/free, allocates memory safety to processes.
* Block device driver(disk/storage) lets applications read/write to storage without knowing the details of the hardware.
***Higher level services***
* Process management
* File Management
* Device management
* Memory management
* Storage management
* Security

**Monolithic OS:**
Everything is included
pros: everything is included/inlining, compile time optimization
cons: no customization/not too portable/large memory footprint

---
**Modular OS:**
A type of operating system has a basic set of services and APIs that come with it. Anything not included can be added as a module. It can dynamically install new modules in the operating system
pros:maintainability/smaller footprint/less resource needs
cons:all the modularity/indirection can reduce some opportunities for optimization; maintenance can still be an issue as modules from different codebases can be slung together at runtime.

---
**Microkernel**
Only requires the most basic operating system components. Everything else will run outside of the operating system at user-level. This setup requires lots of interprocess communication, as the traditional operating system components run within application process.
> the microkernel often supports IPC as a core abstraction

Pros: size/verifiability(great for embedded device)
Cons:bad portability/harder to find common OS components due to specialized use case/expensive cost of frequent user/kernel crossing

---
### Linus and Mac OS architecture
***Linus***
* Hardware
* Linus Kernel
* Standard libraries
* Utility program
* User application
kernel consist of several logical components
* Virtual file system
* Memory management
* Process management
***Mac OS X***
* I/O kit for device drivers
* Kernel extension kit for dynamic loading of kernel components
* Mach microkernel memory management thread scheduling IPC
* BSD component Unix interoperability POSIX API support Network I/O interface]
* All applications sit above this layer

---

### Processes and Process Management

Operating system manage the hardware on behalf of applications.

An ***application*** is just a program stored on disk. At this stage, it's static(not running)
- When you **launch the application**, the operating system loads it into memory and starts executing it.
- At that moment, the application becomes a **process**.
A ***Process*** is an active entity.

### What a process looks like in memory
***Encapsulation of Process State***
A process contains all the information needed for a running application:
- Code: The program instructions that are executed
- Data: Variables and constants that are initialized when the program first loads
- Heap: Memory ***dynamically*** allocated during execution
- Stack: Memory used for function calls, local variables, and return address.
All of this together forms the ***the state of the process***

***Address Space***
The operating system gives each process its own ***Address space*** - a continuous range of virtual memory addresses from $V_{0} \quad V_{max}$
This isolates processes from one another
Inside this space, different sections are laid out in different regions.
![](</images/Screenshot 2025-10-03 at 2.34.15 PM.png>)
From the diagram:
- **Text (Code)** at the bottom: compiled instructions of the program.
- **Data** above it: static/global variables initialized at load time.
- **Heap** grows upwards: used for dynamic memory allocation (malloc, calloc, etc.).
- **Stack** grows downwards: keeps track of function calls (stack frames), parameters, local variables, and return addresses.
    - The stack grows and shrinks dynamically as functions are called and return (LIFO behavior).
---


***Virtual Addresses***: the potential range of addresses in process address go from V_0 to V_max. （They are virtual because they don't have to correspond to actual locations in the physical memory.

- The operating system (with help from the Memory Management Unit, MMU) maps each virtual address to a physical address.
- This mapping allows the layout in virtual memory to be independent of the layout in physical memory(OS can move the data around in RAM without the process ever knowing)
- The mapping between virtual address and physical address is stored in a data structure called a page table.( Each process has its own page table)
This enables: ***Isolation, Flexibility, Security***

#### Execution state of a process
The operating system must track what a process is doing at any moment

If the OS stops a process, it needs to remember exactly where it left off so the process can resume seamlessly.

***Keep Components of Process Execution***:
Program Counter (PC):
- A special CPU register that stores the address of the next instruction to execute
- This tells the CPU exactly where the process is in its code.
CPU Register:
- Store temporary data 
- These must be saved/restored when switching between processes.
Stack and Stack Pointer (SP):
- Each process has its own stack
- The stack pointer register keeps track of the top of this stack

#### Process Control Block(PCB):
- The OS keeps all of this information (PC, registers, stack pointer, memory info, state, etc.) in a data structure called the **Process Control Block (PCB)**.
- The PCB is crucial for **context switching**:

It contains:
- Process state
- process number
- program counter
- CPU register
- memory limits
- CPU scheduling info and more
PCB is created when the process is initially created and is also initialized at that time. Certain fields of the PCB may be changed when process state changes. ( Some fields can change often)
---

#### How Execution Works with PCB
PCB is stored in memory on behalf of each process
It holds all the critical execution state:
- When a process is scheduled to run:
    - The OS **loads the PCB** of that process into the CPU registers.
    - The CPU then executes instructions from the state saved in the PCB.
- When the process is **interrupted** (e.g., time slice is over, or I/O event occurs):
    - The OS **saves the process’s state back into its PCB** in memory.
    - That way, the process can resume later exactly where it left off.

#### Context Switching
A **context switch** happens when the CPU switches from one process (say P1) to another (say P2).

This operation is EXPENSIVE:
direct cost: number of CPU cycles required to load and store a new PCB to and from memory
indirect cost: Even though the OS can switch processes quickly by saving/restoring the PCB, the **loss of cache data** makes the resumed process slower initially. (too many context switches reduce overall system performance)
>Hot cache: If the process is currently running and its needed data is already in cache, execution is very efficient
>Cold cache: If the process was swapped out(due to context switch), its cache data gets cleared.(when it resumes later, cache is empty. CPU has to refill it from memory)


#### Process Life Cycle: States
![](</images/Screenshot 2025-10-04 at 12.52.07 AM.png>)
1. New: When a process is first created, it is in the new state. The OS does admission control (allocates a PCB and initialize resources). The process is not yet ready to run. 
2. Ready: Once initialized, the process moves to the ready state. It is prepared to run but is waiting for the CPU to become available.
3. Running: When the scheduler picks a process from the ready queue, it goes into the **running** state. The process instructions are now being executed on the CPU. ( the active state of execution)
4. Interrupt or I/O: - a. **Interrupt / Time Slice Ends →** The process is moved back to **ready** (so another process can use the CPU). b. **I/O request or event wait →** The process moves to the **waiting** state.
5. Wait: the process is waiting for some event (e.g., I/O completion, network packet arrival). It is not using the CPU, but is still active in the system.
6. Terminated: When a process finishes execution (normally or due to an error), it moves into the **terminated** state. The OS cleans up its resources and removes its PCB.

A process can create child processes. All processes form a tree-like hierarchy
When the OS boots, it creates **root processes** with special (privileged) access. These root processes then create other processes, forming the process hierarchy.

***Two Main Mechanisms of Process Creation***:
**fork**
- The OS makes a new **PCB** for the child.
- It copies all the values from the parent’s PCB (program counter, registers, stack, etc.).
- As a result, the child process is an almost identical copy of the parent.
- Both parent and child resume execution **right after the fork call**, with the same program counter (but they can tell which is which because fork() returns different values: 0 for child, PID for parent).
***exec()***
- Replaces the process’s memory with a **new program**.
- The child still has its PCB (from fork), but its contents are overwritten with a new program’s values.
- Now the program counter points to the first instruction of the new program.
- Used when you want to run a different program (e.g., ls, bash, etc.) instead of just copying the parent.

### Role of the CPU scheduler
The CPU scheduler manages how processes use the CPU resources. It determines which process will use the CPU next, and how long that process has to run.

OS must:
preempt (interrupt and save current context)
schedule (run scheduler to choose next process)
dispatch (dispatch process and switch into its context)

The CPU’s efficiency is defined as:

$$\text{Efficiency} = \frac{\text{Time spent executing processes}}{\text{Total computation time}}$$

- **Numerator (useful work):** How long the CPU actually spends running processes.
- **Denominator (total time):** Includes both **execution time** and **scheduling overhead** (time spent switching between processes).
>Every time the OS does a **context switch** (saving PCB, restoring another), it uses CPU cycles that don’t directly help execute a process.
- This time is called **scheduling overhead**.
- If context switches happen too often, a lot of CPU time is wasted.

 **Timeslice**
The **timeslice** is how long a process gets to run before the scheduler interrupts it.
If the timeslice is **too short**:
    Many context switches → high overhead → poor efficiency.
If the timeslice is **too long**:
    Efficiency is high, but responsiveness is poor (other processes wait too long).

> Design: What are appropriate timeslice value/what criteria is used to decide which process is the next to run.
![](</images/Screenshot 2025-10-04 at 1.51.38 AM.png>)
For ***I/O*** request, send back to ready queue

### Inter Process Communication
Processes interact is called ***Inter process mechanisms(IPCs)***
**Message Passing IPC** via a shared buffer: 
 - Process A sends data → copied to kernel buffer → copied to Process B’s buffer.
Overhead( Data is copied **twice** (sender → kernel, kernel → receiver))
OS acts a postman

***Shared Memory IPC***:
The OS sets up a **shared region of memory**. Process A and Process B both map this region into their own address space. They can then read/write directly into this shared memory (like a common workspace).
Overhead: No
Disadvantage: reimplement code

### Process vs. Thread
A single-threaded process is represented by two components:
- Address space(virtual <-> physical memory mapping)
- execution context CPU registers stack
All of this information is represented by the OS in a process control block

A thread is a smaller unit of execution within a process
- Multiple threads run inside the same address space
- They share code, data, files/resources
But each thread has its own execution context: Its own registers, program counter, stack. Each thread has its own data structure to represent information specific to its execution context.

#### Multithreading
***benefits***
1. While each thread is executing the same code, each thread maybe executing a different instruction(in the sense of different line or function) -->parallel(speed up the program's execution)
2. Specialization:(we can give higher priority to tasks that handle more important tasks or service higher paying customers ) --> each thread may only need a small amount of data/code --> the entire state might fit into the cache(fast)
>memory requirements for a multiprocess application are greater than those of a multithreaded application.

Even if in a single CPU: 
- Context switching between **processes** is expensive because:
    - The OS must reload **virtual-to-physical address mappings**.
- Context switching between **threads in the same process** is cheaper because:
	- They share the **same address space and mappings**.
	$t.ctx.switch_{thread} <t.ctx.switch_{process}$
Even though only **one thread runs at a time**, threads let the CPU stay busy by switching when one is waiting.

**Synchronization Mechanisms:**
Since threads share the same virtual to physical address mappings, and they share the same address space. We could see ***Data race*** problem: One thread can try to read the data while another modifies it. It can lead to inconsistencies.
**Mutual exclusion**(through mutex): Only one thread at a time is granted access to some data. The remaining threads must wait their turn.
**Condition variable**: a thread can wait on another thread, and to be able to exactly specify what condition the thread is waiting on.

***Thread Creation***
We need some data structure to represent a thread.
- Thread ID
- Program counter
- Stack pointer
- Register values
- Stack
- Other attributes(priority attributes)
Creating a thread is conceptually like a **fork** (not the UNIX fork(), but similar idea).
- You specify:
    - The **procedure (proc)** the thread should run.
    - The **arguments (args)** to pass to it.
The new thread is created with its **own program counter pointing to the start of proc**, and its own registers + stack.
Now the process has multiple threads, all sharing the same address space, but each thread runs independently.
Once a child thread finishes, we need a way for it to return results or signal the parent.
We also need to ensure the parent doesn’t exit before its child threads (otherwise the process would die and kill the threads).
The parent thread can call **join** on a child thread.
- This means:
    - The parent is **blocked** until the child finishes.
    - Once done, the child’s result is returned.
    - The child thread then exits, and its resources are deallocated.
This ensures proper synchronization — parent waits for child before moving on if necessary.

```
Thread thread1;
Shared_list list;
thread1 = fork(safe_insert, 4);
safe_insert(6);
join(thread1);
```
The issue is that -  The order of parent thread and child is unpredictable unless we synchronize (with join or locks).

***Mutexes***: 
To support mutual exclusion, OS support a construct called a mutex(lock). When a thread locks a mutex, it has exclusive access to the shared resource. Other thread attempting to lock the same mutex will not be successful. These threads will be blocked on the lock operation, meaning they will not be able to proceed until the mutex owner releases the mutex.

As a data structure a mutex should have:
- lock status
- owner
- blocked threads
The portion of the code protected by the mutex is called the ***critical section***. The critical section should contain any code that would necessitate restricting access to one thread at a time( executed by one thread at a given moment in time).

Unlock:
- the end of a clause following a lock statement is reached
- an unlock function is explicitly called
***Condition Variables***
A condition variable lets threads **wait until a certain condition is true**
It works with a mutex:
    A thread can **wait(mutex, condition_variable)** → it pauses until another thread signals the condition.
    Another thread can **signal(condition_variable)** → wake up a waiting thread when the condition is met.

The data structure should contain:
- mutex reference
- list of waiting threads

```
// Implementtation of wait
Wait(mutex, cond){
// atomatically release mutex and place thread on wait queue
	release(mutex);
	add_to(cond.wait_queue, this_thread);
	sleep(this_thread);
	
	// later when signaled
	remove_from(cond.wait_queue, this_thread);
	acquire(mutex);
	// thread resumes execution
	}
```

***Readers/Writers problem***
The problem is: We have a shared state; multiple thread may want to read(does not modify) or write(modify).
The rule is: Many readers can read simultaneously; Only one writer can write at a time; Reader and writers cannot access at the same time.
> A naive approach would be wrapping everything in one mutex
> But this forces only one thread at a time, even if multiple readers could safely run in parallel(too restrictive)
```
// Introduce counters
read_counter = number of readers currently reading
write_counter = 0 or 1

// conditions
if read_counter == 0 and write_counter == 0: resource is free (reader or writer can proceed)

if read_counter > 0: readers may continue but writers must wait

if writer_counter == 1: no one can write or read
```
We can also merge into one variable:
```
resource_counter = 0: resource is free
resource_counter > 0: many readers are active
resource_counter = -1: a writer is active
```
This is called a ***Proxy variable***: it encodes the access state of the resource
```
// reader
lock(counter_mutex){
	while (resource_counter == -1)
		Wait(counter_mutex, read_phase); // wait until safe to read
	resource_counter++; // Add this reader
	}//unlock
	
Lock(counter_mutex){
	resource_counter--;
	if (resource_counter == 0)
		Signal(write_phase);
		}
```
```
// Entry
Lock(counter_mutex) {
    while (resource_counter != 0)           // Readers or writer active
        Wait(counter_mutex, write_phase);   // Wait until free
    resource_counter = -1;                  // Mark writer active
} // unlock

// ... write data ...

// Exit
Lock(counter_mutex) {
    resource_counter = 0;                   // Resource free again
    Broadcast(read_phase);                  // Wake up all readers
    Signal(write_phase);                    // Wake up one writer
} // unlock
```

***Critical Section***
A critical section is the part of code where a thread accesses shared state

***Spurious Wake-up***
A spurious wake-up happens when:
- A thread is waiting on a condition variable
- It gets woken up
- But it still cannot proceed because the mutex is locked or the condition isn't actually satisfied
A waste of CPU.

***Deadlocks***
Happens when two (or more) threads are waiting on each other's resources, and because of that, none can proceed.
- Each thread is "holding something" the other needs
- Since they are both waiting, they both freeze forever.
Solution: 1. Release before re-locking; 2. Lock all resource upfront; 3. Maintain a lock order(best solution)
***Kernel Vs. User-level Threads***
Kernel-level threads are the threads that the OS itself manages. It visible to the kernel and the kernel scheduler decides which CPU they run on and when they run how many cn run at the same time.

User-level threads are managed in user space. The OS doesn't see them - it only sees the process. To actually execute, a user thread must be mapped to a kernel thread. Then the kernel scheduler puts it on a CPU.

#### Multithreading models
1. One-to-one model:
Each user-level thread = 1 kernel-level thread
When you create a thread in your program, the kernel creates a real kernel thread too.
Pros: Kernel fully understands threading ( better scheduling, true parallelism on multi-CPU systems); Each thread can be managed independently
Cons: Expensive; Limited by OS policies and different OS different rules.
2. Many-to-one:
**Mapping**: Many User threads → 1 Kernel thread. The user-level thread library decides which user thread runs, but the kernel only sees **one** thread.
**Pros**:
Portable (not tied to OS kernel support).   
Fast thread operations (no system calls needed).
**Cons**:
Kernel doesn’t even know it’s multithreaded.
If one user thread blocks (e.g., I/O), the **whole process blocks**.
No true parallelism (only 1 kernel thread exists).
 3. Many-to-Many Model:
**Mapping**: Many User threads ↔ Many Kernel threads. Kernel is aware of multithreading, but not every user thread must have a dedicated kernel thread.
- Supports:
    - **Bound threads** → fixed mapping (user thread always tied to same kernel thread).
    - **Unbound threads** → flexible mapping (user thread can run on any available kernel thread).
**Pros**:
    Best of both worlds → concurrency + flexibility.
    If one kernel thread blocks, others can continue.
    Good scalability for large apps.
 **Cons**:
 More complex (needs coordination between user-level library and kernel-level scheduler).
####  **Scope of Multithreading**

- **Process Scope (User-Level)**
    - Threads managed by the **user-level library** inside one process.
    - The kernel only sees the process, not the individual threads.
    - Example: Process A has 4 threads, Process B has 2 threads → kernel might still give A and B the _same CPU share_, so each thread in A gets **less time** than each thread in B.
- **System Scope (Kernel-Level)**
    - Threads are visible to the **kernel scheduler**.
    - Kernel allocates CPU to _all threads across the system_.
    - Example: A has 4 threads, B has 2 threads → kernel may give A twice the CPU share as B, since A has more threads.
**Difference**: With **process scope**, fairness is at the _process level_. With **system scope**, fairness is at the _thread level_.

---
#### **Boss/Workers Pattern**

**Setup**:
    1 boss thread → hands out work.
    Multiple worker threads → actually do the work.
**Variants**:
    **Direct assignment** → boss tracks idle workers and assigns tasks.
        - Simpler worker logic, but boss is overloaded.
    **Work queue** → boss just drops tasks into a queue, workers pull tasks.
        - Reduces boss overhead, but requires worker synchronization
**Scaling**:
    Use a **thread pool** to avoid creating/destroying threads each time.
    Can dynamically grow/shrink pool size depending on load.
**Pros**: Simple, easy to understand.
**Cons**: Bottleneck at boss thread; load balancing tricky.

#### **Pipeline Pattern**
**Setup**: Break work into **stages** (like an assembly line).
  Example: A 6-step process → 6 thread types, each for one stage.
**Execution**:
Tasks flow stage by stage.
At any time, different tasks are in different pipeline stages.
**Scaling**: If one stage is slow (the **bottleneck**), add more threads to that stage.
**Pros**: Specialization, good cache locality, natural concurrency.
**Cons**: Hard to keep balanced; bursts of input can overwhelm a stage.
#### **Layered Pattern**
**Setup**: Group related subtasks into **layers**.
Each layer has multiple threads that can perform _any subtask_ in that layer.
Work flows through all layers in order.
**Example**:
    Layer 1: Input handling
    Layer 2: Processing
    Layer 3: Output/storage
**Pros**: Some specialization without being as rigid as a pipeline.
**Cons**: Synchronization between layers can be complex.

#### Pthread
