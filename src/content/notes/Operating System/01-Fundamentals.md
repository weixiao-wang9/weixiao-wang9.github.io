---
type: source
course: "[[Operating System]]"
---

# Operating System Fundamentals

## What is an Operating System?

An operating system is a layer of systems software that:
* directly has privileged access to the underlying hardware
* manages hardware on behalf of one or more applications according to predefined policies
* ensures that applications are isolated and protected from one another

## Computing System Components

The computing system consists of:
1. Central Processing Unit (CPU)
2. Physical Memory
3. Network interfaces (Ethernet/WiFi Card)
4. GPU
5. Storage Device (Disk, Flash drives)

In addition, a computing system can have higher level applications (programs).

## Functions of an Operating System

1. **Hide hardware complexity**: Provides a higher level abstraction
2. **Manage underlying hardware resources**: Allocates memory for applications, schedules them for execution on the CPU, controls access to various network devices
3. **Provides isolation and protection**: When applications are running concurrently, the OS ensures they can do what they need to without hurting one another

## Examples of Operating Systems

**Desktop operating systems:**
* Microsoft Windows
* Unix-based systems: OS X, Linux

**Embedded operating systems:**
* Android
* iOS

---

## OS Elements

An operating system provides a number of high level abstractions, as well as mechanisms to operate on these abstractions.

### Abstractions
* **Application abstractions**: process, thread
* **Hardware abstractions**: file, socket, memory page

### Corresponding Mechanisms
* create/schedule
* open/write/allocate

Operating systems may also integrate specific **policies** that determine exactly how the mechanisms will be used to manage the underlying hardware.

> Example: A policy could determine the maximum number of sockets that a process has access to

### Memory Management Example

The main abstraction is **memory page**, which corresponds to some addressable region of memory of some fixed size.

The operating system integrates mechanisms to operate on the page:
* `allocate`: allocates the memory in DRAM
* `map`: maps that page into the address space of the process, so the process can interact with the underlying DRAM

The page may be moved to different spaces of memory later using a policy (e.g., LRU).

---

## OS Design Principles

1. **Separation of mechanism and policy**: Incorporate flexible mechanisms that can support a number of policies
2. **Optimize for the common case**:
   - Where will the OS be used?
   - What will the user want to execute on that machine?
   - What are the workload requirements?

---

## OS Protection Boundary

Computer systems distinguish between at least two modes of execution:
* **user-level** (unprivileged)
* **kernel-level** (privileged)

The OS must have direct access to hardware, so it must operate in **kernel mode**.

Hardware access can only be utilized in kernel mode from the OS directly.

### User/Kernel Transitions

Applications usually operate in **user-mode**. When privileged instructions are encountered during a non-privileged execution, the application will be **trapped**. This means the application's execution will be interrupted, and control will be handed back to the OS.

The OS can:
1. Decide whether to grant the access or potentially terminate the process
2. Expose an interface of **system calls**, which the application can invoke to allow privileged access of hardware resources
3. Support **signals**, which is a way for the OS to send notifications to the application

---

## System Call Flow

### Running a Process
* You start with a **process** (a program that's currently running)
* Sometimes, that process needs to use **hardware resources** (like disk, network, keyboard, etc.)
* But processes run in **user mode**, where they can't directly access hardware — only the **OS kernel** can

### Making a System Call

1. The process asks the OS for help via a **system call**
   - This is basically: _"Hey OS, can you do this privileged thing for me?"_
2. Control **switches** from user mode → kernel mode
3. The OS executes the requested operation (maybe accessing hardware)
4. Once finished, the OS passes control (and results/data) back to the process (kernel mode → user mode)

### Cost of Context Switching

* Switching from user mode ↔ kernel mode is called a **context switch**
* This is **not free** — it takes CPU cycles to save state, switch privileges, and restore state
* System calls are not necessarily cheap operations

### Passing Arguments to System Calls

When making a system call, the process may need to pass arguments (e.g., file name, buffer, size).

This can be done in two ways:
1. **Directly**: arguments copied into registers/stack and passed into the kernel
2. **Indirectly**: pass a **pointer** to where the data is stored in memory, and the OS reads from there

### Synchronous vs. Asynchronous Mode

**Synchronous mode**:
* The process **waits** until the system call finishes
* Example: reading from a file — you can't continue until the data arrives

**Asynchronous mode**:
* The process can continue doing other work while the OS finishes the request in the background
* Example: non-blocking I/O

---

## Crossing the OS Boundary

User/Kernel transitions are common and useful throughout the course of application execution.

The hardware supports user/kernel transitions:
* The hardware will cause a trap on illegal executions that require special privilege
* It initiates transfer of control from process to operating system when a trap occurs

**Cost of transitions**:
* User/Kernel transition requires instructions to execute, which can take ~100ns on a 2GHz Linux box
* The OS may bring some data into the hardware cache, which will bounce out some memory used by another application

---

## OS Services

An operating system provides applications with access to the underlying hardware. The OS exposes certain services:

### Services Directly Linked to Hardware
* **Scheduling component (CPU)**: decides which process gets CPU time and when
* **Memory manager (physical memory)**: keeps track of what part of RAM is used/free, allocates memory safely to processes
* **Block device driver (disk/storage)**: lets applications read/write to storage without knowing the details of the hardware

### Higher Level Services
* Process management
* File management
* Device management
* Memory management
* Storage management
* Security

---

## OS Architectures

### Monolithic OS

Everything is included in one large kernel.

**Pros**:
* Everything is included/inlining
* Compile time optimization

**Cons**:
* No customization
* Not too portable
* Large memory footprint

### Modular OS

A type of operating system that has a basic set of services and APIs. Anything not included can be added as a **module**. It can dynamically install new modules in the operating system.

**Pros**:
* Maintainability
* Smaller footprint
* Less resource needs

**Cons**:
* All the modularity/indirection can reduce some opportunities for optimization
* Maintenance can still be an issue as modules from different codebases can be slung together at runtime

### Microkernel

Only requires the most basic operating system components. Everything else will run outside of the operating system at user-level.

This setup requires lots of interprocess communication, as the traditional operating system components run within application processes.

> The microkernel often supports IPC as a core abstraction

**Pros**:
* Size
* Verifiability (great for embedded devices)

**Cons**:
* Bad portability
* Harder to find common OS components due to specialized use case
* Expensive cost of frequent user/kernel crossing

---

## Linux and Mac OS Architecture

### Linux
* Hardware
* Linux Kernel
* Standard libraries
* Utility programs
* User applications

**Kernel consists of several logical components**:
* Virtual file system
* Memory management
* Process management

### Mac OS X
* I/O kit for device drivers
* Kernel extension kit for dynamic loading of kernel components
* Mach microkernel: memory management, thread scheduling, IPC
* BSD component: Unix interoperability, POSIX API support, Network I/O interface
* All applications sit above this layer
