---
type: index
course: "[[Operating System]]"
---

# Operating System Study Guide

This directory contains organized notes on Operating System concepts, following a logical learning progression.

## Learning Path

Study these notes in order for the best learning experience:

### 1. [Fundamentals](01-Fundamentals.md)
**Topics**: OS definition, architecture, protection modes, system calls, OS services
- What is an operating system?
- OS design principles
- User/kernel mode transitions
- Monolithic vs. Modular vs. Microkernel architectures
- Linux and Mac OS architecture

### 2. [Processes and Threads](02-Processes-and-Threads.md)
**Topics**: Process lifecycle, PCB, context switching, threading models
- Process vs. application
- Memory layout (code, data, heap, stack)
- Virtual addresses and address space
- Process Control Block (PCB)
- Process states and lifecycle
- Thread creation and management
- Multithreading patterns (Boss/Workers, Pipeline, Layered)

### 3. [Synchronization](03-Synchronization.md)
**Topics**: Mutexes, condition variables, deadlocks, synchronization primitives
- Data races and synchronization problems
- Mutexes and critical sections
- Condition variables
- Readers/Writers problem
- Deadlock prevention
- Spurious wake-ups
- Pthread synchronization

### 4. [CPU Scheduling](04-CPU-Scheduling.md)
**Topics**: Scheduling algorithms, fairness, real-time scheduling
- Scheduling metrics (throughput, wait time, utilization)
- FCFS, SJF, Round Robin
- Priority scheduling and priority inversion
- MLFQ (Multilevel Feedback Queue)
- Linux O(1) Scheduler
- Linux CFS (Completely Fair Scheduler)
- Multiprocessor scheduling
- Cache affinity and load balancing
- NUMA awareness

### 5. [Memory Management](05-Memory-Management.md)
**Topics**: Virtual memory, paging, segmentation, page tables
- Paging and segmentation
- MMU (Memory Management Unit)
- TLB (Translation Lookaside Buffer)
- Page tables (single-level, multi-level, inverted)
- Page faults and swapping
- Memory allocators (Buddy, Slab)

### 6. [Inter-Process Communication](06-Inter-Process-Communication.md)
**Topics**: IPC mechanisms, shared memory, message passing
- Message-based IPC (pipes, message queues, sockets)
- Memory-based IPC (shared memory)
- SysV vs. POSIX shared memory
- Synchronization for shared memory
- RPC (Remote Procedure Call)

### 7. [I/O and Filesystems](07-IO-and-Filesystems.md)
**Topics**: Device drivers, disk I/O, filesystem structure
- I/O devices and device drivers
- CPU-device communication (MMIO, polling, interrupts)
- PIO vs. DMA
- Synchronous vs. asynchronous I/O
- Block device stack
- Virtual File System (VFS)
- Filesystem structure (ext2, inodes)
- Disk access optimizations
- Journaling

---

## Quick Reference

### Key Concepts by Topic

**Process Management**: PCB, context switching, fork, exec
**Thread Management**: pthread, mutex, condition variables
**Scheduling**: timeslice, priority, fairness, vruntime
**Memory**: virtual address, page table, TLB, page fault
**IPC**: shared memory, message passing, sockets
**I/O**: device driver, interrupt, DMA, VFS

### Common Patterns

- **Convoy Effect**: Long jobs blocking short jobs in FCFS
- **Priority Inversion**: Low-priority task blocking high-priority task
- **Deadlock**: Circular wait on resources
- **Cache Affinity**: Keep tasks on same CPU for cache benefits
- **Readers/Writers**: Multiple readers OR one writer

---

## Study Tips

1. **Start with fundamentals**: Understand OS architecture and protection modes before diving into specific mechanisms
2. **Follow dependencies**: Processes → Threads → Synchronization → Scheduling
3. **Connect concepts**: Notice how scheduling relates to context switching, how IPC relates to synchronization
4. **Think about trade-offs**: Every design decision has pros and cons
5. **Practice with examples**: Work through scheduling algorithms, page table calculations, etc.

---

*Last updated: 2026-02-01*
