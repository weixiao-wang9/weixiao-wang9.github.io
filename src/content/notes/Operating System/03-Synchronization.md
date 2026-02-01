---
type: source
course: "[[Operating System]]"
---

# Thread Synchronization

## The Problem

Since threads share the same virtual-to-physical address mappings and the same address space, we encounter **data race** problems:
* One thread can try to read data while another modifies it
* This leads to inconsistencies and unpredictable behavior

We need mechanisms to ensure threads coordinate properly when accessing shared resources.

---

## Mutexes (Mutual Exclusion)

### What is a Mutex?

To support mutual exclusion, the OS provides a construct called a **mutex** (lock).

* When a thread locks a mutex, it has **exclusive access** to the shared resource
* Other threads attempting to lock the same mutex will not be successful
* These threads will be **blocked** on the lock operation, meaning they will not be able to proceed until the mutex owner releases the mutex

### Mutex Data Structure

As a data structure, a mutex should have:
* **Lock status** (locked/unlocked)
* **Owner** (which thread holds the lock)
* **Blocked threads** (list of threads waiting for the lock)

### Critical Section

The portion of the code protected by the mutex is called the **critical section**.

The critical section should contain any code that would necessitate restricting access to one thread at a time (executed by one thread at a given moment in time).

### Unlock Operations

A mutex is unlocked when:
* The end of a clause following a lock statement is reached
* An unlock function is explicitly called

---

## Condition Variables

### What is a Condition Variable?

A **condition variable** lets threads **wait until a certain condition is true**.

It works with a mutex:
* A thread can **wait(mutex, condition_variable)** → it pauses until another thread signals the condition
* Another thread can **signal(condition_variable)** → wake up a waiting thread when the condition is met

### Data Structure

The condition variable data structure should contain:
* **Mutex reference**
* **List of waiting threads**

### Implementation of Wait

```c
// Implementation of wait
Wait(mutex, cond) {
    // Atomically release mutex and place thread on wait queue
    release(mutex);
    add_to(cond.wait_queue, this_thread);
    sleep(this_thread);

    // Later when signaled
    remove_from(cond.wait_queue, this_thread);
    acquire(mutex);
    // Thread resumes execution
}
```

---

## Readers/Writers Problem

### The Problem

We have a shared state where:
* Multiple threads may want to **read** (does not modify)
* Multiple threads may want to **write** (modify)

### The Rules

* Many readers can read simultaneously
* Only one writer can write at a time
* Readers and writers cannot access at the same time

> A naive approach would be wrapping everything in one mutex, but this forces only one thread at a time, even if multiple readers could safely run in parallel (too restrictive).

### Solution: Using Counters

Introduce counters:

```c
read_counter = number of readers currently reading
write_counter = 0 or 1
```

Conditions:
* If `read_counter == 0` and `write_counter == 0`: resource is free (reader or writer can proceed)
* If `read_counter > 0`: readers may continue but writers must wait
* If `write_counter == 1`: no one can write or read

### Optimization: Proxy Variable

We can merge into one variable:

```c
resource_counter = 0: resource is free
resource_counter > 0: many readers are active
resource_counter = -1: a writer is active
```

This is called a **proxy variable**: it encodes the access state of the resource.

### Reader Implementation

```c
// Reader Entry
Lock(counter_mutex) {
    while (resource_counter == -1)
        Wait(counter_mutex, read_phase);  // Wait until safe to read
    resource_counter++;  // Add this reader
} // unlock

// ... read data ...

// Reader Exit
Lock(counter_mutex) {
    resource_counter--;
    if (resource_counter == 0)
        Signal(write_phase);
} // unlock
```

### Writer Implementation

```c
// Writer Entry
Lock(counter_mutex) {
    while (resource_counter != 0)           // Readers or writer active
        Wait(counter_mutex, write_phase);   // Wait until free
    resource_counter = -1;                  // Mark writer active
} // unlock

// ... write data ...

// Writer Exit
Lock(counter_mutex) {
    resource_counter = 0;                   // Resource free again
    Broadcast(read_phase);                  // Wake up all readers
    Signal(write_phase);                    // Wake up one writer
} // unlock
```

---

## Common Synchronization Issues

### Critical Section

A **critical section** is the part of code where a thread accesses shared state.

It must be protected by appropriate synchronization primitives (mutexes, condition variables, etc.).

### Spurious Wake-up

A **spurious wake-up** happens when:
* A thread is waiting on a condition variable
* It gets woken up
* But it still cannot proceed because the mutex is locked or the condition isn't actually satisfied

This is a waste of CPU cycles.

**Solution**: Always use a `while` loop instead of `if` when checking conditions:

```c
while (condition_not_met) {
    Wait(mutex, cond);
}
```

---

## Deadlocks

### What is a Deadlock?

A **deadlock** happens when two (or more) threads are waiting on each other's resources, and because of that, none can proceed.

* Each thread is "holding something" the other needs
* Since they are both waiting, they both freeze forever

### Example

```
Thread 1:               Thread 2:
Lock(A)                 Lock(B)
Lock(B)  <-- waits      Lock(A)  <-- waits
...                     ...
```

Both threads are stuck waiting for each other.

### Solutions

1. **Release before re-locking**: Release all locks before attempting to acquire new ones

2. **Lock all resources upfront**: Acquire all necessary locks at the beginning (all-or-nothing approach)

3. **Maintain a lock order** (best solution):
   - Define a global ordering of locks
   - All threads must acquire locks in the same order
   - This prevents circular wait conditions

**Example of lock ordering**:
```c
// Always lock in order: A before B
Lock(A);
Lock(B);
// ... critical section ...
Unlock(B);
Unlock(A);
```

---

## Kernel vs. User-Level Threads

### Kernel-Level Threads

**Kernel-level threads** are threads that the OS itself manages:
* Visible to the kernel
* The kernel scheduler decides which CPU they run on and when they run
* The kernel knows how many can run at the same time

### User-Level Threads

**User-level threads** are managed in user space:
* The OS doesn't see them - it only sees the process
* To actually execute, a user thread must be **mapped to a kernel thread**
* Then the kernel scheduler puts it on a CPU

---

## Pthread Synchronization

Pthread is a POSIX standard for threading that provides:
* Thread creation/management
* Mutex locks
* Condition variables
* Read-write locks
* Barriers

### Basic Pthread Mutex Example

```c
pthread_mutex_t lock;
pthread_mutex_init(&lock, NULL);

pthread_mutex_lock(&lock);
// Critical section
pthread_mutex_unlock(&lock);

pthread_mutex_destroy(&lock);
```

### Basic Pthread Condition Variable Example

```c
pthread_mutex_t lock;
pthread_cond_t cond;

pthread_mutex_init(&lock, NULL);
pthread_cond_init(&cond, NULL);

// Thread 1 (waiting)
pthread_mutex_lock(&lock);
while (!condition_met) {
    pthread_cond_wait(&cond, &lock);
}
pthread_mutex_unlock(&lock);

// Thread 2 (signaling)
pthread_mutex_lock(&lock);
// ... modify shared state ...
condition_met = 1;
pthread_cond_signal(&cond);
pthread_mutex_unlock(&lock);
```

---

## Summary

* **Mutexes** provide mutual exclusion for critical sections
* **Condition variables** allow threads to wait for specific conditions
* The **readers/writers problem** demonstrates how to allow concurrent reads while protecting writes
* **Deadlocks** occur when threads wait circularly for each other's resources
* **Lock ordering** is the best solution to prevent deadlocks
* **Spurious wake-ups** require using `while` loops with condition variables
