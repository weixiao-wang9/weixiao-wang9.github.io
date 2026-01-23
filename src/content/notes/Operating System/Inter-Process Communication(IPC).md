A ***process*** has its own virtual address space. 
>Can't see each other's memory
>Can't directly call each other's functions
>Only interact via the OS

But in real system, processes need to:
- Send data (communication)
- Coordinate who does what and when (coordination)
- Avoid race conditions (synchronization)

IPC = all the OS mechanisms that make that possible.

1. Message-based IPC
	- Data is sent as messages via some OS-managed channel
	- pipes, message queues, sockets
2. Memory-based IPC
	- Processes share part of memory
	- shared memory segments, memory-mapped files
Plus:
Higher- level: Remote Procedure Call (RPC)
Synchronization primitives: mutexes, semaphores, condition variables, etc.


### Message-based IPC

Processes do:
`send/write` a message to a port or handle
`recv/read` a message from a port

The OS:
Creates and maintains the channel
Manages buffers, queues, scheduling, synchronization

Cost: user/kernel crossings + copies

Each send/receive usually does:
1. User -> kernel (system call)
2. Copy data from process memory -> kernel buffer
3. Kernel maybe moves data around internally
4. Kernel -> user (on receive)
5. Copy data from kernel buffer -> process memory

For a request-response (A->B then B->A):
4 system calls total (send, recv, send, recv)
4 copies total (two each way)
>Pros:
>Simple to use: OS hides details
>Synchronization largely handled by the OS
>Works between unrelated processes
>Often works across machines
>Cons:
>Repeated system call overhead
>Repeated data copying
>Can be slow for large data

***Pipes***: is a unidirectional data channel: writer -> pipe -> reader.

Two endpoints (file descriptors)
- One end: write
- One end: read
Data is a byte stream, no message boundaries
`cat` writes bytes to the pipe
`grep` reads bytes from the pipe

Message Queues:
Channel understands messages, not just bytes:
- Sender sends a message(struct/buffer + length)
- Receiver receives one whole message at a time
The OS:
- Can support message priorities
- Can choose which message to deliver next (scheduling)
>API in Unix:
>	`SysV` Message queues
>	`POSIX` Message queues
>These often provide: Blocking/non-blocking send/recv; Priority-based ordering; Flags for different behaviors

Sockets:
Sockets generalize IPC to local + network communication.
A socket is an endpoint: think "file descriptor + protocol"
You get a socket with socket(...), which:
	Creates a kernel buffer for that socket
	Associates a protocol stack
Local vs Remote:
* Same machine:
		Can be UNIX domain sockets (faster, no network stack)
* Different machines:
		Use IP addresses, ports, network hardware
Sockets are the most flexible and widely-used IPC method(especially when crossing machines)

### Shared Memory IPC (memory-based)
Instead of copying data through the kernel each time:
OS maps the same physical pages into the virtual address spaces of multiple processes/
So: Process A and B both have some addresses (maybe different virtual addresses) that refer to the same physical memory

After the mapping is set up:
	Each process just loads/stores to that region as if it were normal memory
	No system call needed per access
>Pros:
>Extremely fast after setup
>No user/kernel crossings per access
>Zero-copy data sharing is possible
>Very good for large data and frequent communication
>Cons:
>The OS only sets up maps; you must:
>	Handle synchronization (avoiding races)
>	Define a protocol (where to put data, when it's ready)
>	Harder to get right than message-based IPC

Physical pages do not need to be contiguous
Virtual addresses in each process can be different
The OS sets up the mapping in page tables

### Copy vs Map trade-offs
**Copy (message-based IPC)**
- Each send/recv involves copying:
    - A → Kernel → B
- CPU cycles spent for **every transfer**
- No extra memory setup cost beyond basic buffer allocations
Good when:

- Messages are small
- Communication is infrequent
- Simplicity is more important than raw speed
 **Map (shared memory)**

- One-time setup:
    - System calls to create/mmap shared region
    - OS sets up page table entries
    
- After that:
    
    - Processes just read/write directly
    - No per-message kernel transitions

Data copies might still happen:

- If process A builds data in its **private** memory, then copies into shared region
- You can reduce this by:
    - Allocating structures directly in the shared region
Good when:

- Data is **large**
- Communication is **frequent**
- Long-lived shared region
    
OS example:
Windows **Local Procedure Calls (LPC)**:
	For **small messages**, it just **copies** via a port-like mechanism
	 For **large messages**, it uses **mapping** semantics (shared memory)

### SysV shared Memory
SysV shared memory is an older Unix API based on **segments**.
**Segments as resources**

- OS creates and manages **shared memory segments**.
- Each segment:
    - Has a **key** (identifier)
    - Maps to some set of physical pages (not necessarily contiguous)
- OS enforces **global limits**:
    - Max number of segments (e.g., 4000)
    - Max total size

Segments are **persistent**:

- Created once
- Can be **attached** / **detached** by many processes over time
- Not destroyed until explicitly removed

 **Getting a key: ftok**
Different processes need to agree on a segment identifier. You don’t want to hardcode numeric IDs.
`key_t ftok(const char *pathname, int proj_id);`
- Deterministic: same (pathname, proj_id) → same key
- The OS doesn’t store these; it’s just a hashing function the app uses.
**Creating/opening a segment: shmget**
`int shmget(key_t key, size_t size, int shmflg);`
- key: from ftok or special value IPC_PRIVATE
- size: number of bytes for the segment
- shmflg: permission bits + flags (e.g. IPC_CREAT)
This returns:
A **shmid** (shared memory ID) that the kernel uses internally.

**Attaching shmat**:
`void *shmat(int shmid, const void *shmaddr, int shmflg);`
- shmid: from shmget
- shmaddr:
    - If NULL: OS picks a suitable virtual address
    - If non-null: you _request_ a specific virtual address
- Return: pointer to start of shared memory region in this process
`struct shm_data *ptr = (struct shm_data *)shmat(...);`
ptr->field is just normal memory access, but actually shared

**Detaching: shmdt**
`int shmdt(const void *shmaddr);`
- Invalidates the mappings in this process’s page table.
- Segment itself still exists in the kernel, unless explicitly removed.
**Controlling/destroying: shmctl**
`int shmctl(int shmid, int cmd, struct shmid_ds *buf);`
Used for:
    - Getting info
    - Changing permissions/parameters
    - **Destroying** segment with cmd = IPC_RMID
IPC_RMID:
- Marks the segment for removal.
- Actual removal may happen when no process is attached.

### POSIX  Shared Memory
POSIX takes a more **file-like** approach.
Files in tmpfs:
POSIX shared memory objects look like **files** but:
	  Live in a **tmpfs** (memory-backed pseudo-filesystem)
	  Represent chunks of physical memory
- OS reuses its existing file-handling infrastructure.

**Create/open object**
	`int shm_open(const char *name, int oflag, mode_t mode);`
	Returns a **file descriptor**.
**Size the object**:
`ftruncate(fd, size);`
**Map it** into your address space:
`void *addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);`
**Unmap** when done:
`munmap(addr, size);`
**Remove shared memory object**
`shm_unlink(name);`

### Synchronization for Shared Memory
Once multiple processes share memory, you get the same race condition problems as with multithreading—plus some extra complexity.
Rules:
- **Never** have multiple writers (or writer+reader) touching shared data without synchronization.
- You need something like:
    - **Mutexes** (mutual exclusion)
    - **Condition variables** or semaphores (to signal data availability)


### **Pthreads sync across processes**

Pthreads can be used across processes **if**:

1. The synchronization objects (mutexes, cond vars) live in **shared memory**.
2. They are initialized with the **PTHREAD_PROCESS_SHARED** attribute.

Steps:
1. Create a shared segment (SysV or POSIX).
2. Define a struct in that shared region, e.g.:
`typedef struct {
    pthread_mutex_t lock;
    char buffer[BUF_SIZE];
} shm_data_t;`
3. initialize attributes:
`pthread_mutexattr_t attr;
pthread_mutexattr_init(&attr);
pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
pthread_mutex_init(&shm_ptr->lock, &attr);
`
4. Now processes can pthread_mutex_lock(&shm_ptr->lock) and coordinate.

When PTHREAD_PROCESS_SHARED is not supported or is inconvenient, you can use:
- **Message queues**: implement higher-level protocols
    - Example:
        - A writes to shared memory, then sends “ready” message.
        - B receives it, reads from shared memory, sends “ok” message.
      
- **Semaphores**:
    
    - Binary semaphore (0/1) can act like a mutex or signal.
    - sem_wait blocks if value is 0, otherwise decrements to 0.
    - sem_post increments (and potentially wakes up a waiter).
### **Design considerations for shared-memory IPC**
Imagine two **multithreaded** processes communicating via shared memory. You have to design:
**How many shared segments?**
**Option A: One big segment**
- Pros:
    - Fewer OS objects to manage
    - Flexible: you can carve it up any way you want
- Cons:
    - You must implement your own **allocator** inside that region:
        - How to allocate/free memory inside it?
        - How to avoid fragmentation?
        - How to track ownership?

**Option B: Multiple segments**

- Maybe one segment per communication pair (thread A ↔ thread B, etc.)
- Pros:
    - Simpler per-pair logic
    - Natural isolation; bugs in one pair don’t corrupt others
- Cons:
    - Possibly many segments → overhead
    - Need a way to manage and distribute segment IDs (or names)
    - Better to **pre-allocate** a pool of segments so you don’t pay creation cost mid-execution

Often a good hybrid is:
- Pre-allocate a fixed number of shared regions
- Maintain a queue/pool of “free” region IDs
- Threads check out a region when they need it, return it when done

---

 **How big should segments be?**

Question: Is the size of data known and bounded?
- If **size is known and small**:
    - You can allocate fixed-size segments (e.g., one segment per message, or per channel).
    - But there’s usually a **max segment size** in the OS, so this only works for moderate sizes.
        
- If **message sizes vary or can be large**:
    
    - Use a fixed-sized shared buffer and transfer in **rounds/chunks**.
    - Example protocol:
        1. Sender writes a header into shared memory: total message size, etc.
        2. Sender writes CHUNK_SIZE bytes at a time, signaling receiver as it fills.
        3. Receiver reads each chunk and stores it in its own large buffer.
        4. When done, sender signals “complete”.
    
This requires:
- Agreement on:
    - Chunk size
    - Where in shared memory the chunk goes
    - How to track progress (e.g., offsets, counters, flags)
- Synchronization:
    - Mutexes/semaphores/condition variables to avoid races

---
### RPC
RPC is higher level than raw IPC
Instead of “send message” / “recv message”, you say:
- “Call function foo(args...) in another process.”

RPC describes:
- **Data formats** (e.g., XDR, protobuf)
- **Exchange protocol** (how requests and responses are structured)
- Often handles:
    - Serialization
    - Network errors
    - Timeouts
Under the hood, it still uses IPC primitives (sockets, shared memory, etc.), but gives you a function-call abstraction.