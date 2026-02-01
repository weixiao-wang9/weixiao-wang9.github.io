---
type: source
course: "[[Operating System]]"
---
### Why I/O stuff matters
A real program doesn’t just talk to the CPU and RAM. It needs:
- storage (disks / SSDs),
- input (keyboard, mouse, mic),
- output (screen, speakers, network).
The OS and hardware together have to answer:
1. **How does the CPU talk to a device?**
2. **How does a user program talk to a device safely and portably?**
3. **How can this be fast, even though many devices are slow?**
4. **How do we keep data consistent and recover from crashes?**

### I/O devices: 
Most I/O devices have a similar _structure_, even if they do wildly different things.
**Registers**
A device exposes **registers** that the CPU can read/write:
- **Command registers** – “Do this thing.”
    - e.g., “Start transmitting this packet,” “Begin reading from disk block 123.”
- **Data registers** – “Here’s the data” / “Give me the data.”
    - Used when the CPU itself moves bytes to/from the device.
- **Status registers** – “What’s going on?”
    - e.g., bits like: _busy, error, ready, interrupt pending_, etc.
From the CPU’s point of view: these registers look like a few special memory addresses or special I/O ports. Writing to those addresses is how you control the device.
**Inside the device**
Inside, a device has its own “mini computer”:
- **Microcontroller** (device’s CPU).
- **On-device memory** (buffers, firmware, queues).
- Possibly extra logic:
    - ADC/DAC for audio.
    - PHY chips for network (copper/optical).
    - Flash controllers, etc.
So you’re not manually toggling pins. You send commands to this little embedded system, and it does the heavy work.
### CPU–Device interconnect: how devices are plugged in
Devices are connected to the CPU complex via **controllers** and **buses**.
- **PCI/PCI-X**: older bus standards.
- **PCI Express (PCIe)**: newer, higher bandwidth, lower latency.
- Systems often keep older-style buses or bridges for compatibility.
Other buses
- **SCSI bus** for disks,
- **Peripheral/expansion bus** for low-speed devices (keyboards, etc.)(most common Peripheral Component Interconnect or PCI Express)
**Controllers** decide which bus a device uses (e.g., a SCSI controller for SCSI disks).
**Bridges** connect different bus types together (e.g., PCI to PCIe, or PCI to SCSI)

### Device drivers: the OS’s side of the handshake
A **device driver** is just kernel code(software components ) that knows:
- how to speak that device’s protocol,
- how to configure it,
- how to handle errors and interrupts,
- how to translate OS-level requests (“read this file”) into device-level actions.

They are responsible for **device access, management and control**
Device drivers are often provided by the device manufacturer per OS version, and each OS standardizes interfaces for **device independence** and **device diversity**

Drivers are **device-specific**, but OSes provide a **standard driver framework**:
### Types of devices
OSes classify devices mostly by _how they behave_:
1. **Block devices**
    - Think: disks, SSDs.
    - Data is read/written in **fixed-size blocks** (e.g., 512B, 4KB).
    - You can directly access block _i_ without reading 0…i-1 first (random access).
2. **Character devices**
    - Think: keyboard, serial ports.
    - Provide a **byte/character stream**.
    - You just read/write bytes in order.
3. **Network devices**
    - A bit in-between: you get variable-sized **packets/frames** (chunks of data)
OS defines standard operations per type:
- Block devices: read/write block N.
- Character devices: get/put character.
- Network devices: send/receive packet.

**Pseudo Devices**:
These files do **not** correspond to physical hardware. They are implemented entirely by the operating system kernel to provide a special software function or service.
`/dev/null`, `/dev/zero`, `/dev/random`, `/dev/tty`

**Devices as files: /dev**
UNIX-like systems represent devices as **special files** under /dev:
- /dev/sda, /dev/tty, /dev/null, etc.
- These are _not_ regular files, but the OS hooks file operations (read, write, ioctl) to device drivers.
- Filesystems like devfs and tmpfs manage these device nodes.
This “everything is a file” approach gives user programs a uniform API: open/read/write/close.

### How the CPU talks to devices
1. **Memory-mapped I/O**
Device registers are **mapped into physical address space**.
When the CPU does store or load to those addresses, the bus logic routes that to the **device**, not normal RAM.
Which addresses are used is configured via **Base Address Registers (BARs)** (part of PCI configuration).
2. **I/O port model**
x86 has special **in/out instructions** (in, out) and a separate **I/O port space**.
CPU says: out port-number, value — which sends data to that device port.
Less common on non-x86 architectures nowadays; MMIO is more universal.

**Polling vs. interrupts**
Once a device does something (e.g., disk read is done, packet arrived), the CPU needs to know.
**Polling**
- CPU periodically checks device **status registers** (“Are you done yet?”).
- Pros:
    - OS chooses when to poll (can align with times of low cache disruption).
- Cons:
    - Can introduce **latency** (you only notice completion at next poll).
    - Excess polling wastes CPU time if nothing’s happening.
**Interrupts**
- Device **raises an interrupt line**.
- CPU pauses what it’s doing, runs an **interrupt handler**.
- Pros:
    - Near-immediate response; no wasted polling.
- Cons:
    - Each interrupt has overhead:
        - Mode switch, saving registers, cache pollution,
        - Possible changes to interrupt masks, etc.
Often systems use **hybrid schemes**

Once command and addresses are set, actual data must move between device and memory.

**Programmed I/O (PIO)**
With just basic PCI support a system can request an operation from a device using a method called programmed I/O (PIO). This requires no additional hardware support. The CPU issues instructions by writing to the device's command registers and transfers data via the data registers
$$\text{num\_instructions} = \text{Instructions for Command} + \text{Instructions for Data}$$
 Example: sending a 1500B network packet using 8-byte data registers:
    - Each access transfers 8 bytes.
    - 1500 / 8 = 187.5 → **round up** to 188 transfers.
    - Plus 1 write to the command register.
    - = **189 CPU accesses** total.
- For each 8-byte chunk, CPU does something like:
`*DATA_REG = *(uint64_t *)(buffer + offset);`
Simple, no extra hardware beyond the bus and device.
Downside: **CPU is busy the whole time**, doing loads/stores instead of useful computation.

Workflow: Command $\rightarrow$ Wait/Poll $\rightarrow$ Data Write $\rightarrow$ Repeat.

**Direct Memory Access (DMA)**
We still write commands to the device via command registers, but data movement is controlled by configuring the DMA controller.
Flow:
1. CPU writes **command register**: “Transmit a packet.”
2. CPU configures DMA with:
    - physical memory address of buffer,
    - size of buffer, direction, etc.
3. DMA engine takes over:
    - Reads packet from RAM,
    - Transfers to device or vice versa.
4. Device signals completion (typically via interrupt).
Advantages:
- CPU does **O(1)** work for big transfers: **issue command + configure DMA**
- Big transfers become much more efficient.
Caveats:
- **DMA setup is expensive** in cycles (longer than one simple memory access).
    - For **small transfers**, PIO can be faster.
- DMA operates on **physical memory**, so:
    - buffers must stay resident until DMA completes → **pinned memory** (non-swappable).
Pinning is important: you can’t swap out a page from under an ongoing DMA transfer.

### OS Bypass: user-level access to devices
![](</images/Screenshot 2025-12-04 at 6.12.11 PM.png>)**User process → system call → kernel → driver → device**(normal process)
**OS bypass** 
lets a user process talk **directly** (almost) to the device:
- Certain device registers and buffers are **mapped into the process’s address space**.
- Kernel still sets things up initially (permissions, mappings), but then gets out of the way.
- A **user-level driver library** (provided by the vendor) handles the hardware protocol from user space.
Benefits:
- No **syscall overhead** for each operation.
- Can greatly reduce latency for high-performance networking / storage.
Constraints:
- Device must have enough register space so some can be mapped to processes while others stay reserved for the OS.
- Device must handle **demultiplexing**:
    - When data arrives, device must figure out which process’s buffers it belongs to.
    - Normally, the kernel stacks do this demux.
    
### **Synchronous vs asynchronous I/O**
Suppose your thread issues an I/O request (read from disk, send packet, etc).
**Synchronous I/O**
- The calling thread **blocks** until the operation completes.
- Kernel:
    - Puts the thread on the device’s **wait queue**.
    - Schedules other threads meanwhile.
    - When device finishes and signals, kernel wakes up the waiting thread.
Programming model is simple: call read(), and it returns “when done.”
**Asynchronous I/O**
- The thread issues the request and **does not block**.
- It can:
    - Do more work,
    - Later check if the I/O is complete, or
    - Be notified via a signal/callback/completion queue.
CPU can compute while I/O is in progress.
### Block Device Stack: from files to blocks
Applications care about **files**, not sectors.
Layers:
1. **User process**
    - Uses a **file abstraction** (open, read, write).
2. Kernel **Filesystem (FS)**
    - Interprets file paths, permissions,
    - Maps file offsets to **block numbers** on some block device.
3. **Generic block layer**
    - OS-provided uniform interface for all block devices on that OS.
    - Hides differences between, say, SATA disk vs NVMe SSD vs RAM disk.
4. **Device driver**
    - Talks the specific protocol of that device type and model.
5. **Block device (disk/SSD)**
### Virtual File System (VFS): multiple filesystems as one tree
![](</images/Screenshot 2025-12-04 at 7.20.39 PM.png>)
Problems VFS solves:
- Single unified tree (/) even if data lives:
    - on different disks,
    - on different filesystem types,
    - or even on remote machines (NFS, etc.).
- Allow different FS implementations optimized for different devices.
The **VFS layer**:
- Exposes the same API to user processes (POSIX).
- Expects each concrete filesystem to implement a specific set of **VFS callbacks**:
    - e.g., “How to create/delete/lookup files, read/write file data, etc.”
**Key VFS abstractions**
1. **File**: elements on which the VFS operations
2. **File descriptor**: OS representation of file
    - Small integer returned by open().
    - Process-local handle to a file _instance_ (with offset, flags, etc).
3. **inode (index node)**
    - Persistent structure describing **one file**:
        - Owner, permissions
        - File size
        - Timestamps
        - **Pointers to the blocks** holding the file’s data.
    - Important because file data blocks can be **scattered** across the disk.
4. **dentry (directory entry)**
    - Short-lived in-memory object representing one **path component**.
    - Example: accessing /users/ada → dentries for /, users, ada.
    - Cached in a **dentry cache** so subsequent path lookups don’t re-read directories from disk.
    - **Not persisted** to disk; purely in-memory lookup accel.
5. **superblock**
    - Describes overall layout of a **filesystem instance**:
        - Where inodes live,
        - Where data blocks live,
        - Which blocks are free/used.
    - Each mounted filesystem has one superblock (in memory) representing its state.
Mapping:
- **On disk**: superblock, inodes, data blocks, allocation bitmaps.
- **In memory**: VFS structures (superblock, inodes, dentries, file objects).
### EXT2 on-disk layout
A disk partition formatted as ext2:
![](</images/Screenshot 2025-12-04 at 7.59.49 PM.png>)
1. **Boot block** (first block)
    - Often used for bootloader.
    - Linux ext2 itself doesn’t use it directly for files.   
2. **Block groups**
    - The rest of the partition is split into multiple **block groups**.
    - Group size is a logical design choice, not necessarily tied to physical disk geometry.
Each **block group** contains:
- **Superblock copy** (or group descriptor structures).
- **Group descriptor**
    - Tracks:
        - pointers to bitmaps,
        - counts of free inodes/blocks,
        - number of directories, etc.
- **Bitmaps**
    - **Block bitmap**: which blocks are free/used.
    - **Inode bitmap**: which inodes are free/used.
- **Inode table**
    - An array of inode structures (e.g., 128 bytes each in ext2).
- **Data blocks**
    - The actual contents of files and directories.
        
Bitmaps let the filesystem quickly find free space and free inodes.

---

### Inodes and indirect pointers
Each file is uniquely identified by an inode, which contains a list of all blocks of data and its metadata

**Basic inode with direct pointers**
Imagine an inode that is 128 bytes and stores **4-byte block pointers**.
- Number of pointer slots = 128 / 4 = 32.
If each data block is 1 KB:
- Each direct pointer → 1 KB.
- Max data per file with only direct pointers:
    - 32 pointers × 1 KB per block = 32 KB.

- **Inode Size:** 128 bytes (The total space reserved in the inode for all pointers).
- **Pointer Size:** 4 bytes (The size of a disk block address/pointer).
- **Calculation:**
    $$ \text{Number of pointer slots} = \frac{\text{Inode Size}}{\text{Pointer Size}} = \frac{128 \text{ bytes}}{4 \text{ bytes/pointer}} = \mathbf{32 \text{ pointers}}$$

So a file larger than 32 KB wouldn’t fit. This is too small in practice.
**Fix: indirect pointers**
Inodes use a mix of:
- some **direct pointers** (fast for small files),
- plus **indirect pointers** for big files
Types:
1. **Single indirect pointer**
    - It doesn’t point to data directly.
    - It points to a **block of pointers**.
    - That block (1 KB) is filled with 4-byte pointers:
        - 1 KB / 4 B = 1024 / 4 = 256 pointers.
    - Each of those points to a data block.
    So one single-indirect pointer can reference:
    - 256 data blocks × 1 KB per block = 256 KB of data.

2. **Double indirect pointer**
    - Pointer in inode → block of _single-indirect pointers_.
    - Each of those → block of data pointers.
    - Calculation:
        - First level: 1 KB / 4 B = 256 pointers.
        - Each of those points to another 1 KB block of 256 pointers.
        - Total data blocks = 256 × 256 = 65,536 blocks.
        Each data block = 1 KB:
        - Total bytes = 65,536 × 1 KB.
        - 1 KB = 1024 bytes.
        - 65,536 × 1024 bytes = 67,108,864 bytes.
        - Divide by (1024 × 1024) to get MB:
            - 67,108,864 / 1,048,576 = 64.
        → **64 MB** of data referenced by one double-indirect pointer.

You can also have **triple indirect** pointers for truly huge files (same pattern: another level of indirection).
$$\text{max\_file\_size} = \left(\text{direct} + \text{single\_indirect} + \text{double\_indirect} + \text{triple\_indirect}\right) \times \text{block\_size}$$
Trade-offs:
- Pros:
    - Small inode structure but supports large files.
    - Small files only need direct pointers → fewer disk accesses.
- Cons:
    - Accessing data via double/triple indirect pointers requires more disk reads:
        - Inode → indirect block(s) → data block.
    - Worst case: more seeks and latency.
### Disk access optimizations
Disks (especially spinning ones) are slow compared to RAM/CPU, mainly due to:
- **Seek time** (moving head),
- **Rotational latency**.
So OS and filesystem add many optimizations:
**Buffer cache (page cache)**
- Keep recently used file data blocks in **main memory**.
- Reads:
    - If data is in cache → no disk I/O (cache hit).
    - Otherwise read from disk and cache it.
- Writes:
    - Often **write-back**: modify in-memory buffer and mark it dirty.
    - Later, flush dirty blocks to disk.

`fsync()` exists so applications can force dirty data to be committed to disk (important for databases, etc.).
### I/O scheduling
Multiple pending requests to disk can be **reordered** to reduce head movement.
Example:
- Current head at block 7.
- Requests arrive: write block 25, then block 17.
- Instead of:
    - 7 → 25 → 17 (backtrack),
- Scheduler does:
    - 7 → 17 → 25.
This is the idea behind algorithms like **elevator/SCAN** schedulers.
### **Prefetching (read-ahead)**
Assumption: programs often access data with **spatial locality** (e.g., reading sequentially).
Example:
- You request block 17.
- The filesystem also pulls blocks 18 and 19 into cache.
- If you then request 18, it’s already in memory → low latency.
Trade-off:
- Uses extra disk bandwidth and memory, but can significantly increase hit rate and reduce average latency.
### Journaling
Problem: write-back caching + reordering means:
- Data and metadata updates are sitting in memory.
- If the system **crashes** (power loss, kernel panic), the disk might end up in an **inconsistent state**:
    - Allocation bitmaps and inodes and data blocks might not match.

**Journaling** (write-ahead logging) fixes this by:
1. Writing **intended changes** to a **sequential log (journal)** on disk:
    - “I plan to change inode X, block Y, etc.”
    - Appends are **sequential**, so fast even on spinning disks.

2. Marking the journal transaction as **committed**.
3. Later, applying those recorded changes to their final locations on disk。
On crash:
- On mount, filesystem replays **committed but not-yet-applied** transactions from the journal, bringing disk back to a consistent state.
This doesn’t guarantee you keep the _very latest_ user data (depends on journaling mode), but it guarantees **filesystem consistency** (no half-updated metadata)