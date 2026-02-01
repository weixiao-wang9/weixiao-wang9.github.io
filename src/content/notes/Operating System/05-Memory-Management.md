---
type: source
course: "[[Operating System]]"
---


The OS has to manage physical memory(DRAM) for many processes at once.

Two main goals:
1. Give each process its own view of memory(virtual memory)
2. Safely and efficiently translate virtual to physical

### Paging
* Virtual address space is split into fixed-size pages.
* Physical memory is split into page frames of the same size
* OS maintains a **page table** to map:
		Virtual Page Number(VPN) to Physical Frame Number(PFN)
* Within a page, the offset is the same in virtual and physical addresses.

### Segmentation
* Memory is split into **variable-sized segments**
* Each segment represents a logical region: code, heap, stack, etc.
* Address = (segment selector, offset)
	* Hardware(specifically the MMU) uses selector and goes to a special table in memory(Descriptor Table). And it looks for the row matching that ID(Segment Descriptor)
	* Segment Descriptor contains: 1. *base*: The physical address where this segment starts in RAM. 2. *Limit*: The size of the segment
	* Adds offset to base
	$$\text{Physical Address}= Base + offset$$

### Hardware Support
#### MMU(Memory Management Unit)
* Lives inside the CPU package
* CPU issues a virtual address
* MMU translates it to a physical address using page table
>MMU raises a fault:
>1. Accessing an address that hasn't been allocated
>2. Violating permissions (protection fault)
>3. Page not present in memory (page fault)

Hardware provides registers to help with translations:
	For paging: a register points to the current page table
	For segmentation: registers hold base, size, and number of segment
Whenever we context switch between processes, OS changes this register to point to that process's page table.

#### TLB(Translation Lookaside Buffer)
Translating on every memory reference is expensive. So MMU has a small cache:
* TLB stores recent(virtual page -> physical frame) translations.
* On memory access:
	* Check TLB first.
	* TLB hit -> skip walking the page table
	* TLB miss ->walk the page table, then cache the result

### How Paging Works:VPN, PFN, Offset
Virtual Address is split into two parts:
1. VPN: Virtual Page Number (high bits)
2. Offset: Position inside that page (low bits)

Process:
1. Use VPN to index into page table -> get PFN + flags.
2. Combine PFN and offset -> full physical address.

Since all addresses inside a page share the same PFN, we only need one page table entry per page, not per byte.

### Allocation on First Touch
When you allocate: OS reserves virtual addresses but might not allocate physical frames yet.
The first time the process actually touches that page(load/store):
* MMU sees there is no mapping -> raises a page fault
* OS allocates a free physical frame
* Fills in a page table entry: VPN -> PFN
* Retry the instruction -> now succeeds
This ensures physical memory is only used for regions actually accessed.

### Page Swapping & Valid Bit
If a process hasn't used some pages for a long time:
* OS might evict those pages to disk (Swap)
* The physical frame is reused for something else
* Page table entry still exists, but:
	* Valid/present bit = 0
	* Indicates page is on disk
On access:
1. MMU sees valid bit = 0 -> raises a page fault
2. OS checks:
	* Is the access legal
	* Where is the page on disk
3. OS allocates a new physical frame.
4. Reads page from disk into that frame.
5. Updates page table entry (VPN -> new PFN, set present bit = 1)
6. Restarts the faulting instruction -> this time it succeeds
>The page usually does not come back to the same PFN it had before

### Page Table Entries (PTEs)
Each PTE contains:
* PFN (Physical Frame Number)
* At least one *valid/present bit*
* Many other bits
>**Dirty bit**: Set when the page is written and Helps decide if the page must be written back to disk (if clean, we can drop it)
>**Access(reference) bit**: Set when page is accessed (read or write) and used for page replacement (LRU-ish)
>**Protection bits**: Read/write/execute permissions
### Page Fault Handling Details
When MMU detects an invalid or illegal access:
It generates a page fault.
CPU:
* Pushes an error code on the kernel stack.
* Traps into OS kernel -> calls *page fault handler*
Error code encodes:
* Was the page present or not
* Was it a read or write
* Was it a user or kernel access

### Page Table Size
#### 32-bit architecture(Flat Page Table)
Virtual address = 32 bits -> addressable space ($2^{32}$ bytes = 4GB)
Page size = 4KB = $2^{12}$ bytes
Number of pages:$2^{32} / 2^{12} = 2^{20} \text{ pages}$
Each PTE = 4 bytes (32 bits ) including PFN + flags
Size of page table:
$2^{20} \text{ entries} \times 4\ \text{bytes} = 2^{22} \text{ bytes} = 4\ \text{MB}$ per process
Too big

#### Multi-Level Page Tables(2-level & beyond)
**Structure**
Outer level: Page Table Directory(PTD)
* Each entry points to an inner page table(or indicates "not present")
Inner level: Actual Page Tables
* Entries point to **page frames** in physical memory
* Contains PFN + flags
We only allocate inner tables for regions of virtual memory that are actually used.

#### Virtual Address Split
Virtual address bits are split
$$[ \text{outer index} | \text{inner index} | \text{offset} ]$$
Example in text:
	•	Outer index: 12 bits
	•	Inner index: 10 bits
	•	Offset: 10 bits
Interpretation:
	•	Offset = 10 bits → page size = 2^{10} = 1 KB.
	•	Inner index = 10 bits → each inner page table has 2^{10} = 1024 entries.
	•	Each entry maps 1 KB physical memory → one inner table covers:
$2^{10}\ \text{entries} \times 2^{10}\ \text{bytes} = 2^{20} \text{ bytes} = 1\ \text{MB}$

So one page table covers 1 MB of virtual space.

If there’s a gap ≥ 1 MB in virtual memory, we just don’t create the inner page tables for that range. That’s the space saving.

#### More levels
More levels saves memory

More levels = more memory accesses to walk the page tables:
	•	1-level: 1 access to PT + 1 to data = 2 memory accesses.
	•	4-level: up to 4 accesses for PT + 1 to data = 5 memory accesses.
>This is why TLB is critical: with high TLB hit rate, you rarely do a full walk.
>Without TLB
	•	1-level page table → 2 memory accesses per load/store.
	•	4-level → 5 accesses (4 levels + final data).
This would be terrible performance.
With TLB
	•	On every memory access:
	•	Check TLB.
	•	TLB hit: get physical address immediately, no page table walk.
	•	TLB miss: walk page tables → fill TLB.
Because memory accesses have strong locality (looping over arrays, stack reuse, etc.), TLB hit rates are usually high, so the average overhead is low.

### Inverted Page Tables
Normal page tables:
* Per-process
* Mapping direction: virtual -> physical
Problem: if there are many processes, total virtual memory size(sum of all processes) is huge.
Instead of "one page table per process," have one system-wide table
* Each entry corresponds to one physical frame
* Entry stores:
	* Process ID (PID)
	* Virtual page number (virtual address part)
	* Possibly flags
Logical address now includes:
PID(of process doing that access)/Virtual Address/offset

On translation:
1. Use PID + VPN to search the inverted table
	* Find the entry whose (PID, VPN) matches
	* Index of that entry in the table is the PFN
2. Combine PFN with offset -> physical address
>This is a linear search(naively). In practice: rely on TLB and hashing

#### Page Size: 4KB, 2MB, 1GB, etc
$$\text{Page Size}= 2^{\text{offset bits} }$$
>offset = 10 bits -> 1KB page
>offset = 12 bits -> 4 KB page
>Linus x86:
>Normal page: 4KB
>Large page: 2MB
>Huge page: 1 GB (offset 30 bits)

Benefits of larger pages:
1. More bits are in the offset, fewer bits in VPN
2. Fewer pages for same address space size
3. Smaller page tables(fewer PTEs)
4. Higher TLB converge(same number of TLB entries now cover larger memory)
>Large pages: page table smaller by factor of 512
>Huge pages smaller by factor of 1024

Downside: Internal fragmentation(wasted space inside the page if it doesn't fill densely)

Systems usually support multiple page sizes and use them selectively.

### Memory Allocation: Kernel vs User

**Kernel Allocators**
Allocate memory for
* Kernel data structures
* Per-process static state: code, initial stack, etc
Keep track of free physical pages
**User-level Allocators**
Manage dynamic process state = heap
*API*
	`malloc(size)` → ask kernel (via brk/mmap) for more memory, then manage internally.
	`free(ptr)` → return memory to user-level allocator’s pool (not always back to kernel).
Once kernel hands pages to the process:
* Kernel isn't involved in the fine-grained allocations
* User-level allocator works inside these pages
#### Bad Allocation Example 
-> external fragmentation
Free memory exists but not in the right shape
Allocators must place blocks smartly so that, when freed, free blocks **coalesce** into larger contiguous regions, minimizing external fragmentation.

### Linux Kernel Allocators: Buddy + Slab
#### Buddy Allocator
* Manages memory in blocks that are powers of two
* Start with a large block
* When a request comes in: 
	Recursively split into halves until block is small enough
Each split produces two *buddies*
>External fragmentation is reduced, because buddies get merged
>Internal fragmentation exists

#### Slab Allocator
Kernel objects often have sizes that are not powers of two
Buddy allocator alone is inefficient for these
