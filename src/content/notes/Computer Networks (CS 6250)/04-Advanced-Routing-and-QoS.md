---
type: source
course: "[[Computer Networks (CS 6250)]]"
created: 2026-02-01
prerequisites: "[[02-Network-Layer-and-Routing]]"
---

# Advanced Routing and Quality of Service

> **Prerequisites**: [[02-Network-Layer-and-Routing]]
> **Learning Goals**: After reading this, you will understand router architecture internals, packet classification algorithms, Head-of-Line blocking, QoS scheduling algorithms (Fair Queuing, DRR), and traffic shaping mechanisms (Token/Leaky Bucket).

## Introduction

Moving beyond basic destination-based forwarding, modern routers must handle complex tasks: classifying traffic based on multiple criteria (firewall rules, QoS policies), switching packets at line rate, and managing traffic to prevent congestion. This file covers the hardware and algorithmic challenges of high-speed routers, from lookup optimizations to sophisticated scheduling and traffic shaping.

**Key Challenges Addressed**:
- How to classify packets based on multiple header fields (not just destination IP)
- How to switch packets at multi-Gbps speeds without blocking
- How to fairly share bandwidth among competing flows
- How to shape traffic to meet service level agreements

---

## Router Architecture Internals

### The Two Planes Revisited

**Control Plane (The Brain)**:
- **Implemented in software** (Routing Processor - general-purpose CPU)
- **Function**: Runs routing protocols (BGP, OSPF) to build the routing table
- **Speed**: Slow (milliseconds to seconds)
- **Location**: Centralized processor

**Forwarding Plane (The Muscle)**:
- **Implemented in hardware** (ASICs, FPGAs on line cards)
- **Function**: Moves packets from input port to output port using forwarding table (FIB)
- **Speed**: Fast (nanoseconds - line rate)
- **Location**: Distributed across input/output ports

**Key Insight**: Separating these planes allows innovation in control logic (SDN) without changing hardware.

---

### Key Router Components

**1. Input Ports**:
- **Physical termination**: Convert optical/electrical signals to bits
- **Data link processing**: Decapsulate frame, extract packet
- **Lookup**: Match destination IP against forwarding table (FIB) - **critical bottleneck**
- **Queueing**: Buffer packets if switching fabric is busy

**2. Switching Fabric**:
- **Function**: Interconnect that moves packets from input to output
- **Types**:
  - **Memory-based**: Packets copied to/from shared memory (oldest, slowest - limited by memory bandwidth)
  - **Bus-based**: Packets traverse shared bus (limited by bus bandwidth)
  - **Crossbar-based**: N×N parallel paths (fastest, most common in modern routers)

**3. Output Ports**:
- **Queueing**: Buffer packets waiting for transmission (critical for QoS)
- **Scheduling**: Determine which packet to send next (FIFO, priority, fair queueing)
- **Data link processing**: Encapsulate packet in frame
- **Physical transmission**: Convert bits to signals

**4. Routing Processor**:
- Runs control plane software
- Builds routing table and compiles into forwarding table
- Distributes FIB to line cards

---

### Lookup Algorithms Revisited

**Challenge**: IP forwarding requires **Longest Prefix Match (LPM)** on every packet

**Why Simple Solutions Fail**:
- **Linear search**: O(n) - acceptable for 100 routes, prohibitive for 800,000+ routes
- **Caching**: High hit rates (80-90%) but misses still require full lookup; Internet traffic flows are diverse and short-lived

**Trie-Based Solutions**:

**Unibit Trie**:
- Check 1 bit at a time
- **Advantages**: Memory efficient, simple
- **Disadvantages**: Slow (32 memory accesses for IPv4, 128 for IPv6)

**Multibit Trie**:
- Check k bits at a time (stride = k)
- **Controlled Prefix Expansion**: Expand prefixes to match stride boundaries
  - Example: Prefix "1*" (length 1) expanded to "10*" and "11*" (length 2)
- **Trade-off**: More memory for fewer lookups
- **Performance**: Stride = 8-16 bits → 2-4 memory accesses per lookup
- **Result**: Line-rate forwarding at multi-Gbps speeds

**Example**:
```
Stride = 2 (check 2 bits at once)

Original Prefixes:
  1*    → Port A
  11*   → Port B
  101*  → Port C

After Expansion (to align with stride 2):
  10*   → Port A  (expanded from 1*)
  11*   → Port B  (already length 2)

Level 1: Check bits [0:1]
  00, 01, 10*, 11*

Level 2 (for 10*): Check bits [2:3]
  100*, 101*, 110*, 111*

Lookup for 10110...:
  Step 1: Check first 2 bits = 10 → match 10*
  Step 2: Check next 2 bits = 11 → match 101*
  Result: Forward to Port C
```

---

## Packet Classification

### Motivation

**Problem**: Longest prefix matching only considers **destination IP**

**Need**: Route/filter based on **multiple fields** simultaneously:
- Firewall rules: Block traffic from specific source IPs to specific destination ports
- QoS policies: Prioritize video traffic (TCP port 443 from specific sources)
- NAT/VPN: Identify flows for translation/tunneling

**Classification Criteria**:
- Source IP prefix
- Destination IP prefix
- Source port range
- Destination port range
- Protocol (TCP, UDP, ICMP)
- TCP flags (SYN, ACK, etc.)

---

### Simple Classification Approaches

**1. Linear Search**:
- Check each rule sequentially until a match
- **Complexity**: O(n) where n = number of rules
- **Acceptable**: < 100 rules
- **Prohibitive**: Thousands of rules (modern firewalls have 10,000+ rules)

**2. Caching**:
- Store recently matched flow → rule mappings
- **Hit rate**: 80-90% in practice
- **Problem**: Cache misses still require full linear search (or worse)
- **Challenge**: Internet flows are short (many new flows) and diverse

**3. MPLS (Multi-Protocol Label Switching)**:
- Assign a **label** at the network edge based on full classification
- Intermediate routers only look at the label (fast lookup)
- **Advantage**: Avoids re-classification at every router
- **Disadvantage**: Requires label distribution infrastructure

---

### 2D Classification Algorithms

**Problem**: Classify based on two dimensions (e.g., source IP and destination IP)

**Representation**: Rules as rectangles in 2D space
```
Destination IP Prefix
^
|  [Rule 2]
|     +------+
|     |      |
|  [Rule 1]  |
|  +------+  |
|  |      +--+
|  +------+
+----------------> Source IP Prefix
```

---

**1. Set-Pruning Tries**:

**Structure**:
- Build a destination trie
- At each destination prefix node, attach a **source trie** containing all source prefixes compatible with that destination

**Lookup**:
- Traverse destination trie
- At each node, check source trie for source IP
- Record best match seen so far
- Continue to more specific destination nodes

**Advantages**:
- Worst-case lookup time: O(W) where W = address width (32 bits)

**Disadvantages**:
- **Memory explosion**: Source prefixes replicated across multiple destination nodes
- Example: If 1000 destination prefixes exist, each source prefix might be replicated 1000 times

**Memory Formula**: O(N × W) where N = number of rules, W = address width

---

**2. Backtracking**:

**Structure**:
- Build a destination trie with **pointers** to source tries (instead of duplicating)
- Destination nodes point to the source trie for that destination prefix

**Lookup**:
- Traverse destination trie to find longest match
- Check pointed source trie for source IP
- If no match, **backtrack** to parent destination node and try its source trie
- Repeat until match found

**Advantages**:
- **Memory efficient**: No replication of source tries
- Memory: O(N × W)

**Disadvantages**:
- **Time cost**: Backtracking steps can be expensive
- Worst-case: O(W²) if backtracking to root multiple times

**Example**:
```
Destination Trie:
  Root
    ├─ 0* → Source Trie A
    └─ 1* → Source Trie B
         └─ 11* → Source Trie C

Lookup for (src=101*, dst=110*):
  Step 1: Traverse to 11* (longest dst match)
  Step 2: Check Source Trie C for 101* → No match
  Step 3: Backtrack to 1*
  Step 4: Check Source Trie B for 101* → Match found
```

---

**3. Grid of Tries**:

**Structure**:
- Hybrid approach combining Set-Pruning and Backtracking
- Use **switch pointers** to jump directly to the next relevant source trie (instead of backtracking)
- Precompute which source trie to check next if current fails

**Lookup**:
- Traverse destination trie
- Check source trie at current destination node
- If no match, follow **switch pointer** to next source trie (no backtracking up the tree)
- Continue until match found

**Advantages**:
- **Eliminates backtracking**: O(W) worst-case time
- **Memory reasonable**: More than backtracking, less than set-pruning

**Disadvantages**:
- More complex to build (precompute switch pointers)
- Higher memory than pure backtracking

**Example**:
```
Destination nodes with switch pointers:

  11* → Source Trie C --switch--> Source Trie B --switch--> Source Trie A
   |
  1* → Source Trie B --switch--> Source Trie A
   |
  Root → Source Trie A

Lookup for (src=101*, dst=110*):
  Step 1: Check Source Trie C → No match
  Step 2: Follow switch pointer to Source Trie B → Match found
  (No backtracking needed!)
```

---

## Switching and Scheduling

### Crossbar Switches

**Structure**: N inputs × N outputs with N² crosspoints

**How It Works**:
- Each input can connect to any output
- Multiple connections can happen simultaneously (if no conflicts)

**Example (3×3 crossbar)**:
```
Input 1 ---+---+---+
           |   |   |
Input 2 ---+---+---+
           |   |   |
Input 3 ---+---+---+
           |   |   |
         Out1 Out2 Out3
```

**Advantages**:
- **Parallel switching**: Multiple packets can be switched simultaneously
- **High throughput**: N times faster than single-path switches

**Challenge**: **Head-of-Line (HOL) Blocking**

---

### Head-of-Line (HOL) Blocking

**Problem**: In an input-queued crossbar switch, packets at the head of the queue can block packets behind them

**Example**:
```
Input Queue 1: [Packet A → Output 1] [Packet B → Output 2]
Input Queue 2: [Packet C → Output 1] [Packet D → Output 3]

Time Slot 1:
  Packet A and Packet C both want Output 1 (conflict!)
  Arbitration: Packet A wins
  Packet C waits (blocked)

  BUT: Packet D wants Output 3 (free!)
  Problem: Packet D is stuck behind Packet C
  Packet D cannot be switched even though Output 3 is available

Result: Throughput limited to ~58% of capacity (even with infinite buffers)
```

**Why 58%?**
- Mathematical analysis shows random traffic patterns with HOL blocking limits throughput to approximately 58.6% of maximum

---

### Solutions to HOL Blocking

**1. Output Queuing (Knockout Scheme)**:

**Idea**: Move all buffering to output ports instead of input ports

**How It Works**:
- Switching fabric runs **k times faster** than input link speed
- Example: If input links are 10 Gbps, fabric runs at k × 10 Gbps
- All packets destined for an output are delivered to output queue immediately

**Advantages**:
- **Eliminates HOL blocking**: 100% throughput possible
- Simple scheduling (no input coordination needed)

**Disadvantages**:
- **Expensive**: Requires very fast switching fabric (k × line rate)
- **Scalability**: Difficult for high-speed routers (100+ Gbps links)

**Typical k value**: 4-8× speedup

---

**2. Virtual Output Queues (VOQs) with Parallel Iterative Matching (PIM)**:

**Idea**: Eliminate HOL blocking by maintaining **separate queues** at each input for each output

**Structure**:
- Input port has N queues (one per output port)
- Packet destined for output j goes to queue j at input

**Example (3×3 switch)**:
```
Input 1: [Queue for Out1] [Queue for Out2] [Queue for Out3]
Input 2: [Queue for Out1] [Queue for Out2] [Queue for Out3]
Input 3: [Queue for Out1] [Queue for Out2] [Queue for Out3]
```

**No More HOL Blocking**:
- Packet to Output 1 doesn't block packet to Output 3 (different queues)

**New Problem**: How to schedule which input-output pairs to connect each time slot?

**Parallel Iterative Matching (PIM) Algorithm**:

**Goal**: Find a maximal matching (connect as many input-output pairs as possible without conflicts)

**Three Phases per Iteration**:

**1. Request**:
- Each input sends requests to all outputs for which it has queued packets
- Example: Input 1 has packets for Output 1 and 3 → sends requests to both

**2. Grant**:
- Each output randomly selects one requesting input and grants permission
- Example: Output 1 receives requests from Input 1 and 2 → randomly picks Input 1

**3. Accept**:
- Each input receives grants from multiple outputs → randomly accepts one
- Example: Input 1 receives grants from Output 1 and 3 → randomly accepts Output 1

**Iterate**: Repeat 3-4 times to find better matchings

**Example**:
```
Initial state:
  Input 1: has packets for Output 1, 3
  Input 2: has packets for Output 1, 2
  Input 3: has packets for Output 2, 3

Iteration 1:
  Request: I1→[O1,O3], I2→[O1,O2], I3→[O2,O3]
  Grant: O1→I1 (random), O2→I2 (random), O3→I3 (random)
  Accept: I1→O1, I2→O2, I3→O3

Result: Schedule transfers I1→O1, I2→O2, I3→O3 (perfect matching!)
```

**Advantages**:
- **High throughput**: Approaches 100% with iterations
- **Distributed**: No central scheduler bottleneck
- **Fast**: Simple random selection, no complex computation

**Disadvantages**:
- **Randomness**: Not guaranteed optimal (but good in practice)
- **Iterations needed**: Typically 3-4 iterations for good performance

---

## Quality of Service (QoS)

### Motivation

**Problem**: All traffic treated equally (best-effort)

**Issues**:
- Video calls dropped due to bulk file download
- Critical financial transaction delayed by email
- Real-time gaming laggy due to software updates

**Solution**: **Quality of Service** - Differentiate traffic and provide guarantees

**QoS Dimensions**:
- **Bandwidth**: Minimum/maximum rate guarantees
- **Delay**: Upper bound on latency
- **Jitter**: Variation in delay
- **Loss**: Packet drop rate

---

### FIFO with Tail Drop

**Simplest Approach**: First-In-First-Out queue with drop when full

**How It Works**:
```
Packets arrive → Queue (FIFO) → Transmit
If queue full → Drop incoming packet (tail drop)
```

**Advantages**:
- Simple to implement (O(1) per packet)
- Low overhead

**Disadvantages**:
- **No fairness**: One flow can monopolize the queue
- **No priority**: Critical traffic treated same as bulk traffic
- **Tail drop issues**: Drops affect all flows simultaneously (TCP synchronization)

**Example Problem**:
```
Flow A: Sending video (delay-sensitive)
Flow B: Sending file download (bulk, not delay-sensitive)

Flow B fills the queue → Flow A's packets dropped → Video freezes
```

---

### Fair Queuing (Bit-by-Bit Round Robin)

**Goal**: Allocate bandwidth fairly among competing flows

**Ideal Algorithm**: **Bit-by-Bit Round Robin**

**How It Works (Conceptual)**:
1. Identify all active flows (flows with queued packets)
2. In each "round", serve 1 bit from each flow
3. Repeat until all queues empty

**Example**:
```
Flow A: 3000 bits queued
Flow B: 2000 bits queued
Flow C: 1000 bits queued

Round 1: Serve 1 bit from A, B, C (3 bits total)
Round 2: Serve 1 bit from A, B, C
...
Round 1000: C finishes
Round 1001-2000: Serve 1 bit from A, B
Round 2001-3000: Serve 1 bit from A

Finish times:
  C: Round 1000
  B: Round 2000
  A: Round 3000
```

**Finish Time Formula**:
```
Finish_time(packet) = max(Current_round, Finish_time(previous_packet_in_flow)) + Packet_size
```

**Fairness Property**:
- Each flow receives equal share of bandwidth
- If n flows, each gets 1/n of link capacity

**Problem**: **Cannot send 1 bit at a time in practice** (packets are indivisible)

---

### Packet-by-Packet Fair Queuing (FQ)

**Practical Implementation**: Simulate bit-by-bit round robin at packet granularity

**How It Works**:
1. Calculate "finish time" for each packet (as if sending bit-by-bit)
2. Always transmit the packet with the **smallest finish time**
3. Update virtual time (simulated round number)

**Data Structures**:
- Per-flow queues
- Priority queue (min-heap) of packet finish times

**Algorithm**:
```
On packet arrival (flow i, size P):
  finish_time = max(virtual_time, finish_time_prev[i]) + P
  Insert packet into flow i's queue
  Insert finish_time into priority queue

On link available:
  packet = extract_min(priority_queue)
  Transmit packet
  virtual_time = packet.finish_time
```

**Example**:
```
Flow A: Packet 1 arrives (size 1000) at time 0
  finish_time = max(0, 0) + 1000 = 1000

Flow B: Packet 1 arrives (size 500) at time 0
  finish_time = max(0, 0) + 500 = 500

Priority Queue: [B:500, A:1000]

Link available:
  Transmit B's packet (smallest finish time = 500)
  virtual_time = 500

Flow B: Packet 2 arrives (size 500) at time 0
  finish_time = max(500, 500) + 500 = 1000

Priority Queue: [A:1000, B:1000]

Link available:
  Transmit A's packet (finish_time = 1000, tie broken arbitrarily)
```

**Complexity**: O(log n) per packet (priority queue operations)

**Advantages**:
- **Perfect fairness** (in the long run)
- **Isolation**: Misbehaving flow doesn't affect others

**Disadvantages**:
- **High complexity**: O(log n) per packet
- **Per-flow state**: Memory overhead for large n

---

### Deficit Round Robin (DRR)

**Goal**: Approximate fair queuing with **O(1) complexity**

**Key Idea**: Use "credits" (quantum) to allow flows to send multiple packets per round

**Data Structures**:
- **Quantum (Q)**: Fixed amount of "credit" given per round (e.g., 1500 bytes)
- **Deficit Counter (DC[i])**: Unused credit from previous round for flow i
- **Active List**: Flows with queued packets

**Algorithm**:
```
Each round:
  For each flow i in active list:
    DC[i] += Q  (add quantum)

    While (DC[i] >= size of head packet in flow i):
      Transmit packet
      DC[i] -= size of transmitted packet

    If flow i has no more packets:
      DC[i] = 0  (reset deficit)
      Remove from active list
```

**Example**:
```
Q = 1000 bytes

Round 1:
  Flow A: DC=0, Packet size=1500
    DC = 0 + 1000 = 1000 < 1500 → Cannot send
    DC[A] = 1000 (carry deficit forward)

  Flow B: DC=0, Packet size=500
    DC = 0 + 1000 = 1000 ≥ 500 → Send packet
    DC = 1000 - 500 = 500 (carry forward)

Round 2:
  Flow A: DC=1000
    DC = 1000 + 1000 = 2000 ≥ 1500 → Send packet
    DC = 2000 - 1500 = 500

  Flow B: DC=500
    DC = 500 + 1000 = 1500 ≥ 500 → Send packet
    DC = 1500 - 500 = 1000
```

**Why "Deficit"?**
- If a flow cannot send in one round (deficit), it accumulates credit for next round
- Eventually, accumulated credit allows sending large packets

**Fairness**:
- Each flow gets approximately Q bytes per round
- Long-term bandwidth share ≈ equal (like fair queuing)

**Complexity**: **O(1)** per packet (no priority queue, no sorting)

**Advantages**:
- **Constant-time**: Fast, scalable
- **Simple**: Easy to implement in hardware
- **Fair**: Good approximation of fair queuing

**Disadvantages**:
- **Not perfect fairness**: Short-term variations
- **Delay**: Large packets might wait one full round

**Typical Deployment**: High-speed routers (where O(log n) is too expensive)

---

## Traffic Shaping and Policing

### Motivation

**Problem**: Bursty traffic can overwhelm the network

**Example**:
- User's camera uploads burst of photos (10 MB in 1 second)
- Network link is 1 Mbps average
- Burst causes congestion, packet loss for other users

**Solution**: **Traffic Shaping** - Smooth out bursts to match contracted rate

**Related Concept**: **Traffic Policing** - Drop/mark packets exceeding rate (enforced by network)

---

### Leaky Bucket

**Analogy**: Bucket with a hole at the bottom

**How It Works**:
- Packets arrive at variable rate (fill the bucket)
- Packets transmitted at **constant rate** (leak out the hole)
- If bucket overflows → drop packets

**Model**:
```
         Packets arrive (variable rate)
                |
                v
         +-------------+
         |   Bucket    |  ← Buffer (size B)
         |             |
         +-------------+
                |
                v
        Constant rate R (leak)
```

**Parameters**:
- **Bucket size (B)**: Maximum burst size (buffer capacity)
- **Leak rate (R)**: Output rate (constant)

**Algorithm**:
```
On packet arrival:
  If bucket_level + packet_size <= B:
    Add packet to bucket
  Else:
    Drop packet

Continuously (every time unit):
  If bucket not empty:
    Transmit R bits
    bucket_level -= R
```

**Characteristics**:
- **Output**: Perfectly smooth (constant rate R)
- **Burst handling**: Bursts absorbed in bucket (up to size B)
- **Overflow**: Packets dropped if burst exceeds B

**Example**:
```
B = 5000 bytes, R = 1000 bytes/sec

Time 0: Burst of 3000 bytes arrives → bucket_level = 3000
Time 1: Transmit 1000 bytes → bucket_level = 2000
Time 1.5: Burst of 4000 bytes arrives → bucket_level = 6000 > B
  → Accept 3000 bytes (bucket now full at 5000), drop 1000 bytes
Time 2: Transmit 1000 bytes → bucket_level = 4000
...
```

**Use Case**: **Smoothing** traffic for constant bitrate (CBR) services (e.g., voice)

---

### Token Bucket

**Analogy**: Bucket accumulates tokens at a fixed rate; need token to send a packet

**How It Works**:
- Tokens arrive at constant rate R
- Bucket holds up to B tokens
- To transmit a packet of size P, consume P tokens
- If not enough tokens → packet waits (or dropped in policing mode)

**Model**:
```
     Tokens arrive at rate R
                |
                v
         +-------------+
         | Token Bucket|  ← Capacity B tokens
         |             |
         +-------------+
                ↑
                | Consume tokens to send packet
         Packets arrive
```

**Parameters**:
- **Bucket size (B)**: Maximum burst size (in bytes/tokens)
- **Token rate (R)**: Average rate

**Algorithm**:
```
Continuously (every time unit):
  If tokens < B:
    tokens += R × time_elapsed
    tokens = min(tokens, B)  (cap at B)

On packet arrival (size P):
  If tokens >= P:
    Transmit packet
    tokens -= P
  Else:
    Queue packet (or drop if policing)
```

**Characteristics**:
- **Allows bursts**: Can send burst of size B immediately (if tokens available)
- **Average rate**: Long-term rate limited to R
- **Flexible output**: Not constant rate (bursty output allowed)

**Example**:
```
B = 5000 bytes, R = 1000 bytes/sec

Time 0: tokens = 5000 (bucket full)
  Burst of 4000 bytes arrives → Send immediately
  tokens = 5000 - 4000 = 1000

Time 1: tokens = 1000 + 1000 = 2000
  Packet of 1500 bytes arrives → Send
  tokens = 2000 - 1500 = 500

Time 2: tokens = 500 + 1000 = 1500
  Burst of 3000 bytes arrives → Can only send 1500 bytes
  Send 1500 bytes, queue 1500 bytes
  tokens = 0
```

**Key Difference from Leaky Bucket**:
- **Leaky Bucket**: Output is constant rate (smooth)
- **Token Bucket**: Output can be bursty (up to B bytes burst) but average rate R

**Use Case**: **Rate limiting** with burst tolerance (e.g., SLAs allowing bursts up to B)

---

### Token Bucket vs Leaky Bucket Summary

| Aspect | Token Bucket | Leaky Bucket |
|--------|--------------|--------------|
| **Output Rate** | Variable (allows bursts) | Constant |
| **Burst Handling** | Allows burst up to B immediately | Smooths burst over time |
| **Average Rate** | R (long-term) | R (always) |
| **Use Case** | Rate limiting with burst tolerance | Traffic smoothing for CBR |
| **Example** | ISP SLA: 10 Mbps average, 50 MB burst | VoIP codec: constant 64 Kbps |

**Analogy Comparison**:
- **Token Bucket**: Credit card with limit B, replenishes at rate R → Can spend up to limit in burst
- **Leaky Bucket**: Water tank with hole → Water drips out at constant rate regardless of input

---

## Summary

### Key Takeaways

1. **Router Architecture**:
   - Control plane (software) computes routes, forwarding plane (hardware) moves packets
   - Multibit tries achieve line-rate lookups (2-4 memory accesses for IPv4)

2. **Packet Classification**:
   - Required for firewalls, QoS, NAT (multi-field matching)
   - Grid of Tries: O(W) time, eliminates backtracking with switch pointers
   - Trade-off: Memory vs. time complexity

3. **Head-of-Line Blocking**:
   - Limits input-queued switches to 58% throughput
   - Solutions: Output queuing (expensive) or Virtual Output Queues + PIM (practical)

4. **QoS Scheduling**:
   - **FIFO**: Simple but unfair
   - **Fair Queuing**: Perfect fairness but O(log n) complexity
   - **Deficit Round Robin**: O(1) approximation of fair queuing, scalable

5. **Traffic Shaping**:
   - **Leaky Bucket**: Smooths traffic to constant rate (CBR services)
   - **Token Bucket**: Allows bursts up to B, average rate R (SLAs)

### Common Patterns

**Router Design Trade-offs**:
- Memory vs. Speed (multibit tries)
- Centralized vs. Distributed (output queuing vs. PIM)
- Fairness vs. Complexity (Fair Queuing vs. DRR)

**QoS Mechanisms**:
- Classification → Scheduling → Shaping
- Proactive (shaping at source) vs. Reactive (policing at network)

**Algorithm Selection**:
- Few rules → Linear search
- Exact match → Hashing
- Prefix match → Tries
- Multi-field → Grid of Tries or decision trees

---

## See Also

- [[02-Network-Layer-and-Routing]] - Routing algorithms and BGP
- [[03-Transport-Layer]] - TCP congestion control (complements QoS)
- [[05-Modern-Architectures]] - SDN control of QoS policies
- [[06-Application-Layer-Services]] - VoIP and streaming QoS requirements

**Next**: [[05-Modern-Architectures]]
