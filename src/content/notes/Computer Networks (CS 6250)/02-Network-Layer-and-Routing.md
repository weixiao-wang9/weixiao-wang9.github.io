---
type: source
course: "[[Computer Networks (CS 6250)]]"
created: 2026-02-01
prerequisites: "[[01-Fundamentals-and-Architecture]]"
---

# Network Layer and Routing

> **Prerequisites**: [[01-Fundamentals-and-Architecture]]
> **Learning Goals**: After reading this, you will understand IP addressing and routing, distinguish intradomain vs interdomain routing, explain BGP policy routing, and understand router internals and optimization.

## Introduction

The Network Layer is responsible for moving datagrams from source host to destination host across the Internet. This involves two critical functions:
1. **Forwarding**: Moving packets from input to output within a router (data plane)
2. **Routing**: Determining the path packets take through the network (control plane)

This file covers routing algorithms, protocols (OSPF, RIP, BGP), the business side of the Internet, and router architecture.

---

## Routing vs. Forwarding

### Key Distinction

**Forwarding** (Data Plane):
- **Local action**: Transferring a packet from an incoming link to an outgoing link **within a single router**
- **Fast**: Happens in nanoseconds (hardware-based)
- Uses a **forwarding table** (also called FIB - Forwarding Information Base)

**Routing** (Control Plane):
- **Global action**: Determining the "good paths" from source to destination across the **entire network**
- **Slower**: Happens in seconds/minutes (software-based)
- Uses **routing protocols** to build routing tables
- Routing tables are compiled into forwarding tables

**Analogy**:
- Routing = Planning your road trip route (Google Maps)
- Forwarding = Making each turn at each intersection

---

## Intradomain Routing

**Definition**: Routing within a single **administrative domain** (e.g., within one ISP, one university network, one company)

**Also called**: Interior Gateway Protocols (IGPs)

**Goal**: Find the shortest/fastest path within the domain

**Two Main Algorithm Classes**:
1. **Link State** (Global knowledge)
2. **Distance Vector** (Local knowledge)

---

## Link State Routing

### Concept

**How It Works**:
- Each router has **complete knowledge** of the network topology and all link costs
- All routers run the **same algorithm** independently
- Uses **Dijkstra's Algorithm** to compute shortest paths

**Process**:
1. **Discover neighbors**: Each router identifies its directly connected neighbors
2. **Measure link costs**: Determine the cost to each neighbor (delay, bandwidth, etc.)
3. **Flood topology info**: Broadcast **Link State Advertisements (LSAs)** to all routers
4. **Build topology map**: Each router constructs a complete graph of the network
5. **Compute paths**: Run Dijkstra's algorithm to find shortest paths to all destinations

### Dijkstra's Algorithm

**Input**: Graph with nodes (routers) and weighted edges (link costs)

**Output**: Shortest path tree from source to all other nodes

**Algorithm**:
```
Initialize:
  - Set distance to source = 0
  - Set distance to all others = ∞
  - Mark all nodes as unvisited

Repeat until all nodes visited:
  1. Select unvisited node u with minimum distance
  2. Mark u as visited
  3. For each unvisited neighbor v of u:
       if distance[u] + cost(u,v) < distance[v]:
          distance[v] = distance[u] + cost(u,v)
          predecessor[v] = u
```

**Complexity**: O(n²) where n = number of nodes
- Can be optimized to O(n log n) with priority queue

**Example**:
```
Network:
    A ---2--- B
    |         |
    1         3
    |         |
    C ---1--- D

From A:
  Step 1: Visit A (dist=0)
          Update: C=1, B=2
  Step 2: Visit C (dist=1)
          Update: D=2
  Step 3: Visit B (dist=2) or D (dist=2)
          ...

Result: Shortest paths from A:
  A→C: 1
  A→B: 2
  A→D: 2
  A→B→D: 5 (not chosen, longer than A→C→D)
```

---

### OSPF (Open Shortest Path First)

**Type**: Link State protocol

**Key Features**:

**1. Hierarchical Structure**:
- Network divided into **Areas**
- **Area 0 (Backbone)**: Central area connecting all other areas
- **Area Border Routers (ABRs)**: Connect areas to the backbone
- Reduces LSA flooding scope

**2. Link State Advertisements (LSAs)**:
- Routers flood LSAs when:
  - Link state changes (link up/down)
  - Cost changes
  - Periodically (every 30 minutes by default)
- LSAs propagate only within an area (or to backbone if inter-area)

**3. Cost Metric**:
- Typically based on **bandwidth**: Cost = Reference Bandwidth / Link Bandwidth
- Administrators can manually set costs

**4. Fast Convergence**:
- Triggered updates when topology changes
- Converges faster than Distance Vector protocols

**Advantages**:
- Scalability through hierarchy
- No routing loops (uses complete topology)
- Supports load balancing (equal-cost multi-path)

**Disadvantages**:
- Complex configuration
- Higher memory requirements (stores full topology)

---

## Distance Vector Routing

### Concept

**How It Works**:
- Routers only know about their **immediate neighbors**
- **Iterative and distributed**: Each router updates its table based on neighbors' information
- Uses **Bellman-Ford Equation** to compute shortest paths

**Process**:
1. Each router maintains a **distance vector**: Distances to all destinations
2. Periodically, routers exchange distance vectors with neighbors
3. Each router updates its table using the Bellman-Ford equation
4. Process repeats until convergence (no more changes)

### Bellman-Ford Equation

**Formula**:
```
D_x(y) = min over all neighbors v { cost(x,v) + D_v(y) }
```

**Meaning**:
- `D_x(y)` = Distance from router x to destination y
- `cost(x,v)` = Direct link cost from x to neighbor v
- `D_v(y)` = Neighbor v's distance to y
- Choose neighbor v that minimizes total distance

**Example**:
```
Router A wants to reach Router D

A has neighbors: B (cost 2), C (cost 1)
B says: D is 3 hops away
C says: D is 1 hop away

D_A(D) = min {
  cost(A,B) + D_B(D) = 2 + 3 = 5,
  cost(A,C) + D_C(D) = 1 + 1 = 2  ← Choose this
}

Result: A routes to D via C, total distance = 2
```

---

### RIP (Routing Information Protocol)

**Type**: Distance Vector protocol

**Key Features**:

**1. Metric**:
- Uses **hop count** as cost (number of routers traversed)
- Maximum hop count: 15 (16 = infinity/unreachable)
- Simple but doesn't account for bandwidth or delay

**2. Updates**:
- Routers broadcast distance vectors every **30 seconds**
- **Timeout**: If no update from neighbor for 180 seconds, mark routes via that neighbor as unreachable

**3. Triggered Updates**:
- Send immediate update if routing table changes

**Advantages**:
- Simple to configure and implement
- Low overhead (small routing tables)

**Disadvantages**:
- Slow convergence (can take minutes)
- Prone to **count-to-infinity** problem

---

### Count-to-Infinity Problem

**Problem**: When a link cost **increases** or a link **fails**, Distance Vector protocols can loop indefinitely while incrementing distance.

**Example**:
```
Initial state:
  A → B (cost 1)
  B → C (cost 1)
  A knows: C is 2 hops away (via B)
  B knows: C is 1 hop away (direct)

Link B-C fails:
  B no longer has direct route to C
  B receives update from A: "C is 2 hops away"
  B thinks: I can reach C via A (cost = 1 + 2 = 3)
  B updates: C is 3 hops away

  A receives update from B: "C is 3 hops away"
  A thinks: I can reach C via B (cost = 1 + 3 = 4)
  A updates: C is 4 hops away

  This continues: 3 → 4 → 5 → 6 → ... → ∞
```

**Why It Happens**:
- Routers don't know the **path** their neighbors use
- A and B keep bouncing the route back and forth

**Mitigation: Poison Reverse**:
- If router X routes to destination Z through neighbor Y, then X advertises to Y: "Distance to Z = ∞"
- Prevents Y from routing back through X

**Example with Poison Reverse**:
```
A routes to C via B
  A tells B: "My distance to C = ∞" (poison)

When B-C link fails:
  B receives from A: "C = ∞"
  B cannot use A as path to C
  B immediately marks C as unreachable

No count-to-infinity!
```

**Limitation**: Poison Reverse only prevents 2-node loops, not loops involving 3+ routers.

---

## Link State vs Distance Vector Summary

| Aspect | Link State (OSPF) | Distance Vector (RIP) |
|--------|-------------------|----------------------|
| **Knowledge** | Complete topology | Only neighbors |
| **Algorithm** | Dijkstra | Bellman-Ford |
| **Updates** | Triggered by changes | Periodic (30s) |
| **Convergence** | Fast (seconds) | Slow (minutes) |
| **Loops** | No loops | Possible (count-to-infinity) |
| **Memory** | High (full topology) | Low (distance table) |
| **Complexity** | O(n²) or O(n log n) | O(n × neighbors) |
| **Scalability** | Hierarchical (areas) | Limited (max 15 hops) |

---

## Interdomain Routing

### The Internet Ecosystem

**Hierarchy**:

**Tier-1 ISPs (Global Backbone)**:
- Examples: AT&T, NTT, Level 3
- Span continents
- **Peer** with each other (settlement-free exchange)
- Do not pay for transit

**Tier-2 ISPs (Regional)**:
- Connect to Tier-1 ISPs as **customers**
- May peer with other Tier-2 ISPs
- Serve regional areas

**Tier-3 ISPs (Access)**:
- Connect end-users (homes, businesses)
- Pay Tier-2 or Tier-1 ISPs for transit

**Other Key Players**:
- **IXPs (Internet Exchange Points)**: Physical locations where networks interconnect
- **CDNs (Content Delivery Networks)**: Distributed servers (Google, Netflix) that push content closer to users

---

### Autonomous Systems (AS)

**Definition**: A group of routers under the **same administrative authority** with a **unified routing policy**

**AS Number (ASN)**:
- Unique identifier (16-bit or 32-bit)
- Example: AS 7018 (AT&T), AS 15169 (Google)

**Routing Domains**:
- **Intradomain (IGP)**: Routing **within** an AS (OSPF, RIP)
- **Interdomain (EGP)**: Routing **between** ASes (BGP)

---

### BGP (Border Gateway Protocol)

**Purpose**: The standard protocol for exchanging routing information **between** Autonomous Systems

**Type**: Path Vector protocol (variant of Distance Vector)

**Key Features**:
- **Policy-based routing**: Routes chosen based on business relationships, not just shortest path
- **Scalability**: Handles global Internet routing (800,000+ routes)

**Two Flavors**:

**1. eBGP (External BGP)**:
- Runs between routers in **different ASes**
- Exchanges routes learned from external neighbors

**2. iBGP (Internal BGP)**:
- Runs between routers **within the same AS**
- Distributes external routes learned via eBGP to all internal routers
- Does NOT run routing algorithms (just distributes info)

**BGP Message**:
```
BGP Advertisement = {
  Prefix: 192.168.1.0/24
  AS Path: [AS100, AS200, AS300]
  Next Hop: 10.0.0.1
  Attributes: LocalPref, MED, ...
}
```

---

### Business Relationships

**1. Customer-Provider (Transit)**:
- **Customer pays Provider** for Internet access
- Provider forwards **all traffic** for customer (to anywhere on the Internet)
- **Provider exports customer routes** to everyone (customers, peers, other providers)
- **Customer imports all routes** from provider

**Example**:
```
Tier-3 ISP (customer) ← pays → Tier-1 ISP (provider)
```

**2. Peering**:
- **Settlement-free** exchange (no money changes hands)
- Usually between networks of **similar size**
- **Restricted traffic exchange**: Only for their own customers (no transit)
- **Export policy**: Do NOT export peer routes to other peers or providers

**Example**:
```
Tier-1 ISP A ←→ Tier-1 ISP B (peers)
```

**Why Peer?**
- Reduce costs (avoid paying transit fees)
- Improve performance (direct path)
- Increase resilience (backup paths)

---

### BGP Route Selection

**Problem**: Router receives multiple BGP advertisements for the same prefix. Which to choose?

**BGP Decision Process** (in order):

1. **Highest LocalPref** (Local Preference)
   - Operator-defined preference
   - Higher value = more preferred
   - Used to control **outbound traffic** (which exit point to use)

2. **Shortest AS Path**
   - Fewer ASes traversed = shorter path
   - Can be manipulated via AS path prepending

3. **Lowest Origin Type**
   - IGP < EGP < Incomplete

4. **Lowest MED** (Multi-Exit Discriminator)
   - Used to control **inbound traffic** (preferred entrance point)
   - Lower value = more preferred
   - Only compared between routes from the **same neighboring AS**

5. **eBGP over iBGP**
   - Prefer routes learned externally

6. **Lowest IGP Cost to Next Hop**
   - **Hot Potato Routing**: Get rid of traffic ASAP
   - Choose exit point closest to current router

7. **Lowest Router ID**
   - Tiebreaker

**Key Insight**: BGP is **policy-driven**, not distance-driven. Business relationships dictate routing decisions.

---

### BGP Policies and Traffic Control

**Import Preference** (Which routes to accept and prefer):

**Rule**: Customer > Peer > Provider

**Reasoning**:
- **Customer routes** = Revenue (customer pays you)
- **Peer routes** = Free exchange
- **Provider routes** = Cost (you pay provider)

**Example**:
```
AS 100 receives route to 192.168.1.0/24 from:
  - Customer AS 200: LocalPref = 200
  - Peer AS 300: LocalPref = 100
  - Provider AS 400: LocalPref = 50

AS 100 chooses Customer route (highest LocalPref)
```

**Export Rules** (Which routes to advertise to whom):

**Golden Rule**: **Do NOT provide free transit**

| Learn from | Export to Customer? | Export to Peer? | Export to Provider? |
|------------|---------------------|-----------------|---------------------|
| **Customer** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Peer** | ✅ Yes | ❌ No | ❌ No |
| **Provider** | ✅ Yes | ❌ No | ❌ No |

**Reasoning**:
- **Customer routes**: Export everywhere (generate revenue)
- **Peer/Provider routes**: Only export to customers (avoid providing free transit)

**Example**:
```
AS 100 learns route from Peer AS 300
  → AS 100 does NOT advertise to Peer AS 400 or Provider AS 500
  → Otherwise AS 100 would carry transit traffic between AS 300 and AS 400/500 for free
```

---

### Hot Potato Routing

**Definition**: Choose the **closest exit point** from your network to hand off traffic, regardless of the total path length.

**Goal**: Minimize cost by reducing distance traffic travels within your network

**How It Works**:
- AS has multiple exit points to reach a destination
- Choose the exit point with **lowest IGP cost** from current router

**Example**:
```
AS 100 has two routers that can reach AS 200:
  - Router A: IGP cost from source = 5
  - Router B: IGP cost from source = 10

AS 100 chooses Router A (hot potato: get rid of traffic ASAP)

Even if Router B → AS 200 is faster externally, Router A is chosen!
```

**Trade-off**: Optimizes your costs, but may not optimize end-to-end performance.

---

### IXPs (Internet Exchange Points)

**Definition**: Physical locations where multiple networks interconnect to exchange traffic

**Examples**: DE-CIX (Frankfurt), AMS-IX (Amsterdam), Equinix (global)

**Benefits**:
- **Reduced costs**: Avoid expensive transit fees
- **Reduced latency**: Direct connection to other networks
- **Increased capacity**: High-speed fabric

**How They Work**:
- Networks connect to a shared **switching fabric** (Layer 2 switch)
- BGP sessions established between networks over the IXP fabric
- Traffic exchanged locally instead of routing through providers

**Modern Trend: Topological Flattening**:
- More traffic exchanged at IXPs instead of global Tier-1 backbone
- CDNs (Google, Netflix) place servers at IXPs to serve content locally

---

## Router Architecture

### Control Plane vs Forwarding Plane

**Control Plane** (The Brain):
- **Implemented in software** (runs on CPU)
- **Function**: Runs routing protocols (BGP, OSPF) to build routing table
- **Speed**: Slow (milliseconds to seconds)
- **Location**: Routing Processor (general-purpose CPU)

**Forwarding Plane** (The Muscle):
- **Implemented in hardware** (ASICs, FPGAs)
- **Function**: Moves packets from input port to output port using forwarding table (FIB)
- **Speed**: Fast (nanoseconds)
- **Location**: Line cards (input/output ports)

**Key Separation**: Allows innovation in control plane (software-defined networking) without changing hardware.

---

### Router Components

**1. Input Ports**:
- **Physical termination**: Convert optical/electrical signals to bits
- **Data link processing**: Decapsulate frame, extract packet
- **Lookup**: Match destination IP against forwarding table (FIB)
- **Queueing**: Buffer packets if switching fabric is busy

**2. Switching Fabric**:
- **Function**: Interconnect that moves packets from input to output
- **Types**:
  - **Memory-based**: Packets copied to/from memory (oldest, slowest)
  - **Bus-based**: Packets traverse shared bus (limited by bus bandwidth)
  - **Crossbar-based**: Parallel paths (fastest, most expensive)

**3. Output Ports**:
- **Queueing**: Buffer packets waiting for transmission
- **Scheduling**: Determine which packet to send next (FIFO, priority, fair queueing)
- **Data link processing**: Encapsulate packet in frame
- **Physical transmission**: Convert bits to signals

**4. Routing Processor**:
- **Runs control plane software** (OSPF, BGP)
- **Builds routing table**
- **Compiles forwarding table** and distributes to line cards

---

### The Lookup Problem: Longest Prefix Match (LPM)

**Challenge**: IP forwarding uses **CIDR** (Classless Inter-Domain Routing), not exact matches.

**Example Forwarding Table**:
```
Prefix               Next Hop
192.168.1.0/24       Port 1
192.168.0.0/16       Port 2
0.0.0.0/0 (default)  Port 3
```

**Packet arrives**: Destination = 192.168.1.50

**Matches**:
- 192.168.1.0/24 ✅ (24-bit match)
- 192.168.0.0/16 ✅ (16-bit match)
- 0.0.0.0/0 ✅ (default, 0-bit match)

**Rule**: Choose the **longest prefix match** → 192.168.1.0/24 → Port 1

**Why Not Caching?**
- Cache hit rates are low (70-80%)
- Internet traffic flows are **short and diverse** (not repetitive)
- Misses still require full lookup

**Need**: Fast algorithm to handle millions of lookups per second

---

### Trie-Based Lookup Algorithms

**Goal**: Efficiently find the longest matching prefix

**1. Unibit Trie**:

**Structure**:
- Binary tree where each bit of the IP address determines left (0) or right (1) traversal
- Nodes contain prefix information

**Example**:
```
Prefixes:
  1*    (all addresses starting with 1)
  11*   (all starting with 11)
  101*  (all starting with 101)

Trie:
        Root
       /    \
     0*      1*   ← Match "1*"
            /  \
          10*  11* ← Match "11*"
          /
        101*       ← Match "101*"
```

**Lookup for IP 10110...**:
- Start at root
- Go right (1) → match "1*"
- Go left (0) → match "10*" (if exists)
- Go right (1) → match "101*"
- Result: Longest match is "101*"

**Advantages**:
- Memory efficient
- Simple implementation

**Disadvantages**:
- **Slow**: Requires up to 32 memory accesses for 32-bit address (IPv4)
- Not suitable for high-speed routers

---

**2. Multibit Trie**:

**Idea**: Instead of checking 1 bit at a time, check **k bits** at a time (stride = k)

**Example with stride = 2**:
```
Check 2 bits per step:
  00, 01, 10, 11 (4 branches per node)

For 32-bit address:
  32 / 2 = 16 steps (instead of 32)
```

**Trade-off**: **More memory** for **fewer lookups**

**Problem**: Prefixes might not align with stride boundaries

**Solution: Controlled Prefix Expansion**:
- Expand prefixes to match stride length
- Example: Prefix "1*" (length 1) expanded to "10*" and "11*" (length 2)

**Example**:
```
Original Prefixes:
  1*   → Port A
  11*  → Port B

Stride = 2 expansion:
  10*  → Port A  (expansion of 1*)
  11*  → Port B  (already length 2)
```

**Memory Cost**:
- More nodes due to expansion
- But faster lookup (fewer memory accesses)

**Typical Design**:
- Stride = 8 or 16 bits
- 32-bit lookup in 2-4 memory accesses
- Achieves line-rate forwarding (multi-Gbps)

---

### Traffic Engineering

**Definition**: Optimizing network resource utilization by controlling how traffic flows through the network

**Framework**:

**1. Measure**:
- Collect topology (routers, links, capacities)
- Measure traffic demands (traffic matrices)

**2. Model**:
- Predict traffic flow based on routing protocol
- Identify congested links

**3. Control**:
- Adjust link weights (OSPF/IS-IS costs)
- Traffic flows shift to alternate paths
- Re-measure and iterate

**Techniques**:
- **Weight optimization**: Find optimal link weights to balance load
- **MPLS Traffic Engineering**: Explicit path setup (not just shortest path)
- **SDN**: Centralized control for fine-grained traffic steering

---

## Summary

### Key Takeaways

1. **Routing vs Forwarding**: Routing computes paths (control plane), forwarding moves packets (data plane)

2. **Intradomain Routing**:
   - **Link State (OSPF)**: Global knowledge, Dijkstra, fast convergence, hierarchical
   - **Distance Vector (RIP)**: Local knowledge, Bellman-Ford, slow convergence, count-to-infinity problem

3. **Interdomain Routing (BGP)**:
   - **Policy-driven**, not distance-driven
   - Business relationships dictate routing (Customer > Peer > Provider)
   - Export rules prevent free transit
   - Hot potato routing minimizes costs

4. **Router Architecture**:
   - Control plane (software) vs Forwarding plane (hardware)
   - Longest Prefix Match for IP lookup
   - Multibit tries trade memory for speed

5. **Internet Ecosystem**:
   - Hierarchical (Tier-1, Tier-2, Tier-3)
   - IXPs enable local interconnection
   - Topological flattening due to CDNs

### Common Patterns

**Protocol Selection**:
- Intradomain: Optimize for performance (shortest path)
- Interdomain: Optimize for policy/economics (business relationships)

**Scalability**:
- Hierarchy reduces complexity (OSPF areas, AS structure)
- Aggregation reduces table size (CIDR)

**Trade-offs**:
- Link State: Fast but memory-intensive
- Distance Vector: Simple but slow convergence
- BGP: Scalable but complex policy interactions

---

## See Also

- [[01-Fundamentals-and-Architecture]] - Layering and encapsulation basics
- [[03-Transport-Layer]] - End-to-end reliability and congestion control
- [[04-Advanced-Routing-and-QoS]] - Packet classification and QoS
- [[07-Security-and-Governance]] - BGP hijacking and security

**Next**: [[03-Transport-Layer]]
