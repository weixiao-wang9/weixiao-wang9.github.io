---
type: source
course: "[[Computer Networks (CS 6250)]]"
created: 2026-02-01
prerequisites: None
---

# Fundamentals and Architecture

> **Prerequisites**: None (start here)
> **Learning Goals**: After reading this, you will understand the Internet's layered architecture, the end-to-end principle, why IPv4/TCP are hard to replace, and how Layer 2 switching works.

## Introduction

The Internet connects billions of hosts running diverse applications across different types of networks. To manage this complexity, network designers use a **layered architecture** that divides functionality into distinct layers, each offering specific services. This file covers the foundational concepts that underpin all computer networking.

---

## History of the Internet

### Early Developments (1960s-1970s)

**1960s: The Vision**
- **J.C.R. Licklider** proposed the "Galactic Network" concept - a globally interconnected set of computers
- **ARPANET (1969)**: The first packet-switching network, connecting four nodes:
  - UCLA
  - Stanford Research Institute (SRI)
  - UC Santa Barbara (UCSB)
  - University of Utah

**1970s: Protocol Development**
- **NCP (Network Control Protocol)**: The first host-to-host protocol
- **1973: TCP/IP Introduced** by Bob Kahn and Vint Cerf
  - Enabled "open-architecture networking"
  - Allowed different networks to interconnect
  - Foundation of the modern Internet

### Growth and Standardization (1980s-1990s)

**Key Milestones**:
- **1983: DNS (Domain Name System)** - Solved scalability issues for hostname resolution
- **1990: World Wide Web (WWW)** by Tim Berners-Lee
  - Made the Internet accessible to non-technical users
  - Popularized HTTP and HTML
  - Catalyzed explosive growth

**Result**: From 4 nodes in 1969 to billions of connected devices today

---

## Layered Architecture

### Goal and Advantages

**Primary Goal**: Enable communication between hosts running the same applications but located in different types of networks

**Why Layering?**
1. **Scalability**: Can add new protocols and technologies without redesigning everything
2. **Modularity**: Each layer solves a specific problem independently
3. **Flexibility**: Layers can be updated or replaced without affecting others

**Analogy**: Airline System
```
Ticket Purchase    →  Application decides what you want
Baggage Check      →  Prepare items for transport
Gate Assignment    →  Organize departure logistics
Runway Takeoff     →  Physical movement
```

Each layer serves the one above it, hiding implementation details.

### The OSI Model vs Internet Model

**OSI Model (7 Layers)** - International Organization for Standardization:
```
7. Application Layer
6. Presentation Layer
5. Session Layer
4. Transport Layer
3. Network Layer
2. Data Link Layer
1. Physical Layer
```

**Internet Model (5 Layers)** - Practical implementation:
```
5. Application Layer      (Combines OSI Layers 5-7)
4. Transport Layer
3. Network Layer
2. Data Link Layer
1. Physical Layer
```

![](</images/Screenshot 2025-08-21 at 6.57.19 AM.png>)

The Internet model consolidates the top three OSI layers into a single **Application Layer** for simplicity.

---

## Layer-by-Layer Breakdown

### Understanding Each Layer

**Every layer can be explained using three concepts**:

1. **Service**: What the layer provides to the layer above
2. **Interface**: How the layer above accesses this service
3. **Protocol**: The rules peers follow to implement the service

![](</images/Screenshot 2025-08-21 at 7.15.49 AM.png>)

---

### Layer 7-5: Application Layer

**Role**: Provides network services directly to end-user applications

**Key Protocols**:
- **HTTP**: Web browsing (client-server communication)
- **SMTP**: Email transmission
- **FTP**: File transfer between hosts
- **DNS**: Translates domain names to IP addresses

**Data Unit**: **Message**

**Example**:
- **Service**: "Let apps communicate across the network" (e.g., Gmail sending email)
- **Interface**: Application APIs (e.g., browser's HTTP interface)
- **Protocol**: HTTP, FTP, DNS - define the "how"

---

### Layer 6: Presentation Layer (OSI)

**Role**: Formats and translates information between the application and network

**Functions**:
- Data encryption/decryption
- Data compression
- Character encoding translation (ASCII, Unicode)

![](</images/Screenshot 2025-08-22 at 2.14.31 AM.png>)

**Note**: In the Internet model, these functions are handled within the Application Layer.

---

### Layer 5: Session Layer (OSI)

**Role**: Manages sessions between end-user applications

**Functions**:
- Session establishment, maintenance, and termination
- Synchronization
- Dialog control

![](</images/Screenshot 2025-08-22 at 2.21.37 AM.png>)

**Note**: In the Internet model, session management is application-specific.

---

### Layer 4: Transport Layer

**Role**: Provides end-to-end communication between **applications** (not just hosts)

**Key Protocols**:

**TCP (Transmission Control Protocol)**:
- **Connection-oriented**: Establishes connection before data transfer
- **Reliable delivery**: Guarantees data arrives correctly and in order
- **Flow control**: Matches sender and receiver speeds
- **Congestion control**: Prevents network overload

**UDP (User Datagram Protocol)**:
- **Connectionless**: No connection setup
- **Best-effort delivery**: No guarantees (fast but unreliable)
- **No flow or congestion control**

**Data Unit**: **Segment**

**Key Distinction**:
- Network Layer = delivery between **hosts** (machines)
- Transport Layer = delivery between **applications** (processes)

![](</images/Screenshot 2025-08-22 at 3.42.20 AM.png>)

---

### Layer 3: Network Layer

**Role**: Routes datagrams from source host to destination host across the Internet

**How It Works**:
1. Source host's **transport layer** passes a segment to the network layer
2. Network layer **wraps the segment in a datagram** with destination IP
3. Datagram travels through intermediate routers
4. Destination host's network layer receives datagram and passes segment up to transport layer

**Key Protocols**:

**IP (Internet Protocol)**:
- Defines datagram format and addressing
- Specifies how hosts and routers process packets
- **Every Internet device must run IP**

**Routing Protocols**:
- Determine paths datagrams take between sources and destinations
- Examples: OSPF, RIP, BGP

**Data Unit**: **Datagram** (or Packet)

![](</images/Screenshot 2025-08-22 at 3.42.04 AM.png>)

---

### Layer 2: Data Link Layer

**Role**: Transfers frames from one node to the next node on the same link

**Key Distinction**:
- Data Link Layer = **one hop** (node to adjacent node)
- Network Layer = **full journey** (source to destination)

**How It Works** (Example: Host A → Router 1 → Router 2 → Host B):
1. Host A's network layer creates a datagram
2. Host A's data link layer wraps it in a **frame** with MAC addresses
3. Router 1 receives frame, extracts datagram, determines next hop
4. Router 1's data link layer creates a **new frame** for the next link
5. Process repeats until datagram reaches Host B

**Services**:
- Reliable delivery **over a single link** (different from TCP's end-to-end reliability)
- Error detection and correction
- Media access control (MAC)

**Data Unit**: **Frame**

**Key Protocols**: Ethernet, Wi-Fi (802.11), PPP

![](</images/Screenshot 2025-08-22 at 3.41.45 AM.png>)

---

### Layer 1: Physical Layer

**Role**: Transfers individual bits within a frame between two nodes connected by a physical link

**Functions**:
- Converts bits to electrical/optical/radio signals
- Defines physical characteristics (voltage levels, timing, cable specs)

**Data Unit**: **Bits**

**Note**: Protocols vary by transmission medium (copper wire, fiber optic, wireless)

![](</images/Screenshot 2025-08-22 at 3.41.32 AM.png>)

---

## Encapsulation and De-encapsulation

### How Layers Communicate

Layers communicate through **encapsulation** (at sender) and **de-encapsulation** (at receiver).

**Encapsulation** (Sender's side):
- Each layer wraps data from the layer above with its own header (and sometimes trailer)
- Headers contain information needed by the corresponding layer on the receiving side

**Example: Sending "Hello" via a web browser**

1. **Application Layer**: Creates message = `"Hello"`
2. **Transport Layer**: Adds header `Ht` (port numbers, checksum, sequence) → **Segment**
3. **Network Layer**: Adds header `Hn` (source/dest IP, TTL) → **Datagram**
4. **Link Layer**: Adds header `Hl` (source/dest MAC, CRC) → **Frame**
5. **Physical Layer**: Converts to bits and transmits

```
[Physical Bits] ← [Hl | Hn | Ht | "Hello"] Frame
```

**De-encapsulation** (Receiver's side):
- Each layer strips off its header, interprets it, and passes payload upward

**Continuing the example**:
1. **Link Layer**: Removes MAC header (`Hl`), passes packet up
2. **Network Layer**: Removes IP header (`Hn`), passes segment up
3. **Transport Layer**: Removes TCP/UDP header (`Ht`), passes message up
4. **Application Layer**: Sees `"Hello"` and delivers to browser

### Intermediate Devices

**Routers** (Layers 1-3):
- Open Link and Network headers to determine forwarding path
- Do **partial de-encapsulation** to read IP addresses

**Switches** (Layers 1-2):
- Open only Link header to see MAC addresses
- Forward frames based on MAC table

**Key Insight**: Intermediate devices don't go all the way up to the application layer.

---

## The End-to-End Principle

### Core Concept

**Definition**: Application-level functions should **not** be built into the lower levels of the network core. Intelligence should reside at the **end systems** (hosts), not in the network itself.

**Philosophy**:
- Network core should be **simple and minimal**
- End systems should carry the **intelligence**

![](</images/Screenshot 2025-08-22 at 3.51.58 AM.png>)

### Why This Principle?

**Original Design Goals**:
1. **Flexibility**: Moving functions to end systems increases autonomy for application designers
2. **Innovation**: Easier to deploy new applications without changing network infrastructure
3. **Reliability**: Functions like encryption and error correction work better when handled end-to-end

**Example**: File Transfer
- Network provides **best-effort delivery** (might lose packets)
- TCP at **end hosts** ensures **reliability** (retransmits lost packets)
- Network doesn't need to track state for every connection

### Violations of E2E

Despite the principle, some functions have moved into the network core:

**1. Firewalls**
- Inspect and filter packets based on content
- Violates E2E by making decisions about application-layer data

**2. NAT (Network Address Translation) Boxes**
- Translate private IP addresses to public IPs
- Break E2E connectivity (hosts behind NAT are not directly addressable)

![](</images/Screenshot 2025-08-22 at 3.58.33 AM.png>)

**How NAT Works**:
- **Outgoing traffic**: Rewrites source IP/port (private → public)
- **Incoming traffic**: Rewrites destination IP/port (public → private) using a NAT translation table

**Why NAT Violates E2E**:
- External hosts cannot directly initiate connections to NATted devices
- Requires NAT traversal techniques (STUN, TURN)

**Trade-off**: NAT solves IPv4 address exhaustion but complicates peer-to-peer applications.

---

## The Hourglass Shape of Internet Architecture

### Observation

The Internet protocol stack has an **hourglass shape**:
- **Wide at bottom** (Physical/Link): Many technologies (Ethernet, Wi-Fi, fiber, DSL, etc.)
- **Narrow in middle** (Network): Dominated by **IPv4**
- **Wide at top** (Application): Many protocols (HTTP, SMTP, FTP, DNS, etc.)

![](</images/Screenshot 2025-08-22 at 4.20.53 AM.png>)

**Question**: Why has IP become such a dominant "waist"? Why is it so hard to replace IPv4 with IPv6 or introduce new transport protocols?

---

### The EvoArch Model

**EvoArch** (Evolutionary Architecture) is a model that explains this hourglass shape.

**Key Components**:

**1. Layers (L)**:
- Just like OSI/TCP, EvoArch has layers
- Each layer is a "stage" where protocols compete

**2. Nodes**:
- Each **protocol** is a node (e.g., Ethernet, IP, TCP, HTTP)
- The layer of node `u` is written as `l(u)`

**3. Edges (Dependencies)**:
- If protocol `u` **uses** protocol `w`, draw edge `w → u`
- Example: HTTP → TCP (HTTP depends on TCP)
- Forms a **directed acyclic graph (DAG)**

**4. Substrates and Products**:
- **Substrates S(u)**: Protocols that node `u` depends on
  - Example: TCP's substrate = IP
- **Products P(u)**: Protocols that depend on node `u`
  - Example: TCP's products = HTTP, SMTP, FTP, etc.

**5. Layer Generality s(l)**:
- Lower layers are **more general** (more protocols use them)
- Higher layers are **less general** (more specialized)
- `s(l)` = Probability that a node in layer `l+1` picks a substrate in layer `l`
- This probability **decreases** as you go up

**6. Evolutionary Value v(u)**:
- A protocol's value depends on **how many valuable protocols depend on it**
- Computed recursively: If many high-value protocols use you, your value is high
- Example: IP has high value because TCP/UDP depend on it, which in turn have high value because many apps depend on them

**7. Competition**:
- **Competitors C(u)**: Nodes at the same layer that share ≥ fraction `c` of products
- Example: TCP and UDP are competitors (both used by many apps)
- **Competition Threshold c**: How much overlap counts as competition

**Death Mechanism**:
- A node is more likely to **die** if its competitors have **higher evolutionary value**
- If a node dies, its products also die (unless they have alternative substrates)

---

### EvoArch Iteration Process

**Each round has three phases**:

**1. Birth**:
- Small fraction of new nodes added randomly to layers
- Simulates protocol innovation

**2. Update (Top to Bottom)**:
- New node at layer `l` selects:
  - **Substrates** from layer `l-1` with probability `s(l-1)`
  - **Products** from layer `l+1` with probability `s(l)`
- **Recompute values**: Update `v(u)` for all nodes based on new dependencies

**3. Competition and Death**:
- Within same layer, protocols sharing products compete
- Lower-value protocols may die
- If a protocol dies, its products die too (cascade effect)

**Result**: After many rounds, the stack exhibits an **hourglass shape**:
- Broad at bottom (many physical/link technologies)
- Narrow in middle (few dominant protocols like IP)
- Broad at top (many applications)

![](</images/Screenshot 2025-08-22 at 9.59.06 AM.png>)

**Visualization**:
- **Inward edges** (arrows pointing into a node from below) = **Substrates** (dependencies)
- **Outward edges** (arrows leaving a node upward) = **Products** (dependents)

---

### Why IPv4/TCP/UDP Are Hard to Replace

**High Evolutionary Value**:
- Almost all higher-layer protocols depend on IPv4/TCP/UDP
- Replacing them would require replacing all dependent protocols

**Evolutionary Shield**:
- TCP/UDP's stability protects IPv4
- New transport protocols struggle to gain adoption (chicken-and-egg problem)
- IPv6 adoption is slow because it requires changing the narrow waist

**Network Effects**:
- Existing infrastructure is built around IPv4/TCP/UDP
- Switching costs are enormous

**Ramifications**:
1. Many technologies adapted to work **over IP** (Radio over IP, Voice over IP)
2. IPv6 transition has been extremely slow despite address exhaustion

---

## Clean-Slate Internet Architecture Redesign

### Motivation

**Current Internet Challenges**:
- Security vulnerabilities (DDoS, spoofing)
- Lack of accountability (hard to trace attackers)
- Limited QoS support
- Management complexity

**Clean-Slate Approach**:
- Redesign the Internet **from scratch**
- Test new assumptions, architectures, and services
- Use experimental facilities with real users

### Potential Outcomes

1. **Incremental improvements**: New services adopted in today's Internet
2. **Revolutionary change**: Entirely new architecture
3. **Validation**: Proof that current Internet is already optimal

### Example: Accountable Internet Protocol (AIP)

**Goal**: Improve **accountability** at the network layer

**Address Format**: `AD:EID`
- `AD` = Network ID (Autonomous Domain)
- `EID` = Unique host ID within that domain

**Key Features**:

**1. Source Accountability**:
- Verify packet sources (prevent IP spoofing)
- Trace actions to specific end hosts
- "Shut-off message" mechanism to stop misbehaving hosts

**2. Control-Plane Accountability**:
- Origin and path authentication for routing
- Prevents BGP hijacking and route manipulation
- Pinpoint and prevent routing attacks

**Trade-off**: More accountability vs. privacy concerns

---

## Interconnecting Hosts and Networks

### Layer 1: Repeaters and Hubs

**Function**: Receive and forward digital signals to connect different Ethernet segments

**Advantages**:
- Simple and cheap
- Extend signal range

**Disadvantages**:
- **Single collision domain**: All hosts share same bandwidth
- No intelligence (broadcasts everything)

**Use Case**: Legacy networks (mostly replaced by switches)

---

### Layer 2: Bridges and Switches

**Function**: Enable communication between hosts that are **not directly connected**

**Key Difference from Hubs**:
- Forward frames based on **MAC addresses**
- Create **separate collision domains** (one per port)
- **Learn** which hosts are on which ports

**Advantages**:
- Increased bandwidth (parallel transmission on different ports)
- Reduced collisions

**Disadvantages**:
- Buffer overflow risk
- Packet drops under heavy load

---

### Learning Bridges

**Goal**: Maintain a forwarding table to intelligently forward frames

**How Learning Works**:

![](</images/Screenshot 2025-08-22 at 10.34.16 AM.png>)

**Setup**:
- Bridge has **two ports**, creating **two collision domains**
- Port 1 LAN, Port 2 LAN

**Bridge Behavior**:
1. When a frame arrives, bridge looks at **source MAC** and records:
   - Source MAC → Port it arrived on
2. Bridge looks up **destination MAC** in forwarding table:
   - **If known**: Forward only to that port
   - **If source and destination on same port**: Drop (don't waste bandwidth)
   - **If unknown**: Flood to all ports except incoming port

**Example**:
```
Frame from Host A (MAC: AA:AA) arrives on Port 1
  → Bridge learns: AA:AA is on Port 1

Frame destined for Host B (MAC: BB:BB) arrives on Port 1
  → If BB:BB is in table on Port 2 → Forward to Port 2
  → If BB:BB is in table on Port 1 → Drop (same segment)
  → If BB:BB unknown → Flood to Port 2
```

**Benefits**:
- Self-configuring (no manual setup)
- Efficient (only forwards where needed)

---

### The Spanning Tree Protocol (STP)

**Problem**: Network topologies with loops cause **infinite frame forwarding**

**Example**:
```
Bridge A ← → Bridge B
    ↓           ↓
    └─ Bridge C ─┘
```
A frame could loop: A → B → C → A → ...

**Solution: Spanning Tree Algorithm**

**Goal**: Create a loop-free logical topology by disabling certain links

**How It Works**:

**1. Root Bridge Election**:
- Every bridge starts assuming **itself is the root**
- Bridges exchange **configuration messages**: `<RootID, DistanceToRoot, SenderID>`
- Bridges adopt the **best configuration** based on:
  1. Smaller `RootID` wins
  2. If equal RootID → Smaller `DistanceToRoot` wins
  3. If still equal → Smaller `SenderID` wins
- Bridge with **lowest ID** becomes the **root bridge**

**2. Root Port Selection**:
- Each non-root bridge selects one **root port** (shortest path to root)

**3. Designated Bridge per LAN**:
- Each LAN segment elects one **designated bridge** (closest to root)
- Only the designated bridge forwards frames on that segment

**4. Disable Other Ports**:
- All other ports are **disabled** to break loops

**Result**: A tree structure with the root at the top, no loops

**Example**:
```
Initial: All bridges claim to be root
Round 1: Bridge 1 (ID=1) sends <1, 0, 1>
         Bridge 2 (ID=2) sends <2, 0, 2>
         Bridge 3 (ID=3) sends <3, 0, 3>

Round 2: All bridges receive messages
         All adopt Bridge 1 as root (lowest ID)
         Bridge 2 sends <1, 1, 2> (1 hop from root)
         Bridge 3 sends <1, 1, 3>

Round 3: Converges
         Root: Bridge 1
         Bridge 2's root port: toward Bridge 1
         Bridge 3's root port: toward Bridge 1
         Some ports disabled to prevent loops
```

**Advantages**:
- Prevents broadcast storms
- Automatic and distributed (no central controller)

**Disadvantages**:
- Disabled links waste capacity
- Convergence time after topology change

---

### Layer 3: Routers and Layer 3 Switches

**Function**: Forward packets based on **IP addresses** using routing protocols

**Key Difference from Layer 2**:
- Operate at Network Layer
- Use routing algorithms (OSPF, BGP) to compute paths
- Can interconnect different types of networks (e.g., Ethernet to Wi-Fi)

**Covered in detail in**: [[02-Network-Layer-and-Routing]]

---

## Summary

### Key Takeaways

1. **Layered Architecture**: The Internet uses a 5-layer stack (Application, Transport, Network, Data Link, Physical) for modularity and scalability

2. **End-to-End Principle**: Intelligence belongs at end systems, not in the network core; violations include NAT and firewalls

3. **Encapsulation**: Each layer adds headers; intermediate devices do partial de-encapsulation based on their function

4. **Hourglass Shape**: The protocol stack narrows at IP due to evolutionary forces; high-value protocols with many dependents are hard to replace

5. **IPv4/TCP/UDP Stability**: These protocols dominate due to network effects and evolutionary value, making clean-slate redesign difficult

6. **Layer 2 Switching**: Bridges learn MAC addresses and use Spanning Tree to prevent loops

### Common Patterns

**Protocol Design**:
- Each layer solves a specific problem independently
- Lower layers provide services to upper layers
- Layers communicate through well-defined interfaces

**Evolution**:
- Successful protocols become hard to replace
- Innovation happens at the edges (applications), not the core
- Backwards compatibility constrains new designs

**Tradeoffs**:
- Layering adds overhead but provides modularity
- E2E principle promotes simplicity but some functions work better in the network
- Accountability vs. privacy in redesigns

---

## See Also

- [[02-Network-Layer-and-Routing]] - Deep dive into IP, routing protocols, and router architecture
- [[03-Transport-Layer]] - Details on TCP and UDP
- [[05-Modern-Architectures]] - SDN as a departure from traditional architecture

**Next**: [[02-Network-Layer-and-Routing]]
