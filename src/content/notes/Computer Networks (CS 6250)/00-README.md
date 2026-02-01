---
type: meta
course: "[[Computer Networks (CS 6250)]]"
date: 2026-02-01
---

# Computer Networks Study Guide

A comprehensive guide to computer networking concepts, covering the Internet architecture from foundational principles to modern innovations like SDN and security mechanisms.

## 📚 Learning Path

### Prerequisites
Before starting, you should be familiar with:
- Basic programming concepts
- Understanding of binary and hexadecimal numbering systems
- General computer architecture (CPU, memory, I/O)

### Recommended Study Order

**Sequential Topics** (Follow this order):

1. **[[01-Fundamentals-and-Architecture]]** - Start here
   - Internet history and evolution
   - Layered architecture (OSI vs Internet model)
   - End-to-end principle
   - Layer 2 switching and bridges

2. **[[02-Network-Layer-and-Routing]]** - Builds on fundamentals
   - IP protocol and addressing
   - Routing algorithms (Link State, Distance Vector)
   - Intradomain routing (OSPF, RIP)
   - Interdomain routing (BGP, AS relationships)
   - Router architecture and forwarding

3. **[[03-Transport-Layer]]** - Requires network layer understanding
   - UDP and TCP protocols
   - Connection management (3-way handshake)
   - Flow control and congestion control
   - Multiplexing and demultiplexing

4. **[[04-Advanced-Routing-and-QoS]]** - Builds on routing basics
   - Packet classification algorithms
   - Quality of Service mechanisms
   - Traffic scheduling and shaping
   - Switching fabric design

**Modern Topics** (Can be studied after core topics, semi-independent):

5. **[[05-Modern-Architectures]]** - Clean-slate redesigns
   - Software-Defined Networking (SDN)
   - Control/Data plane separation
   - P4 programmable data planes
   - SDN controllers (OpenDayLight, ONOS)
   - Software-Defined Exchange (SDX)

6. **[[06-Application-Layer-Services]]** - Application-level systems
   - Content Distribution Networks (CDNs)
   - Server placement strategies
   - Multimedia streaming (DASH)
   - VoIP and real-time applications

7. **[[07-Security-and-Governance]]** - Security and policy
   - Network security fundamentals
   - DNS abuse and Fast-Flux networks
   - BGP hijacking attacks and defenses
   - DDoS attacks and mitigation
   - Internet censorship and surveillance

**Reference Material** (Use as needed):

8. **[[Appendix-Quick-Reference]]** - Quick lookup
   - Protocol summary table
   - Layer responsibilities
   - Common algorithms
   - Key formulas

---

## 🗺️ Concept Map

```
Fundamentals (Layering, E2E Principle)
    ↓
Network Layer (IP, Routing)
    ├─→ Intradomain (OSPF, RIP)
    ├─→ Interdomain (BGP, AS)
    └─→ Router Hardware
    ↓
Transport Layer (TCP, UDP)
    ├─→ Flow Control
    └─→ Congestion Control
    ↓
    ├─→ Advanced Routing & QoS
    │       ├─→ Packet Classification
    │       └─→ Traffic Shaping
    │
    ├─→ Modern Architectures
    │       ├─→ SDN (OpenFlow, P4)
    │       └─→ SDX
    │
    ├─→ Application Services
    │       ├─→ CDNs
    │       └─→ Multimedia/VoIP
    │
    └─→ Security & Governance
            ├─→ BGP Security
            ├─→ DDoS Defense
            └─→ Censorship
```

---

## 📋 File Descriptions

### 01-Fundamentals-and-Architecture
**Size**: ~12KB | **Estimated Reading Time**: 30-40 minutes

**Topics**:
- Internet history (ARPANET to WWW)
- Layered architecture design principles
- OSI vs Internet protocol stack
- Encapsulation and de-encapsulation
- End-to-end principle and violations (NAT, firewalls)
- Hourglass shape and evolutionary architecture (EvoArch)
- Clean-slate redesign (AIP)
- Layer 2 devices (bridges, switches, Spanning Tree)

**Key Learning Goals**:
- Understand why networks are layered and the tradeoffs
- Grasp the end-to-end principle and its implications
- Explain why IPv4/TCP/UDP are hard to replace
- Understand basic switching and bridging

**Prerequisites**: None (start here)

---

### 02-Network-Layer-and-Routing
**Size**: ~15KB | **Estimated Reading Time**: 45-60 minutes

**Topics**:
- IP protocol fundamentals and addressing
- Routing vs forwarding distinction
- Link State routing (Dijkstra, OSPF)
- Distance Vector routing (Bellman-Ford, RIP)
- Count-to-infinity problem and solutions
- Autonomous Systems and Internet ecosystem
- BGP (Border Gateway Protocol)
- Business relationships (customer-provider, peering)
- BGP policies and attributes (LocalPref, MED)
- Router architecture (control vs forwarding plane)
- Longest Prefix Match and trie algorithms
- Traffic engineering

**Key Learning Goals**:
- Understand how routing protocols compute paths
- Distinguish intradomain vs interdomain routing
- Explain BGP policy routing and business relationships
- Understand router internals and lookup optimization

**Prerequisites**: [[01-Fundamentals-and-Architecture]]

---

### 03-Transport-Layer
**Size**: ~8KB | **Estimated Reading Time**: 25-30 minutes

**Topics**:
- Transport layer role and services
- Multiplexing and demultiplexing (ports)
- UDP: connectionless, unreliable transport
- TCP: connection-oriented, reliable transport
- TCP 3-way handshake
- Flow control (receiver buffer protection)
- Congestion control (network protection)
- AIMD (Additive Increase Multiplicative Decrease)
- Slow start and TCP CUBIC

**Key Learning Goals**:
- Understand the difference between UDP and TCP
- Explain TCP connection management
- Understand flow control vs congestion control
- Grasp how TCP prevents network collapse

**Prerequisites**: [[02-Network-Layer-and-Routing]]

---

### 04-Advanced-Routing-and-QoS
**Size**: ~12KB | **Estimated Reading Time**: 40-50 minutes

**Topics**:
- Packet classification (multi-field matching)
- Classification algorithms (Set-Pruning, Grid of Tries)
- Crossbar switching fabrics
- Head-of-Line (HOL) blocking problem
- Parallel Iterative Matching (PIM)
- Quality of Service (QoS) fundamentals
- Scheduling algorithms (FIFO, Fair Queuing, DRR)
- Traffic shaping (Token Bucket, Leaky Bucket)
- Traffic policing

**Key Learning Goals**:
- Understand why simple forwarding isn't enough
- Explain packet classification challenges
- Understand HOL blocking and solutions
- Compare scheduling algorithms
- Distinguish traffic shaping vs policing

**Prerequisites**: [[02-Network-Layer-and-Routing]]

---

### 05-Modern-Architectures
**Size**: ~10KB | **Estimated Reading Time**: 30-40 minutes

**Topics**:
- Traditional network limitations
- Software-Defined Networking (SDN) paradigm
- Control/Data plane separation
- SDN history (Active Networks → OpenFlow)
- SDN architecture layers
- Controllers: centralized vs distributed (ONOS)
- Southbound APIs (OpenFlow)
- Northbound APIs (REST)
- P4 programming language
- Protocol-independent packet processing
- Software-Defined Exchange (SDX)
- IXP programmability

**Key Learning Goals**:
- Understand the motivation for SDN
- Explain control/data plane separation
- Compare centralized vs distributed controllers
- Understand P4's role in data plane programmability
- See how SDN applies to real problems (SDX)

**Prerequisites**: [[01-Fundamentals-and-Architecture]], [[02-Network-Layer-and-Routing]]

---

### 06-Application-Layer-Services
**Size**: ~8KB | **Estimated Reading Time**: 25-35 minutes

**Topics**:
- Content Distribution Networks (CDNs)
- Server placement strategies (Enter Deep vs Bring Home)
- Server selection mechanisms (DNS, IP Anycast, HTTP redirect)
- Consistent hashing for content mapping
- Internet topology flattening and IXPs
- Multimedia application categories
- VoIP mechanisms and QoS requirements
- Video compression (spatial and temporal redundancy)
- DASH (Dynamic Adaptive Streaming over HTTP)
- Bitrate adaptation algorithms (rate-based, buffer-based)

**Key Learning Goals**:
- Understand how CDNs reduce latency
- Explain server placement tradeoffs
- Distinguish multimedia application requirements
- Understand video streaming architecture (DASH)
- Compare bitrate adaptation strategies

**Prerequisites**: [[03-Transport-Layer]] (helpful but not strictly required)

---

### 07-Security-and-Governance
**Size**: ~8KB | **Estimated Reading Time**: 30-40 minutes

**Topics**:
- Security fundamentals (CIAA: Confidentiality, Integrity, Authentication, Availability)
- DNS abuse (Round-Robin DNS, Fast-Flux Service Networks)
- Network reputation systems (FIRE, ASwatch)
- BGP hijacking types (exact prefix, sub-prefix, squatting)
- BGP hijacking defense (ARTEMIS)
- DDoS attacks (spoofing, reflection, amplification)
- DDoS mitigation (BGP Flowspec, Blackholing)
- Censorship techniques (DNS injection, packet dropping, TCP resets)
- Connectivity disruption (routing, filtering)
- Detection systems (Iris, Augur)
- Case studies (GFW, Egypt, Libya)

**Key Learning Goals**:
- Understand network security properties
- Explain DNS-based attacks
- Understand BGP vulnerabilities and defenses
- Explain DDoS attack mechanisms and mitigation
- Understand censorship techniques and detection

**Prerequisites**: [[02-Network-Layer-and-Routing]] (for BGP understanding)

---

### Appendix-Quick-Reference
**Size**: ~3KB | **Estimated Reading Time**: Quick lookup

**Topics**:
- Protocol quick reference table
- Layer responsibilities summary
- Routing algorithm comparison
- Common formulas and equations
- Port number ranges
- Key acronyms

**Purpose**: Quick lookup during study or review

---

## 🎯 Quick Reference

### Key Concepts by Layer

**Application Layer**
- HTTP, SMTP, DNS, FTP
- CDNs, DASH streaming
- Packet: **Message**

**Transport Layer**
- TCP (reliable, connection-oriented)
- UDP (unreliable, connectionless)
- Flow control, congestion control
- Packet: **Segment**

**Network Layer**
- IP addressing and routing
- OSPF, RIP (intradomain)
- BGP (interdomain)
- Packet: **Datagram**

**Data Link Layer**
- MAC addressing
- Bridges, switches
- Spanning Tree Protocol
- Packet: **Frame**

**Physical Layer**
- Bit transmission
- Physical media
- Packet: **Bits**

### Core Algorithms

- **Dijkstra's Algorithm**: Link State routing (OSPF)
- **Bellman-Ford**: Distance Vector routing (RIP)
- **Spanning Tree**: Loop prevention in Layer 2
- **Longest Prefix Match**: IP forwarding
- **AIMD**: TCP congestion control
- **Deficit Round Robin**: Fair queueing approximation

### Important Protocols

| Protocol | Layer | Purpose |
|----------|-------|---------|
| IP | Network | Addressing and routing |
| TCP | Transport | Reliable delivery |
| UDP | Transport | Fast, unreliable delivery |
| OSPF | Network | Link-state intradomain routing |
| RIP | Network | Distance-vector intradomain routing |
| BGP | Network | Interdomain routing |
| HTTP | Application | Web content transfer |
| DNS | Application | Name resolution |

---

## 💡 Study Tips

### For First-Time Learners
1. **Follow the sequential order** - Each file builds on previous concepts
2. **Draw diagrams** - Network concepts are highly visual
3. **Use packet tracer tools** - Hands-on practice reinforces learning
4. **Focus on "why"** - Understand design decisions, not just mechanisms

### For Review
1. Start with **[[Appendix-Quick-Reference]]** to refresh memory
2. Jump to specific files for deep dives
3. Focus on **Key Learning Goals** sections
4. Review **Concept Map** to see relationships

### For Exam Preparation
1. Can you explain each protocol's purpose?
2. Can you compare alternatives? (OSPF vs RIP, TCP vs UDP)
3. Can you trace a packet through the stack?
4. Can you identify security vulnerabilities?

---

## 🔗 External Resources

- **RFCs**: Official protocol specifications (rfc-editor.org)
- **IETF**: Internet Engineering Task Force (ietf.org)
- **Wireshark**: Packet analysis tool
- **Mininet**: Network emulation for SDN

---

## 📊 Progress Tracking

Track your progress through the course:

- [ ] 01-Fundamentals-and-Architecture
- [ ] 02-Network-Layer-and-Routing
- [ ] 03-Transport-Layer
- [ ] 04-Advanced-Routing-and-QoS
- [ ] 05-Modern-Architectures
- [ ] 06-Application-Layer-Services
- [ ] 07-Security-and-Governance

---

*Last updated: 2026-02-01*
*This study guide reorganized using the Universal Note Organization System*
