---
type: reference
course: "[[Computer Networks (CS 6250)]]"
created: 2026-02-01
---

# Quick Reference Guide

Quick lookup for key concepts, protocols, algorithms, and formulas in computer networking.

---

## Protocol Summary Table

| Protocol | Layer | Port | Purpose | Connection | Reliability |
|----------|-------|------|---------|------------|-------------|
| **HTTP** | Application | 80 | Web content transfer | Connectionless (over TCP) | Yes (via TCP) |
| **HTTPS** | Application | 443 | Secure web | Connectionless (over TCP) | Yes (via TCP) |
| **SMTP** | Application | 25 | Email sending | Connection-oriented (over TCP) | Yes (via TCP) |
| **DNS** | Application | 53 | Name resolution | Connectionless (over UDP) | No |
| **FTP** | Application | 20, 21 | File transfer | Connection-oriented (over TCP) | Yes (via TCP) |
| **SSH** | Application | 22 | Secure remote access | Connection-oriented (over TCP) | Yes (via TCP) |
| **TCP** | Transport | - | Reliable transport | Connection-oriented | Yes |
| **UDP** | Transport | - | Fast transport | Connectionless | No |
| **IP** | Network | - | Host addressing and routing | Connectionless | No |
| **OSPF** | Network | - | Link-state intradomain routing | - | - |
| **RIP** | Network | - | Distance-vector intradomain routing | - | - |
| **BGP** | Network | 179 | Interdomain routing | Connection-oriented (over TCP) | Policy-based |
| **ARP** | Link | - | MAC address resolution | Connectionless | No |
| **Ethernet** | Link | - | LAN communication | Connectionless | No |

---

## Layer Responsibilities

| Layer | Responsibility | Data Unit | Key Protocols | Addressing |
|-------|----------------|-----------|---------------|------------|
| **5. Application** | Application-specific services | **Message** | HTTP, SMTP, DNS, FTP | Application-defined |
| **4. Transport** | Process-to-process communication | **Segment** | TCP, UDP | Port numbers (16-bit) |
| **3. Network** | Host-to-host routing | **Datagram/Packet** | IP, OSPF, RIP, BGP | IP addresses (32-bit IPv4, 128-bit IPv6) |
| **2. Data Link** | Hop-to-hop transmission | **Frame** | Ethernet, Wi-Fi, PPP | MAC addresses (48-bit) |
| **1. Physical** | Bit transmission | **Bits** | Varies by medium | Physical ports |

---

## Port Number Ranges

| Range | Type | Purpose | Examples |
|-------|------|---------|----------|
| **0-1023** | Well-Known | Standard services | HTTP (80), HTTPS (443), SSH (22), DNS (53), SMTP (25) |
| **1024-49151** | Registered | Application-specific | Custom applications |
| **49152-65535** | Ephemeral/Dynamic | Temporary client ports | Assigned by OS for outgoing connections |

---

## Routing Algorithm Comparison

| Algorithm | Type | Knowledge | Update Trigger | Convergence | Loops | Complexity | Protocols |
|-----------|------|-----------|----------------|-------------|-------|------------|-----------|
| **Dijkstra** | Link State | Global topology | Link state change | Fast (seconds) | No | O(n²) or O(n log n) | OSPF |
| **Bellman-Ford** | Distance Vector | Neighbors only | Periodic (30s) or triggered | Slow (minutes) | Possible (count-to-infinity) | O(n × neighbors) | RIP |
| **Path Vector** | Hybrid | AS path | Policy changes | Variable | No (due to path info) | - | BGP |

---

## BGP Route Selection Process

**Order of Decision** (first match wins):

1. **Highest LocalPref** (Local Preference) - Operator-defined preference
2. **Shortest AS Path** - Fewest AS hops
3. **Lowest Origin Type** - IGP < EGP < Incomplete
4. **Lowest MED** (Multi-Exit Discriminator) - Preferred entry point (same AS only)
5. **eBGP over iBGP** - Prefer external routes
6. **Lowest IGP Cost to Next Hop** - Hot potato routing
7. **Lowest Router ID** - Tiebreaker

**Business Relationship Priority**: Customer > Peer > Provider

---

## TCP States

| State | Description |
|-------|-------------|
| **CLOSED** | No connection |
| **LISTEN** | Server waiting for connection |
| **SYN-SENT** | Client sent SYN, waiting for SYNACK |
| **SYN-RECEIVED** | Server received SYN, sent SYNACK, waiting for ACK |
| **ESTABLISHED** | Connection active, data transfer |
| **FIN-WAIT-1** | Sent FIN, waiting for ACK |
| **FIN-WAIT-2** | Received ACK of FIN, waiting for peer's FIN |
| **CLOSE-WAIT** | Received FIN, waiting for application to close |
| **CLOSING** | Both sides sent FIN simultaneously |
| **LAST-ACK** | Sent FIN in response, waiting for ACK |
| **TIME-WAIT** | Waiting to ensure remote received ACK of FIN |

---

## TCP Congestion Control Summary

| Phase | Trigger | cwnd Adjustment | Growth Pattern |
|-------|---------|-----------------|----------------|
| **Slow Start** | Connection start or timeout | Double per RTT (×2) | Exponential |
| **Congestion Avoidance (AIMD)** | cwnd ≥ ssthresh | +1 MSS per RTT | Linear (additive) |
| **Fast Recovery** | 3 Duplicate ACKs | cwnd = ssthresh = cwnd/2 | Halve (multiplicative decrease) |
| **Timeout Recovery** | Timeout | cwnd = 1, ssthresh = cwnd/2 | Restart slow start |

**TCP CUBIC**: Uses cubic function instead of linear AIMD, RTT-independent

---

## QoS Scheduling Algorithms

| Algorithm | Time Complexity | Fairness | Description |
|-----------|----------------|----------|-------------|
| **FIFO** | O(1) | No | First In First Out, simple but unfair |
| **Priority Queuing** | O(1) | No | High-priority always served first |
| **Fair Queuing (Bit-by-Bit)** | O(log n) | Perfect | Theoretical ideal, simulates bit-by-bit service |
| **Weighted Fair Queuing (WFQ)** | O(log n) | Weighted | Assigns weights to flows |
| **Deficit Round Robin (DRR)** | O(1) | Approximate | Constant-time approximation of fair queuing |

---

## Traffic Shaping Mechanisms

| Mechanism | Behavior | Use Case | Parameters |
|-----------|----------|----------|------------|
| **Token Bucket** | Allows bursts, enforces average rate | Bursty traffic with rate limit | Bucket size (B), Token rate (R) |
| **Leaky Bucket** | Smooths traffic to constant rate | Constant-rate output | Bucket size, Output rate |

**Formula**: Token Bucket allows burst of size B, average rate R tokens/sec

---

## Packet Classification Algorithms

| Algorithm | Memory Usage | Lookup Time | Description |
|-----------|--------------|-------------|-------------|
| **Linear Search** | Low | O(n) | Check each rule sequentially |
| **Caching** | Medium | Variable | Cache recent matches |
| **Set-Pruning Tries** | Very High | O(W) | Destination trie with source tries at leaves |
| **Backtracking** | Medium | Variable | Points to source tries, backtracks on miss |
| **Grid of Tries** | High | O(W) | Switch pointers eliminate backtracking |

W = Address width (32 bits for IPv4)

---

## Common Formulas

### Dijkstra's Algorithm
```
distance[u] = minimum distance from source to u
For each unvisited neighbor v of u:
    if distance[u] + cost(u,v) < distance[v]:
        distance[v] = distance[u] + cost(u,v)
```

### Bellman-Ford Equation
```
D_x(y) = min over all neighbors v { cost(x,v) + D_v(y) }

D_x(y) = Distance from router x to destination y
cost(x,v) = Link cost from x to neighbor v
D_v(y) = Neighbor v's distance to y
```

### TCP Timeout Estimation
```
EstimatedRTT = (1 - α) × EstimatedRTT + α × SampleRTT
DevRTT = (1 - β) × DevRTT + β × |SampleRTT - EstimatedRTT|
TimeoutInterval = EstimatedRTT + 4 × DevRTT

Typical: α = 0.125, β = 0.25
```

### Token Bucket
```
Tokens in bucket: min(B, current_tokens + R × time_elapsed)
Packet transmitted if: packet_size ≤ current_tokens
After transmission: current_tokens -= packet_size

B = Bucket capacity (bytes)
R = Token rate (bytes/second)
```

---

## Key Acronyms

| Acronym | Full Name | Context |
|---------|-----------|---------|
| **ACK** | Acknowledgment | TCP, reliable delivery |
| **AIP** | Accountable Internet Protocol | Clean-slate architecture |
| **AIMD** | Additive Increase Multiplicative Decrease | TCP congestion control |
| **ARP** | Address Resolution Protocol | MAC address lookup |
| **AS** | Autonomous System | BGP, interdomain routing |
| **ASN** | Autonomous System Number | AS identifier |
| **BGP** | Border Gateway Protocol | Interdomain routing |
| **CDN** | Content Delivery Network | Distributed content servers |
| **CIDR** | Classless Inter-Domain Routing | IP address aggregation |
| **DASH** | Dynamic Adaptive Streaming over HTTP | Video streaming |
| **DDoS** | Distributed Denial of Service | Security attack |
| **DNS** | Domain Name System | Name to IP resolution |
| **DRR** | Deficit Round Robin | Fair queuing algorithm |
| **eBGP** | External BGP | Between different ASes |
| **FIB** | Forwarding Information Base | Router forwarding table |
| **FIFO** | First In First Out | Simple queuing |
| **FTP** | File Transfer Protocol | File transfer |
| **GFW** | Great Firewall | China's censorship system |
| **HOL** | Head-of-Line (Blocking) | Switching fabric issue |
| **HTTP** | Hypertext Transfer Protocol | Web |
| **iBGP** | Internal BGP | Within same AS |
| **ICMP** | Internet Control Message Protocol | Error reporting (ping, traceroute) |
| **IGP** | Interior Gateway Protocol | Intradomain routing |
| **IP** | Internet Protocol | Network layer |
| **IXP** | Internet Exchange Point | Peering location |
| **LPM** | Longest Prefix Match | IP forwarding |
| **LSA** | Link State Advertisement | OSPF topology update |
| **MAC** | Media Access Control | Layer 2 addressing |
| **MED** | Multi-Exit Discriminator | BGP attribute |
| **MSS** | Maximum Segment Size | TCP segment size |
| **NAT** | Network Address Translation | IP address translation |
| **ONOS** | Open Networking Operating System | Distributed SDN controller |
| **OSPF** | Open Shortest Path First | Link-state intradomain routing |
| **P4** | Programming Protocol-independent Packet Processors | SDN data plane language |
| **QoS** | Quality of Service | Traffic prioritization |
| **RIP** | Routing Information Protocol | Distance-vector intradomain routing |
| **RTT** | Round-Trip Time | Time for packet to go and return |
| **SDN** | Software-Defined Networking | Control/data plane separation |
| **SDX** | Software-Defined Exchange | SDN at IXPs |
| **STP** | Spanning Tree Protocol | Layer 2 loop prevention |
| **TCP** | Transmission Control Protocol | Reliable transport |
| **TTL** | Time To Live | IP header field |
| **UDP** | User Datagram Protocol | Unreliable transport |
| **VoIP** | Voice over IP | Real-time voice |
| **WFQ** | Weighted Fair Queuing | QoS scheduling |

---

## IP Address Classes (Historical)

| Class | First Bits | Range | Default Mask | Purpose |
|-------|------------|-------|--------------|---------|
| **A** | 0 | 0.0.0.0 - 127.255.255.255 | /8 | Large networks |
| **B** | 10 | 128.0.0.0 - 191.255.255.255 | /16 | Medium networks |
| **C** | 110 | 192.0.0.0 - 223.255.255.255 | /24 | Small networks |
| **D** | 1110 | 224.0.0.0 - 239.255.255.255 | - | Multicast |
| **E** | 1111 | 240.0.0.0 - 255.255.255.255 | - | Reserved |

**Note**: Modern Internet uses CIDR, not classes

---

## Private IP Address Ranges (RFC 1918)

| Range | CIDR | Number of Addresses |
|-------|------|---------------------|
| 10.0.0.0 - 10.255.255.255 | 10.0.0.0/8 | 16,777,216 |
| 172.16.0.0 - 172.31.255.255 | 172.16.0.0/12 | 1,048,576 |
| 192.168.0.0 - 192.168.255.255 | 192.168.0.0/16 | 65,536 |

---

## Switching Fabric Types

| Type | Mechanism | Speed | Blocking | Cost |
|------|-----------|-------|----------|------|
| **Memory-based** | CPU copies packets via memory | Slowest (2× bus bandwidth) | Yes | Lowest |
| **Bus-based** | Shared bus | Medium (1× bus bandwidth) | Yes | Medium |
| **Crossbar** | Parallel paths | Fastest (N× port speed) | Only HOL | Highest |

---

## Security Properties (CIAA)

| Property | Definition | Mechanisms |
|----------|------------|------------|
| **Confidentiality** | Only intended parties can read | Encryption (AES, RSA) |
| **Integrity** | Message not modified in transit | Hash functions (SHA), MACs |
| **Authentication** | Verify identity of parties | Digital signatures, certificates |
| **Availability** | Service remains accessible | Redundancy, DDoS mitigation |

---

## Common Attack Types

| Attack | Layer | Description | Mitigation |
|--------|-------|-------------|------------|
| **IP Spoofing** | Network | Fake source IP | Ingress filtering, source verification |
| **BGP Hijacking** | Network | Announce false routes | ARTEMIS, RPKI |
| **DDoS** | Network/Transport | Overwhelm with traffic | Rate limiting, Blackholing, Flowspec |
| **DNS Injection** | Application | Fake DNS responses | DNSSEC |
| **TCP SYN Flood** | Transport | Exhaust connection resources | SYN cookies |

---

## See Also

- [[00-README]] - Main study guide and learning path
- [[01-Fundamentals-and-Architecture]] - Foundation concepts
- [[02-Network-Layer-and-Routing]] - Routing details
- [[03-Transport-Layer]] - TCP/UDP mechanisms
- [[04-Advanced-Routing-and-QoS]] - Advanced topics
- [[05-Modern-Architectures]] - SDN and modern designs
- [[06-Application-Layer-Services]] - CDN and multimedia
- [[07-Security-and-Governance]] - Security mechanisms

---

*Use this reference for quick lookups during study or review. For detailed explanations, refer to the main topic files.*
