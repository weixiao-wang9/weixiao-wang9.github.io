---
type: source
course: "[[Computer Networks (CS 6250)]]"
created: 2026-02-01
prerequisites: "[[02-Network-Layer-and-Routing]]"
---

# Modern Architectures: Software Defined Networking

> **Prerequisites**: [[02-Network-Layer-and-Routing]]
> **Learning Goals**: After reading this, you will understand the SDN paradigm, control/data plane separation, the evolution from Active Networks to OpenFlow, controller architectures (ONOS), P4 programmability, and SDX applications.

## Introduction

Traditional networks are complex, proprietary, and slow to innovate. Routers tightly couple control logic (routing protocols) with forwarding hardware, making it difficult to deploy new protocols or network-wide policies. **Software Defined Networking (SDN)** fundamentally changes this by **separating the control plane from the data plane**, enabling programmable networks managed by centralized (or logically centralized) controllers.

**Key Innovation**: Move intelligence to software controllers, reduce switches to simple forwarding elements controlled via open APIs.

---

## The Problem with Traditional Networks

### Challenges

**1. Complexity**:
- Networks handle diverse equipment: routers, switches, middleboxes (firewalls, NAT, load balancers)
- Each device runs its own protocols (BGP, OSPF, STP)
- Configuration is device-by-device, error-prone

**2. Vendor Lock-in**:
- Proprietary software and closed interfaces
- Cannot mix and match hardware/software from different vendors
- Difficult to innovate (new protocols require vendor support and hardware upgrades)

**3. Slow Innovation**:
- New protocols take years to standardize (IETF process)
- Deployment requires replacing hardware across the network
- Example: IPv6 standardized in 1998, still not fully deployed in 2026

**4. Limited Control**:
- Network operators cannot easily implement custom policies
- Traffic engineering requires manual weight tuning (OSPF costs)
- Security policies scattered across devices

**Example Problem**:
```
Goal: Route video traffic through high-bandwidth links, other traffic through normal links

Traditional approach:
  - Configure every router individually
  - Use complex BGP communities and MPLS tunnels
  - Error-prone, hard to verify

Result: Hours/days to deploy, difficult to troubleshoot
```

---

## The SDN Solution

### Core Principle: Separation of Concerns

**Traditional Router**: Control and Data Plane tightly coupled in one box

**SDN Approach**: Separate the planes
- **Control Plane**: Centralized controller (software) computes routes
- **Data Plane**: Distributed switches (hardware) forward packets

**Benefits**:
1. **Centralized Logic**: Network-wide view enables global optimization
2. **Programmability**: Write software to control network behavior
3. **Innovation**: Deploy new protocols without hardware changes
4. **Vendor Neutrality**: Open interfaces (OpenFlow) allow multi-vendor networks

**Analogy**:
```
Traditional Network = Each car has its own GPS and decides route independently
SDN Network = Central traffic control system directs all cars (network-wide optimization)
```

---

## History and Evolution of SDN

### 1. Active Networks (Mid-1990s to Early 2000s)

**Goal**: Make networks programmable by allowing users to inject code

**Two Approaches**:

**Capsule Model**:
- Packets carry code ("capsules")
- Routers execute code at each hop
- Example: Packet contains code to compress video on-the-fly

**Programmable Router Model**:
- Routers have programmable interfaces
- Users download programs to routers (like installing apps)
- Programs process packets passing through

**Vision**: Accelerate protocol deployment by allowing experimentation

**Why It Failed**:
- **Security**: Arbitrary code execution is dangerous
- **Performance**: Software packet processing too slow
- **Complexity**: Managing code across distributed routers
- **No killer app**: Unclear value proposition

**Legacy**: Inspired SDN's programmability concept, but SDN learned to separate control from data plane

---

### 2. Control/Data Plane Separation (2001-2007)

**Motivation**: Improve network reliability and manageability

**Key Projects**:

**ForCES (Forwarding and Control Element Separation)**:
- Standardized interface between control and forwarding elements
- Open interface (unlike proprietary router internals)

**RCP (Routing Control Platform)**:
- Centralized BGP route computation
- Better control over interdomain routing decisions

**Ethane (Precursor to OpenFlow)**:
- Centralized flow-based access control
- Controller decides which flows are allowed
- Switches enforce decisions

**Insight**: Logically centralized control simplifies management and enables network-wide policies

---

### 3. OpenFlow and Modern SDN (2007-Present)

**Catalyst**: Need for network experimentation in research networks

**Problem**: Production networks cannot experiment with new protocols (risk downtime)

**Solution**: **Network Slicing** - Run experimental protocols alongside production traffic

**OpenFlow (2007)**:
- Standardized API between controller and switch
- Exposes **flow tables** in commodity switches
- Switches match packets on multiple header fields, execute actions

**Key Innovation**: Commodity Ethernet switches already had flow tables (for VLANs, ACLs) - OpenFlow just opened access to them

**Impact**:
- Google deployed OpenFlow in their WAN (B4 network) - 100% SDN by 2012
- Enabled network virtualization (Nicira/VMware NSX)
- Led to programmable data centers (Microsoft, Facebook)

---

## SDN Architecture

### The Layered Model

**SDN Stack** (Bottom to Top):

**1. Infrastructure Layer (Data Plane)**:
- **Components**: Switches, routers (forwarding elements)
- **Function**: Forward packets based on flow tables
- **Key Property**: "Dumb" forwarding devices (no routing protocols)

**2. Southbound Interface (Control-Data Plane API)**:
- **Protocol**: OpenFlow (most common), P4Runtime, NETCONF
- **Function**: Controller programs flow tables in switches
- **Messages**:
  - Controller → Switch: Install/modify/delete flow entries
  - Switch → Controller: Packet-in (for unknown flows), statistics

**3. Network Operating System (Control Plane)**:
- **Components**: SDN Controller (e.g., ONOS, OpenDayLight, Floodlight)
- **Function**: Maintains network topology, computes routes, provides abstractions
- **Key Service**: Provides northbound API for applications

**4. Northbound Interface (Application API)**:
- **Protocol**: REST API, Python/Java APIs
- **Function**: Applications express intent, controller translates to flow rules

**5. Application Layer**:
- **Examples**: Routing, Load balancing, Firewall, Traffic engineering
- **Function**: Implements network logic (what the network should do)

**Diagram**:
```
+--------------------+
|   Applications     |  (Routing, Firewall, Load Balancer)
+--------------------+
         ↕ Northbound API (REST)
+--------------------+
|   Controller       |  (ONOS, OpenDayLight)
|  (Network OS)      |
+--------------------+
         ↕ Southbound API (OpenFlow)
+--------------------+
|    Switches        |  (Forwarding only)
+--------------------+
```

---

### Flow-Based Forwarding

**Traditional Routing**: Match on destination IP → forward to next hop

**SDN Flow Forwarding**: Match on **any combination of header fields** → execute **actions**

**Flow Table Entry**:
```
Match Fields           | Priority | Actions        | Counters
-----------------------|----------|----------------|----------
src=10.0.0.1, dst=*    |   100    | Forward port 3 | 5000 pkts
dst=192.168.1.0/24     |    50    | Forward port 1 | 10000 pkts
*                      |     1    | Drop           | 200 pkts
```

**Match Fields** (12+ fields in OpenFlow 1.0):
- Ingress port
- Ethernet: src/dst MAC, type
- VLAN: ID, priority
- IP: src/dst, protocol, ToS
- TCP/UDP: src/dst port

**Actions**:
- **Forward**: Send to specific port(s)
- **Drop**: Discard packet
- **Modify**: Change header fields (e.g., rewrite destination IP for NAT)
- **Send to controller**: For unknown flows (packet-in message)

**Matching Process**:
1. Packet arrives at switch
2. Match against flow table (highest priority first)
3. If match found: Execute actions
4. If no match: Send to controller (or drop, depending on config)

---

### OpenFlow Protocol

**Controller-to-Switch Messages**:

**1. Flow Mod (Modify Flow Table)**:
- Install new flow entry
- Modify existing entry
- Delete entry (e.g., when route changes)

**2. Stats Request**:
- Query switch for statistics (bytes/packets per flow)

**3. Packet Out**:
- Controller sends packet to switch for forwarding
- Used when controller makes forwarding decision

**Switch-to-Controller Messages**:

**1. Packet In**:
- Switch sends packet to controller when no matching flow entry
- Controller decides what to do (install flow rule or drop)

**2. Flow Removed**:
- Notify controller when flow entry expires or is deleted

**3. Port Status**:
- Notify controller of link up/down events

**Example Flow**:
```
1. New flow arrives at switch (src=10.0.0.1, dst=10.0.0.2)
2. No matching flow entry → Switch sends Packet-In to controller
3. Controller computes path: Switch A port 3 → Switch B port 2 → dst
4. Controller sends Flow-Mod to Switch A: "Match src=10.0.0.1, dst=10.0.0.2 → Forward port 3"
5. Future packets in this flow forwarded directly (no controller involvement)
```

**Granularity**: Flow entries can be specific (per-connection) or aggregate (per-prefix)

---

## SDN Controllers

### Centralized vs. Distributed Controllers

**Centralized Controller** (Single instance):

**Examples**: POX, Floodlight, Ryu

**Advantages**:
- Simple to program (single view of network)
- Consistent decisions

**Disadvantages**:
- **Single point of failure**: Controller down = network down
- **Scalability**: Limited by one machine's CPU/memory
- **Performance bottleneck**: All switches depend on one controller

**Use Case**: Small networks, research, prototyping

---

**Distributed Controller** (Multiple instances in cluster):

**Examples**: ONOS, OpenDayLight

**Goal**: Scalability and fault tolerance

**Architecture**:
- Multiple controller instances form a **cluster**
- Controllers share network state via distributed database
- Each switch connects to one or more controllers

**Challenges**:
1. **State Consistency**: How to keep controllers' views synchronized?
2. **Fault Tolerance**: How to handle controller failures?
3. **Scalability**: How to distribute load across controllers?

---

### ONOS (Open Networking Operating System)

**Design Philosophy**: **Distributed, Scalable, Fault-Tolerant**

**Architecture**:

**1. Controller Cluster**:
- Multiple ONOS instances running on different servers
- Each instance has full network view (eventually consistent)

**2. Mastership Election**:
- For each switch, one controller instance is the **master** (others are backup)
- Master handles all control messages from that switch
- If master fails, another instance takes over

**3. Global Network View**:
- Distributed data store (based on Hazelcast or Atomix)
- Stores:
  - Topology: Switches, links, ports
  - Hosts: Connected end-devices
  - Flow rules: Installed across switches
- **Eventual consistency**: Updates propagate across cluster

**Example**:
```
Cluster: ONOS-1, ONOS-2, ONOS-3

Switch A: Master = ONOS-1, Backup = ONOS-2, ONOS-3
Switch B: Master = ONOS-2, Backup = ONOS-1, ONOS-3

Link A-B fails:
  → Switch A notifies ONOS-1 (master)
  → ONOS-1 updates distributed store
  → ONOS-2, ONOS-3 receive update
  → All have consistent view within milliseconds

ONOS-1 fails:
  → ONOS-2 detects failure (via heartbeat)
  → ONOS-2 becomes new master for Switch A
  → Applications continue running on ONOS-2
```

**Benefits**:
- **High availability**: Controller failure doesn't bring down network
- **Load distribution**: Different switches handled by different masters
- **Scalability**: Add more controller instances as network grows

---

### Controller Services

**Common Services Provided by SDN Controllers**:

**1. Topology Service**:
- Discovers switches, links, hosts
- Maintains graph of network topology
- Notifies applications of topology changes

**2. Path Computation**:
- Computes shortest paths (Dijkstra)
- Supports constraints (e.g., avoid certain links)

**3. Flow Rule Management**:
- Translates high-level intents into flow rules
- Installs/updates flow rules across switches

**4. Device Management**:
- Monitors switch status (up/down)
- Handles switch connections (via OpenFlow)

**5. Statistics Collection**:
- Polls switches for traffic statistics
- Aggregates data for monitoring/analytics

**Application Example**:
```python
# Pseudo-code for L2 forwarding app

def packet_in_handler(event):
    packet = event.packet
    switch = event.switch
    in_port = event.in_port

    # Learn source MAC → port mapping
    mac_table[packet.src_mac] = (switch, in_port)

    # Lookup destination
    if packet.dst_mac in mac_table:
        out_switch, out_port = mac_table[packet.dst_mac]
        if out_switch == switch:
            # Install flow rule
            install_flow(switch, match={dst_mac: packet.dst_mac}, action={output: out_port})
        else:
            # Compute path and install rules
            path = compute_path(switch, out_switch)
            install_path(path, packet.dst_mac)
    else:
        # Flood
        flood(switch, packet, in_port)
```

---

## Programming the Data Plane: P4

### Motivation

**Problem with OpenFlow**: Fixed match fields

**Example**:
- OpenFlow 1.0: 12 match fields (Ethernet, IP, TCP/UDP)
- Want to match on new protocol (e.g., VXLAN, MPLS, custom header)?
- **Solution before P4**: Update OpenFlow spec, wait for vendors to support, upgrade switches

**Limitation**: Innovation bottlenecked by standardization process

**P4 Solution**: Make the data plane itself **programmable**

---

### P4 Overview

**Name**: **P**rogramming **P**rotocol-independent **P**acket **P**rocessors

**Goal**: Allow operators to define:
1. **What headers** switches should recognize
2. **How to parse** those headers
3. **How to process** (match-action) packets

**Key Properties**:

**1. Reconfigurability**:
- Change packet processing without hardware redesign
- Example: Add support for new tunnel protocol via software update

**2. Protocol Independence**:
- Not tied to specific protocols (IPv4, Ethernet, etc.)
- Switches become "white boxes" that can process any protocol

**3. Target Independence**:
- Same P4 program can run on different hardware (software switches, FPGAs, ASICs)
- Compiler maps P4 to target-specific instructions

---

### P4 Programming Model

**Two Main Components**:

**1. Parser**:
- Defines how to extract header fields from packets
- State machine that transitions based on packet contents

**Example** (simplified):
```p4
parser start {
    extract(ethernet);
    return select(ethernet.etherType) {
        0x0800: parse_ipv4;
        0x86DD: parse_ipv6;
        default: ingress;
    }
}

parser parse_ipv4 {
    extract(ipv4);
    return ingress;
}
```

**2. Match-Action Tables**:
- Define what to match on and what actions to execute
- Similar to OpenFlow flow tables but fully customizable

**Example**:
```p4
table ipv4_forwarding {
    reads {
        ipv4.dstAddr : lpm;  // Longest prefix match
    }
    actions {
        forward;
        drop;
    }
}

action forward(port) {
    modify_field(standard_metadata.egress_spec, port);
    modify_field(ipv4.ttl, ipv4.ttl - 1);  // Decrement TTL
}
```

**Control Flow**:
```p4
control ingress {
    apply(ipv4_forwarding);
}
```

---

### P4 Use Cases

**1. Custom Protocols**:
- Data centers with proprietary protocols
- Example: Facebook's custom load balancing headers

**2. In-Network Computing**:
- Switches perform computations (not just forwarding)
- Example: Aggregate statistics, consensus protocols

**3. Network Telemetry**:
- Switches add metadata to packets (e.g., queue depth, latency)
- Example: INT (In-band Network Telemetry)

**4. Rapid Prototyping**:
- Test new protocols without hardware changes
- Deploy updates in minutes (vs. months for hardware)

**Example: INT (In-band Network Telemetry)**:
```p4
// Add switch metadata to packet
action add_int_metadata() {
    push(int_stack, 1);  // Add metadata header
    modify_field(int_stack[0].switch_id, switch_id);
    modify_field(int_stack[0].queue_depth, queue_depth);
    modify_field(int_stack[0].timestamp, timestamp);
}
```

**Result**: Packets carry detailed path information for debugging

---

## SDN Applications

### SDX (Software Defined Internet Exchange)

**Problem**: BGP limitations at IXPs (Internet Exchange Points)

**IXP Reminder**:
- Physical location where multiple networks (ASes) connect
- Exchange traffic directly (peer) to avoid transit costs
- Shared Layer 2 fabric (Ethernet switch)

**BGP Limitations**:
1. **Destination-only routing**: Can only route based on destination prefix
2. **No application awareness**: Cannot route video differently from email
3. **No source-based routing**: Cannot prefer certain peers for specific traffic
4. **Coarse granularity**: Route entire prefixes, not specific flows

**Example Problem**:
```
AS 100 at IXP wants:
  - Route video traffic (port 443, Netflix) via Peer A (high bandwidth)
  - Route other traffic via Peer B (cheaper)

BGP cannot do this: Only destination prefix matching
```

---

### SDX Architecture

**Goal**: Give each IXP participant the **illusion of their own virtual SDN switch**

**How It Works**:

**1. Virtual Switch Abstraction**:
- Each AS thinks it has a dedicated switch with ports to other ASes
- AS defines custom forwarding rules (match on any field)

**2. SDX Controller**:
- Collects policies from all ASes
- Compiles policies into a single set of flow rules
- Installs rules on the physical IXP fabric (real switches)

**3. Policy Composition**:
- SDX merges overlapping/conflicting policies
- Ensures isolation (one AS's policy doesn't affect others)

**Example**:
```
AS 100 policy:
  match: dst=192.168.0.0/16, app=video → forward to AS 200 (Peer A)
  match: dst=192.168.0.0/16, app=other → forward to AS 300 (Peer B)

AS 200 policy:
  match: src=AS 100, dst=10.0.0.0/8 → forward to AS 400

SDX Controller:
  Compiles both policies into flow rules on physical switches
  Installs rules that satisfy both ASes' intents
```

**Benefits**:
1. **Application-aware routing**: Route based on ports, protocols
2. **Traffic engineering**: Fine-grained control over traffic paths
3. **Flexibility**: Change policies in seconds (vs. hours with BGP)
4. **Transparency**: Each AS controls its own policies independently

**Deployment**: Several IXPs (e.g., AMS-IX research testbed) have deployed SDX

---

## Summary

### Key Takeaways

1. **SDN Paradigm**:
   - Separates control plane (software) from data plane (hardware)
   - Centralized/logically centralized control enables network-wide optimization
   - Open interfaces (OpenFlow) break vendor lock-in

2. **Evolution**:
   - **Active Networks** (1990s): Programmability via code injection (failed due to security/performance)
   - **Control/Data Separation** (2000s): Centralized control for better management
   - **OpenFlow/SDN** (2007+): Standardized API, flow-based forwarding, practical deployment

3. **SDN Architecture**:
   - **Layers**: Applications → Northbound API → Controller → Southbound API → Switches
   - **Flow-based forwarding**: Match on multiple fields, execute actions
   - **Controllers**: Centralized (simple) vs. Distributed (scalable, fault-tolerant like ONOS)

4. **P4 Programming**:
   - Makes data plane programmable (define parsers and match-action tables)
   - Protocol-independent, target-independent
   - Enables rapid innovation (custom protocols, in-network computing)

5. **SDX Application**:
   - Applies SDN to IXPs
   - Overcomes BGP limitations (destination-only routing)
   - Application-aware, fine-grained traffic engineering

### Common Patterns

**SDN Design Principles**:
- **Separation of concerns**: Control vs. data plane
- **Centralization**: Global view enables optimization
- **Programmability**: Software-defined behavior

**Trade-offs**:
- **Centralized controllers**: Simple but single point of failure
- **Distributed controllers**: Scalable but complex (consistency challenges)
- **Flow granularity**: Per-flow (fine control) vs. aggregated (scalability)

**Application Development**:
- Applications express intent (what to do)
- Controller translates to flow rules (how to do it)
- Switches execute rules (fast path in hardware)

---

## See Also

- [[02-Network-Layer-and-Routing]] - Traditional routing (BGP, OSPF) vs. SDN
- [[04-Advanced-Routing-and-QoS]] - QoS mechanisms controllable via SDN
- [[07-Security-and-Governance]] - SDN security challenges and solutions

**Next**: [[06-Application-Layer-Services]]
