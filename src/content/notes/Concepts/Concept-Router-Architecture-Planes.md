---
id: 202512170328
type: atom
course: "[[Computer Networks (CS401)]]"
tags: [concept, networks, architecture]
related_topics: [[Concept - Switching Fabrics]]
created: 2025-12-17
---

# Router Architecture: Control vs. Forwarding Plane

## 💡 The Core Idea
A router is split into two distinct functional planes. The **Control Plane** handles intelligence and policy (slow, software), while the **Forwarding Plane** handles data movement (fast, hardware).



## 🧠 The Separation

| Feature | Control Plane | Forwarding Plane |
| :--- | :--- | :--- |
| **Function** | "Routing" (Planning the map) | "Switching" (Driving the car) |
| **Task** | Runs OSPF/BGP, builds Routing Tables. | Looks up destination IP, moves packet to output. |
| **Implementation** | Software (CPU) | Hardware (ASICs/FPGA) |
| **Time Scale** | Seconds (updates) | Nanoseconds (per packet) |

## 🔗 Connections
- **Source:** [[Source - Router Architecture and Lookups]]