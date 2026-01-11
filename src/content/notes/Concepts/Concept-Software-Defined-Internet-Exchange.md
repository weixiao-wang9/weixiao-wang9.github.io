---
id: 202512170417
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - sdn
  - application
  - bgp
related_topics: "[[Concept - Internet Exchange Points (IXPs)]]"
created: 2025-12-17
---

# Software Defined Internet Exchange (SDX)

## 💡 The Core Idea
SDX applies SDN principles to Internet Exchange Points (IXPs) to fix the limitations of BGP. It allows routing decisions based on **Application** (e.g., Netflix vs. Email) or **Source**, rather than just Destination IP.



## 🧠 Mechanism
* **The Illusion:** SDX gives each participant AS the illusion of having its own **Virtual SDN Switch** connecting to all other participants.
* **Policies:** ASes can write custom policies (using languages like Pyretic) to direct traffic.

## 🚀 Use Cases
1.  **Application Specific Peering:** Route "High-Bandwidth Video" differently than "VoIP".
2.  **Inbound Traffic Engineering:** Controlling how traffic *enters* the network (difficult in traditional BGP).
3.  **Wide-Area Load Balancing:** Redirecting requests to different backend servers by modifying packet headers at the exchange.

## 🔗 Connections
- **Source:** [[Source - SDN Part 2 and Applications]]