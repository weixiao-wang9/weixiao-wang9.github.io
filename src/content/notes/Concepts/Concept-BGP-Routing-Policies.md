---
id: 202512170318
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - networks
  - strategy
related_topics: "[[Concept - BGP Protocol Basics]]"
created: 2025-12-17
---

# BGP Routing Policies & Attributes

## 💡 The Core Idea
BGP routing is driven by **money and policy**, not just technical efficiency. ASes select routes to maximize revenue and control traffic flow using specific attributes.



## 🧠 Decision Process
### 1. Import Ranking (The Money Rule)
When a router hears about a destination from multiple neighbors, it prefers:
1.  **Customer Routes:** (Generates revenue).
2.  **Peer Routes:** (Free).
3.  **Provider Routes:** (Costs money).

### 2. Traffic Control Attributes
* **LocalPref (Local Preference):** Used to control **Outbound** traffic.
    * *Example:* "Prefer sending traffic through Provider A because it's cheaper than Provider B".
* **MED (Multi-Exit Discriminator):** Used to control **Inbound** traffic.
    * *Example:* "Tell Neighbor X to enter my network at Router 1 instead of Router 2".

### 3. The "Gao-Rexford" Export Rule
* **Valley-Free Routing:** An AS will never export a route learned from a Peer/Provider to another Peer/Provider. This prevents an AS from becoming a free transit hub for others.

## 🔗 Connections
- **Source:** [[Source - Autonomous Systems and Internet Interconnection]]