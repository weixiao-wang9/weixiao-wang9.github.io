---
id: 202512170318
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - networks
  - infrastructure
related_topics: "[[Concept - Internet Ecosystem Hierarchy]]"
created: 2025-12-17
---

# Internet Exchange Points (IXPs)

## 💡 The Core Idea
An IXP is a physical infrastructure (often a switch in a data center) where multiple networks meet to exchange traffic directly, bypassing upstream providers.



## 🧠 Evolution & Peering
* **Bilateral Peering:** Two networks run a direct cable and BGP session. Hard to scale.
* **Multilateral Peering (Route Servers):** Networks connect to a central **Route Server (RS)**. The RS collects routes and redistributes them, allowing an AS to peer with 50+ networks via a single BGP session.

### Benefits
* Keeps local traffic local (lower latency).
* Reduces costs (less traffic sent to paid transit providers).
* Can provide DDoS mitigation (Blackholing).

## 🔗 Connections
- **Source:** [[Source - Autonomous Systems and Internet Interconnection]]