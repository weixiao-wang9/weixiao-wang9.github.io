---
id: 202512170422
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - security
  - mitigation
  - bgp
related_topics: "[[Concept - BGP Flowspec]]"
created: 2025-12-17
---

# BGP Blackholing

## 💡 The Core Idea
A crude but effective DDoS mitigation strategy where all traffic destined for a victim IP is dropped (sent to a null interface) upstream, before it can overwhelm the victim's network link.

## 🧠 Mechanism
* **Signal:** The victim AS sends a BGP update tagged with a specific **Blackhole Community**.
* **Action:** The upstream Provider or IXP Route Server sees the tag and rewrites the next-hop to a null interface.
* **Result:** Traffic is dropped closer to the source.

## ⚠️ Limitations
* **Collateral Damage:** It drops **all** traffic to the victim IP, including legitimate user traffic. The service becomes unreachable, effectively completing the DDoS attack's goal, but saving the rest of the network infrastructure.

## 🔗 Connections
- **Source:** [[Source - Internet Security]]