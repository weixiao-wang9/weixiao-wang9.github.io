---
id: 202512170319
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - networks
  - technique
related_topics: "[[Concept - Internet Exchange Points (IXPs)]]"
created: 2025-12-17
---

# Remote Peering

## 💡 The Core Idea
Remote Peering allows an AS to connect to an IXP without having a physical router at that facility. They essentially "rent" a long-distance cable (or reseller port) to appear as if they are local.

## 🧠 Implications
* **Cost:** Cheaper for smaller networks who can't afford global infrastructure.
* **Detection:** Hard to detect. Researchers use RTT (Round Trip Time) measurements to infer if a "local" peer is actually distant, but this is prone to errors.

## 🔗 Connections
- **Source:** [[Source - Autonomous Systems and Internet Interconnection]]