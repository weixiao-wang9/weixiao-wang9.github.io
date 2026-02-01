---
id: 202512170421
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - security
  - bgp
  - attack
related_topics: "[[Concept - BGP Routing Policies]]"
created: 2025-12-17
---

# BGP Hijacking Classifications

## 💡 The Core Idea
BGP Hijacking occurs when an attacker manipulates routing advertisements to redirect traffic intended for a victim to the attacker's network.



## 🧠 Types of Attacks

### By Prefix
* **Exact Prefix:** Attacker announces the same prefix as the owner. Traffic routes to whichever path is shortest.
* **Sub-Prefix:** Attacker announces a **more specific** prefix (e.g., announcing $/24$ when the owner announces $/16$). BGP prefers specific paths, so the attacker captures **all** traffic.
* **Squatting:** Announcing a prefix that has not yet been allocated to anyone.

### By Path
* **Type-0:** Announcing a prefix not owned by self.
* **Type-N:** Creating a fake link in the AS-Path to make a route look legitimate or shorter.

## 🔗 Connections
- **Source:** [[Source - Internet Security]]