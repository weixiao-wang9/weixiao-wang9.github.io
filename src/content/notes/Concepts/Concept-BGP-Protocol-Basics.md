---
id: 202512170318
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - networks
  - protocol
related_topics: "[[Concept - BGP Routing Policies]]"
created: 2025-12-17
---

# Border Gateway Protocol (BGP)

## 💡 The Core Idea
BGP is the protocol that glues the internet together. It allows ASes to exchange reachability information over TCP sessions.



## 🧠 Mechanism
* **eBGP (External BGP):** Sessions between routers in *different* ASes. Used to learn routes from neighbors.
* **iBGP (Internal BGP):** Sessions between routers in the *same* AS. Used to disseminate external routes internally.
    * *Constraint:* iBGP routers usually require a "full mesh" connection to ensure all routers know external paths.
* **Attributes:** BGP uses path attributes (AS-PATH, NEXT-HOP) rather than simple metrics to prevent loops and enforce policy.

## 🔗 Connections
- **Source:** [[Source - Autonomous Systems and Internet Interconnection]]