---
id: 202512170416
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - sdn
  - mechanism
related_topics: "[[Concept - SDN Controller Architecture]]"
created: 2025-12-17
---

# Distributed SDN Controllers (ONOS)

## 💡 The Core Idea
To handle large-scale networks, controllers must move from a single server to a distributed cluster that offers scale-out performance and fault tolerance.



## 🧠 ONOS Architecture
**ONOS (Open Networking Operating System)** is a leading distributed controller.
* **Global Network View:** All instances share a consistent view of the network state (Topology, Ports, Links).
* **Clustering:** Multiple ONOS instances run together. If the data plane grows, you simply add more instances.
* **Mastership:**
    * Each switch connects to multiple ONOS instances.
    * Only **one** instance acts as the **Master** for a specific switch.
    * If an instance fails, an election (via ZooKeeper) selects a new master for its switches.

## 🔗 Connections
- **Source:** [[Source - SDN Part 2 and Applications]]