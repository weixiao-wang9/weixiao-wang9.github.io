---
id: 202512170414
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - sdn
  - mechanism
related_topics:
  - - Concept - OpenDayLight Architecture
created: 2025-12-17
---

# SDN Controller Architecture (Northbound & Southbound)

## 💡 The Core Idea
The Controller is the "glue" of SDN. It sits in the middle, translating high-level application intent into low-level device rules.



## 🧠 The Layers

### Southbound Interface (Down)
* **Connects to:** Network Elements (Switches/Routers).
* **Function:** Collects network state (topology, heartbeats) and pushes flow rules.
* **Protocol:** **OpenFlow** is the standard protocol here.

### Network-Wide State Management (Middle)
* **Function:** Stores the "truth" of the network (Links, Hosts, Switch Flow Tables).
* **Implementation:** Distributed across servers for fault tolerance.

### Northbound Interface (Up)
* **Connects to:** Applications (Analytics, Routing).
* **Function:** Exposes the network state to apps via API (e.g., REST).
* **Benefit:** Apps don't need to know the details of the hardware; they just ask the controller.

## 🔗 Connections
- **Source:** [[Source - Software Defined Networking]]