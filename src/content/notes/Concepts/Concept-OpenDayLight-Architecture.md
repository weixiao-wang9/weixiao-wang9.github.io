---
id: 202512170414
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - sdn
  - implementation
related_topics:
  - - Concept - SDN Controller Architecture
created: 2025-12-17
---

# OpenDayLight (ODL) Architecture

## 💡 The Core Idea
OpenDayLight is a widely used, open-source SDN controller that uses a **Model Driven Service Abstraction Layer (MD-SAL)** to integrate various protocols and apps.



## 🧠 MD-SAL Components
MD-SAL allows developers to write services that plug into the controller. It relies on a shared datastore with two trees:

1.  **Config Datastore:** Represents the *desired* state of the network. (What we want to happen).
2.  **Operational Datastore:** Represents the *actual* state of the network received from devices. (What is happening right now).

## 🔗 Connections
- **Source:** [[Source - Software Defined Networking]]