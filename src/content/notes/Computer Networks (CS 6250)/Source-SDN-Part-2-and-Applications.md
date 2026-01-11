---
id: 202512170416
type: source
subtype: lecture_note
course: "[[Computer Networks (CS 6250)]]"
module: "Module 7: SDN Advanced"
title: SDN Architecture, P4, and Applications
author: Course Instructor
status: Finished
tags:
  - input
  - networks
  - sdn
  - p4
  - sdx
created: 2025-12-17
---

# SDN Architecture, P4, and Applications

## 📌 Context & Summary
> [!abstract] Context
> This lecture expands on SDN fundamentals. It details the **SDN Landscape** (from Infrastructure to Applications), contrasts Centralized vs. Distributed Controllers (**ONOS**), and introduces **P4**, a language for programming the data plane itself (making switches protocol-independent). It concludes with **SDX** (Software Defined Exchange), which applies SDN to Internet Exchange Points to fix BGP limitations.

## 📝 Notes & Highlights

### Motivation & Landscape
- **The Problem:** Traditional IP networks are tightly coupled (control and data planes bundled) and complex to manage.
- **The Solution:** SDN separates the control logic (Controller) from the forwarding hardware.
- **The Layers:**
    1.  **Infrastructure:** Forwarding elements (switches).
    2.  **Southbound Interface:** API between Controller and Switch (e.g., OpenFlow).
    3.  **Network OS (Controller):** Provides abstractions and common APIs (e.g., OpenDayLight).
    4.  **Northbound Interface:** API between Controller and Apps.
    5.  **Network Applications:** Routing, Security, Load Balancing.

### Controllers: Centralized vs. Distributed
- **Centralized:** Single point of failure, scaling issues (e.g., Maestro, Beacon).
- **Distributed:** Scalable and fault-tolerant.
    - **ONOS (Open Networking Operating System):** A distributed controller where instances form a cluster. It maintains a "Global Network View" and uses mastership elections to control switches.

### Programming the Data Plane (P4)
- **Limitation of OpenFlow:** OpenFlow started with fixed rule tables; adding new header fields required protocol updates.
- **P4 (Programming Protocol-independent Packet Processors):** A language to configure switches programmatically.
    - **Goals:** Reconfigurability, Protocol Independence, and Target Independence.
    - **Model:** Uses a programmable parser and Match+Action tables.

### SDN Applications: SDX
- **SDX (Software Defined Internet Exchange):** Applies SDN to IXPs to overcome BGP's inability to route based on application type or source IP.
- **Architecture:** Gives each participant AS the illusion of its own virtual switch to define custom policies.

## 🔗 Extracted Concepts
- [[Concept - The SDN Landscape]]
- [[Concept - Distributed SDN Controllers]]
- [[Concept - P4 Programming Language]]
- [[Concept - Software Defined Internet Exchange]]