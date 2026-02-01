---
id: 202512170412
type: source
subtype: lecture_note
course: "[[Computer Networks (CS401)]]"
module: "Module 7: SDN"
title: Software Defined Networking (SDN)
author: Course Instructor
status: finished
tags: [input, networks, sdn, architecture]
created: 2025-12-17
---

# Software Defined Networking (SDN)

## 📌 Context & Summary
> [!abstract] Context
> This lecture introduces SDN, a paradigm shift from traditional, tightly coupled hardware to a programmable network architecture. It covers the **Separation of the Control and Data Planes**, the history of programmable networks (Active Networks -> OpenFlow), and the layered architecture of an SDN Controller (Northbound/Southbound APIs), specifically looking at **OpenDayLight**.

## 📝 Notes & Highlights

### The Problem with Traditional Networks
- **Complexity:** Networks handle diverse equipment (middleboxes, firewalls, routers) with different protocols.
- **Proprietary:** Closed software makes centralized management difficult; interfaces vary by vendor.

### The SDN Solution
- **Separation of Tasks:** Divides the network into the **Control Plane** (Brain/Software) and **Data Plane** (Muscle/Hardware) to speed up innovation.

### History of SDN
1.  **Active Networks (Mid-90s):** Attempted to open network control via "Capsules" (code in packets) or Programmable Routers. Aimed to accelerate protocol standardization.
2.  **Control/Data Separation (2001-2007):** Driven by traffic growth. Focused on logically centralized control and reliability.
3.  **OpenFlow (2007-2010):** Born from the need for large-scale experimentation. Enabled standardized access to flow tables in switches.

### SDN Architecture
- **Infrastructure Layer:** The switches (Data Plane).
- **Control Layer:** Logically centralized controller.
- **Application Layer:** Network management, security, and traffic engineering apps.
- **Key Features:** Flow-based forwarding, Separation of planes, Network control functions, Programmability.

### The Controller
- **Southbound Interface:** Talks to switches (e.g., OpenFlow).
- **Northbound Interface:** Talks to Apps (e.g., REST API).
- **Example:** **OpenDayLight** uses MD-SAL (Model Driven Service Abstraction Layer) to manage config and operational data.

## 🔗 Extracted Concepts
- [[Concept - Control Plane vs Data Plane]]
- [[Concept - Evolution of SDN]]
- [[Concept - SDN Network Architecture]]
- [[Concept - SDN Controller Architecture]]
- [[Concept - OpenDayLight Architecture]]