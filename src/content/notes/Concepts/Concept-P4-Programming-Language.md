---
id: 202512170417
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - sdn
  - p4
  - programming
related_topics: []
created: 2025-12-17
---

# P4 (Programming Protocol-independent Packet Processors)

## 💡 The Core Idea
P4 is a language that allows the controller to redefine **how** a switch parses and processes packets. Unlike OpenFlow, which relies on fixed tables and protocols, P4 makes the hardware **Protocol Independent**.



## 🧠 Key Goals
1.  **Reconfigurability:** The controller can change how the switch parses packets (e.g., defining a new header type on the fly).
2.  **Protocol Independence:** The switch isn't hardcoded for IP or Ethernet. The controller defines the format.
3.  **Target Independence:** The same P4 program can be compiled to run on different hardware (ASIC, FPGA, Software Switch).

## ⚙️ Workflow
* **Configure:** Program the parser to recognize specific headers.
* **Populate:** Add entries to the Match+Action tables to define policies.

## 🔗 Connections
- **Source:** [[Source - SDN Part 2 and Applications]]