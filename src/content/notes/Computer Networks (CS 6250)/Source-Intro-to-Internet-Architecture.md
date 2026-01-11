---
id: 202512170036
type: source
subtype: lecture_note
course: "[[Computer Networks (CS 6250)]]"
module: "Module 1: Introduction, History, and Internet Architecture"
status: Finished
tags:
  - input
  - networks
  - cs
created: 2025-12-17
---

# Introduction, History, and Internet Architecture

## 📌 Context & Summary
> [!abstract] AI Context
> This lecture covers the foundational history of the Internet (from ARPANET to WWW), the design philosophy of the 5-layer Internet Architecture versus the OSI model, and the crucial "End-to-End Principle." It also details the "Hourglass" evolutionary model (EvoArch) explaining why TCP/IP is ossified, and concludes with Layer 2 switching logic, specifically the Spanning Tree Protocol (STP).

## 📝 Notes & Highlights

### History of the Internet
- **1960s:** J.C.R. Licklider proposed the "Galactic Network." ARPANET (1969) connected the first 4 nodes (UCLA, SRI, UCSB, Utah).
- **1970s:** NCP (Network Control Protocol) was the first host-to-host protocol. In 1973, TCP/IP was introduced by Bob Kahn and Vint Cerf to allow open-architecture networking.
- **1980s-90s:** DNS (1983) solved scalability issues for host names. WWW (1990) by Tim Berners-Lee popularized the net.

### The Layered Architecture
- The internet uses a **Layered Architecture** to ensure scalability and modularity.
- **Analogy:** Airline system (Ticket -> Baggage -> Gate -> Runway). Each layer serves the one above it.
- **OSI Model (7 Layers):** Application, Presentation, Session, Transport, Network, Data Link, Physical.
- **Internet Model (5 Layers):** Combines App, Presentation, and Session into one "Application Layer."


### Interconnecting Devices
- **Repeater/Hub (L1):** Physical signal forwarding. Same collision domain.
- **Bridge/Switch (L2):** Forwards based on MAC addresses.
- **Router (L3):** Routes packets based on IP.

## 🔗 Extracted Concepts
- [[Concept - Internet Protocol Stack]]
- [[Concept - Data Encapsulation]]
- [[Concept - End-to-End Principle]]
- [[Concept - Evolutionary Architecture Model]]
- [[Concept - Spanning Tree Protocol]]