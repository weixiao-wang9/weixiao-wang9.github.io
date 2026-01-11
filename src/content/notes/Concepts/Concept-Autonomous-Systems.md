---
id: 202512170317
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - networks
  - definition
related_topics: "[[Concept - BGP Protocol Basics]]"
created: 2025-12-17
---

# Autonomous Systems (AS)

## 💡 The Core Idea
An **Autonomous System (AS)** is the fundamental unit of routing on the Internet. It is a group of routers and IP prefixes operated by a single administrative authority (like an ISP or University) that applies a unified routing policy.

## 🧠 Context
* **Identification:** Each AS is identified by a unique **ASN** (Autonomous System Number).
* **Routing Protocols:**
    * **IGP (Interior Gateway Protocol):** Used *inside* the AS for efficiency (e.g., OSPF, IS-IS).
    * **EGP (Exterior Gateway Protocol):** Used *between* ASes for policy (e.g., BGP).

## 🔗 Connections
- **Source:** [[Source - Autonomous Systems and Internet Interconnection]]