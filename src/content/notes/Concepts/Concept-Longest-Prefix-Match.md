---
id: 202512170335
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - networks
  - algorithm
related_topics: "[[Concept - Trie-Based Lookups]]"
created: 2025-12-17
---

# Longest Prefix Match (LPM)

## 💡 The Core Idea
When forwarding a packet, the router finds the entry in the routing table that shares the **longest common sequence of bits** with the packet's destination address.

## 🧠 Context
With **CIDR** (Classless Inter-Domain Routing), multiple entries might match a single IP.
* *Rule:* If an IP matches both `192.168.1.0/24` and `192.168.1.0/28`, the router chooses the `/28` entry because it is **more specific**.

### Why is this hard?
* **Scale:** Tables have 1,000,000+ entries.
* **Speed:** Must happen at line rate (nanoseconds).
* **Memory:** High-speed memory (SRAM) is expensive; low-speed memory (DRAM) is too slow for inefficient algorithms.

## 🔗 Connections
- **Source:** [[Source - Router Architecture and Lookups]]