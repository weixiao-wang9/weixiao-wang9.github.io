---
id: 202512170413
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - sdn
  - structure
related_topics:
  - - Concept - SDN Controller Architecture
created: 2025-12-17
---

# SDN Network Architecture

## 💡 The Core Idea
The SDN architecture is composed of three distinct layers: Infrastructure (Device), Control (Controller), and Application.



## 🧠 Components
1.  **SDN-Controlled Elements (Infrastructure):** Switches that forward traffic based on rules. They do not run complex routing protocols; they just execute flow tables.
2.  **SDN Controller:** A logically centralized "Network OS" that maintains the global network state and acts as an interface between devices and apps.
3.  **Network-Control Applications:** Programs (Security, Analytics, Routing) that use the controller to manipulate the network.

## 🔑 Key Features
* **Flow-Based Forwarding:** Decisions based on up to 11 header fields (Transport, Network, Link layers), not just destination IP.
* **Programmable:** The network is managed by software applications using APIs.

## 🔗 Connections
- **Source:** [[Source - Software Defined Networking]]