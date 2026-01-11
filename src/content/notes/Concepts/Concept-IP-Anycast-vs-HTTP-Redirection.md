---
id: 202512170435
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - protocol
  - routing
related_topics: "[[Concept - Server Selection Strategies]]"
created: 2025-12-17
---

# Alternative Selection Protocols: Anycast & HTTP

## 💡 The Core Idea
Aside from DNS, CDNs can use network-layer or application-layer protocols to route users to the correct server.

## 🧠 Comparison

### IP Anycast (Network Layer)
* **Mechanism:** Assign the **same IP address** to multiple servers in different locations. BGP routes the user to the "topologically closest" server.
* **Use Case:** DNS Root Servers, Google Public DNS (8.8.8.8).
* **Cons:** BGP doesn't know about server load or link congestion, only hop count.

### HTTP Redirection (Application Layer)
* **Mechanism:** The server receives a GET request and replies with `3xx Redirect` pointing to a new server.
* **Use Case:** Load balancing (e.g., YouTube). If a cluster is overloaded, redirect the user to a neighbor.
* **Cons:** Slow. Requires an initial TCP connection and HTTP exchange just to be told to go elsewhere (extra RTT).

## 🔗 Connections
- **Source:** [[Source - CDNs and Overlay Networks]]