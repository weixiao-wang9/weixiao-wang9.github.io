---
id: 202512170432
type: source
subtype: lecture_note
course: "[[Computer Networks (CS 6250)]]"
module: "Module 11: CDNs & Overlays"
title: Content Distribution Networks (CDNs) and Overlays
status: Finished
tags:
  - input
  - networks
  - cdn
  - dns
  - architecture
created: 2025-12-17
---

# Content Distribution Networks (CDNs) and Overlays

## 📌 Context & Summary
> [!abstract] Context
> This lecture addresses the scalability issues of the traditional client-server model for content delivery. It introduces **CDNs** as a solution to reduce latency and server load. It explores network topology shifts (flattening), server placement strategies (**Enter Deep** vs. **Bring Home**), and the specific mechanisms used to route users to the best server: **DNS**, **IP Anycast**, and **Consistent Hashing**.

## 📝 Notes & Highlights

### Motivation for CDNs
- **Traditional Drawbacks:** A single massive data center suffers from distance-induced latency, redundant transmission of viral content, and single points of failure.
- **The Solution:** CDNs distribute copies of content across geographically dispersed servers to serve users locally.

### Internet Ecosystem Shifts
- **Demand:** Video accounts for the majority of internet traffic.
- **Topological Flattening:** The hierarchy of Tier-1 ISPs is flattening. Traffic is increasingly exchanged locally at **IXPs** (Internet Exchange Points) rather than traversing the global backbone.

### Server Placement Strategies
- **Enter Deep:** Many small clusters deployed deep inside access networks (e.g., Akamai). Minimizes delay but hard to manage.
- **Bring Home:** Fewer large clusters deployed at key IXPs (e.g., Google). Easier to manage but slightly higher latency.

### Server Selection Mechanisms
- **Policy:** How to choose? (Geo-location vs. Network Metrics).
- **Protocol:** How to redirect?
    - **DNS:** The primary method. Maps hostnames to the IP of the best CDN server.
    - **IP Anycast:** BGP routes requests to the topologically closest server sharing an IP.
    - **HTTP Redirection:** Server sends a 3xx response to shift load; adds latency.

### Content-to-Server Mapping
- **Consistent Hashing:** Used to map content to servers in a way that minimizes data movement when servers join or leave the cluster (unlike standard modulo hashing).

## 🔗 Extracted Concepts
- [[Concept - CDN Architecture and Placement]]
- [[Concept - Consistent Hashing]]
- [[Concept - Server Selection Strategies]]
- [[Concept - DNS Architecture]]
- [[Concept - IP Anycast vs HTTP Redirection]]