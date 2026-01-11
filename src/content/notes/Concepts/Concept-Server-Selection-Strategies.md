---
id: 202512170433
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - cdn
  - optimization
related_topics: "[[Concept - DNS Architecture]]"
created: 2025-12-17
---

# CDN Server Selection Policies

## 💡 The Core Idea
When a user requests content, the CDN must select a specific **Cluster** (location) and a specific **Server** (machine).

## 🧠 Cluster Selection (Which Location?)
1.  **Geo-Proximity:** Pick the geographically closest cluster "as the crow flies."
    * *Flaw:* Closest geo-location might have a congested network path or longer BGP route.
    * *Flaw:* Decisions are often based on the user's **LDNS** location, not the user's actual IP.
2.  **Real-time Measurements:**
    * **Active:** LDNS pings multiple clusters (rare/traffic heavy).
    * **Passive:** The CDN monitors TCP connection stats from existing users in the same subnet to predict performance.

## 🧠 Server Selection (Which Machine?)
Once inside the cluster, simple load balancing isn't enough because not all servers hold all content (disk space limits).
* **Content-Aware Hashing:** Requests for specific content are hashed to specific servers (via Consistent Hashing) to maximize cache hit rates.

## 🔗 Connections
- **Source:** [[Source - CDNs and Overlay Networks]]