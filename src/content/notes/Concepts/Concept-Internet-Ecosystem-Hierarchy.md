---
id: 202512170316
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - networks
  - infrastructure
related_topics: "[[Concept - Autonomous Systems]]"
created: 2025-12-17
---

# Internet Ecosystem Hierarchy

## 💡 The Core Idea
The Internet is not a single network but a **Network of Networks** organized into a loose hierarchy based on size and reach.



## 🧠 Structure
* **Tier-1 ISPs:** The "Backbone" (e.g., AT&T, Level-3). They operate globally and peer with every other Tier-1 ISP to reach the entire internet without paying settlement fees.
* **Tier-2 (Regional) ISPs:** Connect to Tier-1 networks and provide connectivity to smaller local ISPs.
* **Tier-3 (Access) ISPs:** Provide the "last mile" connection to homes and businesses.
* **CDNs:** Content Delivery Networks (Google, Netflix) flatten this hierarchy by connecting directly to Access ISPs or IXPs to bypass the backbone.

## 🔗 Connections
- **Source:** [[Source - Autonomous Systems and Internet Interconnection]]