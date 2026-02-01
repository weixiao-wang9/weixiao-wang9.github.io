---
id: 202512170434
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags: [concept, protocol, dns]
related_topics: []
created: 2025-12-17
---

# DNS Architecture and Records

## 💡 The Core Idea
DNS is a distributed, hierarchical database that translates hostnames (human-readable) to IP addresses (router-readable). It is the primary mechanism CDNs use to redirect traffic.



## 🧠 Hierarchy & Query Types
* **Hierarchy:** Root Servers $\rightarrow$ TLD Servers (.com) $\rightarrow$ Authoritative Servers (amazon.com).
* **Iterative Query:** The server replies "I don't know, but ask this server next."
* **Recursive Query:** The server takes the responsibility to find the answer and return it.

## 📄 Resource Records (RR)
* **A:** Hostname $\rightarrow$ IP Address.
* **NS:** Domain $\rightarrow$ Authoritative Name Server.
* **CNAME:** Alias $\rightarrow$ Canonical Name (Crucial for CDNs to redirect `video.site.com` to `cdn.provider.com`).
* **MX:** Mail Server mapping.

## 🔗 Connections
- **Source:** [[Source - CDNs and Overlay Networks]]