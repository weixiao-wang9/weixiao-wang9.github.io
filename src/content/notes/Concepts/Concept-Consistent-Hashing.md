---
id: 202512170433
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags: [concept, algorithm, distributed-systems]
related_topics: []
created: 2025-12-17
---

# Consistent Hashing

## 💡 The Core Idea
In a distributed system (like a CDN cluster), we need to map content to specific servers. Traditional hashing ($Key \% N$) fails because if $N$ changes (server failure/addition), *all* keys must be remapped. **Consistent Hashing** minimizes this disruption.



## 🧠 Mechanism
1.  **The Ring:** Map both **Servers** and **Content** to the same circular ID space (e.g., 0 to $2^{32}-1$) using a hash function.
2.  **Assignment:** A piece of content is assigned to the next server encountered moving clockwise on the ring.
3.  **Resilience:**
    * **Node Joins:** Only keys falling in the gap before the new node need to move to it.
    * **Node Leaves:** Only keys belonging to the failed node move to the next successor.
* **Result:** Minimizes data movement, maintaining load balance.

## 🔗 Connections
- **Source:** [[Source - CDNs and Overlay Networks]]