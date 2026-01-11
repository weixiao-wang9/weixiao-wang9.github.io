---
id: 202512170043
type: atom
status: permanent
tags: [concept, networks, philosophy]
related_topics: [[Concept - Network Address Translation]]
created: 2025-12-17
---

# The End-to-End (E2E) Principle

## 💡 The Core Idea
The network core should be simple and minimal (packet switching only), while intelligence and error correction should reside at the **edges** (end hosts).

## 🧠 Context & Argument
If a function (like error correction or encryption) requires knowledge from the application, it should be implemented in the application layer, not the network layer.

### Benefits
* **Innovation:** Because the core is "dumb," developers can build new applications (video streaming, VoIP) at the edge without asking ISPs to upgrade the core hardware.
* **Efficiency:** Not all applications need reliability (e.g., real-time voice), so the network shouldn't force it.

### Violations of E2E
Sometimes the principle is violated for practical reasons:
1.  **Firewalls:** Intermediate devices that inspect and drop traffic.
2.  **NAT (Network Address Translation):** Modifies IP headers in transit to solve IPv4 address shortage. This breaks the direct addressability of hosts.



## 🔗 Connections
- **Source:** [[Source - Intro to Internet Architecture]]
- **Contradicted by:** [[Concept - Network Address Translation]]