---
id: 202512170424
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - censorship
  - dns
  - mechanism
related_topics: []
created: 2025-12-17
---

# DNS Injection (Great Firewall of China)

## 💡 The Core Idea
DNS Injection is a censorship technique where a firewall monitors DNS queries and injects a fake response (spoofed IP) before the legitimate response can arrive.



## 🧠 Mechanism
1.  **Surveillance:** The GFW inspects DNS queries at the network edge.
2.  **Matching:** If the domain matches a blocklist, the GFW triggers a response.
3.  **Injection:** A fake DNS A record is sent back to the client immediately.
4.  **Race Condition:** Because the fake response usually arrives faster than the real response from the legitimate server, the client accepts the fake one and the connection fails.

## 🔗 Connections
- **Source:** [[Source - Internet Surveillance and Censorship]]