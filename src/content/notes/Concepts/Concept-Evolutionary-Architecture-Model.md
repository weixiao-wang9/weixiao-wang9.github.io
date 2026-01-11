---
id: 202512170044
type: atom
status: permanent
tags: [concept, networks, theory]
related_topics: []
created: 2025-12-17
---

# Evolutionary Architecture (EvoArch) & The Hourglass

## 💡 The Core Idea
The Internet architecture resembles an **Hourglass**: it is wide at the top (many applications) and wide at the bottom (many physical technologies), but narrow at the waist (IPv4, TCP, UDP).



## 🧠 Context & Argument
Why is it hard to replace TCP/IP?
* **The Evolutionary Shield:** The stability of the transport layer (TCP/UDP) acts as a shield for IPv4. New protocols at the transport layer are unlikely to survive competition with TCP/UDP because those legacy protocols already support so many valuable applications.
* **Ossification:** Because everything depends on the "waist," changing it (e.g., moving to IPv6) is incredibly difficult and expensive.
* **Clean-Slate Design:** Researchers (like the 4D group) propose rebuilding the internet from scratch to solve security and mobility issues, as the current "narrow waist" limits radical innovation in the core.

## 🔗 Connections
- **Source:** [[Source - Intro to Internet Architecture]]