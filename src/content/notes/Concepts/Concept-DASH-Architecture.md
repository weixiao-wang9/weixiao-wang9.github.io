---
id: 202512170428
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - streaming
  - protocol
related_topics:
created: 2025-12-17
---

# Dynamic Adaptive Streaming over HTTP (DASH)

## 💡 The Core Idea
DASH is a "pull-based" streaming architecture. The intelligence is at the **Client**, which dynamically requests video chunks of varying quality based on network conditions, while the **Server** remains stateless.



## 🧠 Workflow
1.  **Encoding:** Server stores video divided into chunks (e.g., 5 seconds), encoded at multiple bitrates (e.g., 500kbps, 1.5Mbps, 3Mbps).
2.  **Manifest:** Client downloads a manifest file listing the URLs for all chunks and qualities.
3.  **Adaptation:** The client estimates network conditions and requests the appropriate chunk via HTTP GET.
4.  **Transport:** Uses **TCP** for reliability (no artifacts) and **HTTP** to leverage existing CDNs and bypass firewalls.

## 🔗 Connections
- **Source:** [[Source - Multimedia Applications]]