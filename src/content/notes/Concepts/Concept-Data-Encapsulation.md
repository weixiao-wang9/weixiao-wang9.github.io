---
id: 202512170041
type: atom
status: permanent
tags:
  - concept
  - networks
  - mechanism
related_topics: "[[Concept - Internet Protocol Stack]]"
created: 2025-12-17
---

# Data Encapsulation and De-encapsulation

## 💡 The Core Idea
Encapsulation is the process where data moves down the protocol stack, with each layer adding its own header information (metadata) to the payload received from the layer above. De-encapsulation is the reverse process at the receiving end.



## 🧠 Context & Mechanism
### The Process
1.  **Application:** Creates a **Message**.
2.  **Transport:** Adds header ($H_t$) to create a **Segment**. This header helps the receiver identify the application ports and perform error detection.
3.  **Network:** Adds header ($H_n$) to create a **Datagram**. Contains source/destination IP addresses.
4.  **Data Link:** Adds header ($H_l$) to create a **Frame**.

### Intermediate Devices
* **Switches (Layer 2):** Decapsulate only up to the Link Layer to read MAC addresses.
* **Routers (Layer 3):** Decapsulate up to the Network Layer to read IP addresses.
* **End Hosts:** Implement all 5 layers.

## 🔗 Connections
- **Source:** [[Source - Intro to Internet Architecture]]