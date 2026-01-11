---
id: 202512170421
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - security
  - defense
  - protocol
related_topics: []
created: 2025-12-17
---

# BGP Flowspec

## 💡 The Core Idea
BGP Flowspec is an extension to the BGP protocol that allows administrators to distribute fine-grained traffic filtering rules (like firewalls) across AS borders.

## 🧠 Capability
* **Granularity:** Unlike standard BGP (which routes based on destination IP), Flowspec can match based on Source IP, Packet Length, Protocol, and Drop Rate.
* **Actions:** Can discard traffic (rate limit to 0), rate limit, or redirect.
* **Pros:** More scalable than manual ACLs and leverages the existing BGP control plane.
* **Cons:** Relies on trust between networks; generally used for intra-domain rather than inter-domain security.

## 🔗 Connections
- **Source:** [[Source - Internet Security]]