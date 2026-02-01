---
id: 202512170420
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - security
  - dns
  - attack
related_topics: []
created: 2025-12-17
---

# Fast-Flux Service Networks (FFSN)

## 💡 The Core Idea
Fast-Flux is a technique used by attackers to hide malicious content servers (motherships) behind a constantly changing network of compromised hosts (flux agents).



## 🧠 Mechanism
* **Rapid Churn:** It uses very short Time-To-Live (TTL) values in DNS records.
* **Proxy Network:** When a user queries the domain, they get a list of IPs belonging to compromised machines (proxies).
* **Relay:** These proxies forward the HTTP request to the actual backend control node (mothership).
* **Resilience:** If one proxy is taken down, DNS simply returns a different set of IPs in the next lookup, making the scam hard to kill.

## 🔗 Connections
- **Source:** [[Source - Internet Security]]