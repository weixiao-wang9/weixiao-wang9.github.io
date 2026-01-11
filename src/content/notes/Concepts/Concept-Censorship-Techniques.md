---
id: 202512170424
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - censorship
  - technique
related_topics: []
created: 2025-12-17
---

# Common Censorship Techniques

## 💡 The Core Idea
Censors use a layered approach to blocking content, ranging from crude IP blocking to sophisticated deep packet inspection.

## 🧠 The Toolkit

### 1. Packet Dropping (IP Blocking)
* **Action:** Drop all traffic to a specific IP address.
* **Cons:** High collateral damage (Overblocking). Shared hosting sites often put innocent sites on the same IP as blocked sites.

### 2. DNS Poisoning
* **Action:** Hijack the DNS resolution process to return a wrong IP.
* **Pros:** No overblocking (targets specific domain names).

### 3. TCP Reset (RST) Injection
* **Action:** Monitor traffic for keywords. If found, send a forged TCP RST packet to both client and server to kill the connection immediately.

### 4. Content Inspection (Proxies)
* **Action:** Route traffic through a proxy that reads the full content.
* **Cons:** Extremely expensive and hard to scale.

## 🔗 Connections
- **Source:** [[Source - Internet Surveillance and Censorship]]