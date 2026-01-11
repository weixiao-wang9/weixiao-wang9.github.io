---
id: 202512170424
type: source
subtype: lecture_note
course: "[[Computer Networks (CS 6250)]]"
module: "Module 9: Censorship"
title: Internet Surveillance and Censorship
status: Finished
tags:
  - input
  - networks
  - security
  - censorship
created: 2025-12-17
---

# Internet Surveillance and Censorship

## 📌 Context & Summary
> [!abstract] Context
> This lecture explores how governments and organizations restrict Internet access. It details technical mechanisms like **DNS Injection**, **Packet Dropping**, and **BGP disruption**. It examines real-world case studies (China's GFW, Egypt, Libya) and introduces measurement systems like **Augur** (TCP side channels) and **Iris** (DNS measurements).

## 📝 Notes & Highlights

### DNS Censorship
- **Definition:** Large-scale traffic filtering by suppressing domain resolution.
- **Example:** **The Great Firewall of China (GFW)** injects fake DNS responses to block domains.
- **Techniques:**
    1.  **Packet Dropping:** Discarding all traffic to specific IPs. Simple but causes "overblocking" (collateral damage).
    2.  **DNS Poisoning:** Returning incorrect IP addresses for a domain request.
    3.  **Content Inspection:** Using Proxies or IDS to inspect payloads for keywords. Precise but expensive.
    4.  **TCP Resets:** Sending forged TCP RST packets to tear down connections containing sensitive keywords.

### Connectivity Disruptions
- **Routing Disruption:** Withdrawing BGP prefixes to make entire networks unreachable (e.g., Egypt 2011).
- **Packet Filtering:** Configuring firewalls to block traffic matching certain criteria (e.g., Libya 2011).

### Detection Systems
- **Iris:** Uses open DNS resolvers worldwide to detect **DNS manipulation** by measuring consistency and independent verifiability.
- **Augur:** Uses **TCP/IP side channels** (Global IP ID counters) to detect connectivity blocking between two hosts without direct access to them.

## 🔗 Extracted Concepts
- [[Concept - DNS Injection and GFW]]
- [[Concept - Censorship Techniques]]
- [[Concept - Connectivity Disruption]]
- [[Concept - Censorship Detection]]
