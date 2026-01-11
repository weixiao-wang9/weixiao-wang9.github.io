---
id: 202512170419
type: source
subtype: lecture_note
course: "[[Computer Networks (CS 6250)]]"
module: "Module 8: Network Security"
title: Internet Security and Network Abuse
status: Finished
tags:
  - input
  - networks
  - security
  - bgp
  - ddos
created: 2025-12-17
---

# Internet Security and Network Abuse

## 📌 Context & Summary
> [!abstract] Context
> This lecture covers the landscape of network security, starting with the four fundamental properties (Confidentiality, Integrity, Authentication, Availability). It explores specific attacks on infrastructure, including **DNS Abuse** (Fast-Flux networks), **BGP Hijacking** (Prefix/Path manipulation), and **DDoS** (Reflection/Amplification). It concludes with mitigation techniques like **BGP Flowspec** and **Blackholing**.

## 📝 Notes & Highlights

### Security Properties
- **Confidentiality:** Ensuring the message is only available to the intended parties (e.g., via encryption).
- **Integrity:** Ensuring the message has not been modified in transit.
- **Authentication:** Verifying the identity of the communicating parties.
- **Availability:** Ensuring the service remains accessible despite failures or attacks.

### DNS Abuse
- **Round Robin DNS (RRDNS):** Cycles through a list of IPs for a single domain to distribute load.
- **Fast-Flux Service Networks (FFSN):** Used by attackers. Rapidly changes DNS answers with very short TTLs using a network of compromised proxies (flux agents) to hide the "mothership".

### Network Reputation
- **FIRE (FInding Rogue nEtworks):** Monitors data plane sources (Botnets, Drive-by-downloads, Phishing) to identify malicious networks based on the longevity of malicious content.
- **ASwatch:** Monitors the control plane (BGP). "Bulletproof" ASes show distinct wiring patterns (frequent provider changes) compared to legitimate networks.

### BGP Hijacking
- **Types:**
    - **Exact Prefix:** Announcing a path for the same prefix as the owner.
    - **Sub-prefix:** Announcing a more specific prefix (e.g., $/24$ vs $/16$) to attract traffic.
    - **Squatting:** Announcing an unallocated prefix.
- **Defense:** **ARTEMIS** system uses a local configuration file and BGP monitoring to detect anomalies.
- **Mitigation:** Prefix deaggregation (fighting sub-prefix with sub-prefix) or Multiple Origin AS (MOAS) announcements.

### DDoS Attacks & Mitigation
- **Spoofing:** Falsifying the source IP to impersonate legitimate servers or hide the attacker's location.
- **Amplification/Reflection:** Sending small requests to reflectors (e.g., DNS servers) with a spoofed source IP (the victim), causing the reflectors to flood the victim with large responses.
- **BGP Flowspec:** Fine-grained traffic filtering propagated via BGP.
- **Blackholing:** Dropping traffic to a specific destination (victim) to save the rest of the network, often causing collateral damage.

## 🔗 Extracted Concepts
- [[Concept - Security Fundamentals]]
- [[Concept - Fast-Flux Service Networks]]
- [[Concept - BGP Hijacking Types]]
- [[Concept - DDoS Reflection and Amplification]]
- [[Concept - BGP Flowspec]]
- [[Concept - BGP Blackholing]]