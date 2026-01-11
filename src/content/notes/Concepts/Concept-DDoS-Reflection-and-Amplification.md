---
id: 202512170421
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - security
  - ddos
  - attack
related_topics: []
created: 2025-12-17
---

# DDoS Reflection and Amplification

## 💡 The Core Idea
A technique to increase the volume of a Denial of Service attack by tricking third-party servers (reflectors) into flooding the victim.



## 🧠 Mechanism
1.  **Spoofing:** The attacker sends a request to a reflector (e.g., DNS server) but falsifies the source IP address to be the **victim's IP**.
2.  **Reflection:** The server replies to the request, sending the response to the victim instead of the attacker.
3.  **Amplification:** The attacker chooses protocols where the **response is much larger than the request** (e.g., a 60-byte DNS query triggers a 3000-byte response). This multiplies the bandwidth hitting the victim.

## 🔗 Connections
- **Source:** [[Source - Internet Security]]