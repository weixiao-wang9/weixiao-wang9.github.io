---
id: 202512170426
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - censorship
  - measurement
  - tcp
related_topics: []
created: 2025-12-17
---

# Augur: Detecting Filtering via Side Channels

## 💡 The Core Idea
Augur detects if two hosts (e.g., a Reflector and a Site) are blocked from talking to each other, *without* having direct control over either of them.

## 🧠 Mechanism: IP ID Side Channel
It leverages the global **IP ID counter** in IP headers.
1.  **Probe:** Measure the Reflector's current IP ID.
2.  **Perturb:** Send a spoofed packet to the Site, pretending to be the Reflector.
    * If the Site receives it, it replies to the Reflector.
    * The Reflector replies with a RST, incrementing its IP ID.
3.  **Measure:** Check the Reflector's IP ID again.
    * **Incremented?** The hosts communicated (No filtering).
    * **Not Incremented?** The packets were blocked (Filtering detected).

## 🔗 Connections
- **Source:** [[Source - Internet Surveillance and Censorship]]