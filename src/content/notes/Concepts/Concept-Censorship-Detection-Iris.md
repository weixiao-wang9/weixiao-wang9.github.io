---
id: 202512170425
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - censorship
  - measurement
  - dns
related_topics: []
created: 2025-12-17
---

# Iris: Detecting DNS Manipulation

## 💡 The Core Idea
Iris is a system that uses thousands of **Open DNS Resolvers** worldwide to measure DNS manipulation without requiring user participation.



## 🧠 Methodology
1.  **Scan:** Find open resolvers in Internet infrastructure (avoiding home routers).
2.  **Query:** Ask these resolvers for sensitive domains.
3.  **Verify:** Check if the response is valid using:
    * **Consistency Metrics:** Does the IP match what other resolvers see?
    * **Independent Verifiability:** Does the returned IP present a valid HTTPS certificate for that domain?

## 🔗 Connections
- **Source:** [[Source - Internet Surveillance and Censorship]]