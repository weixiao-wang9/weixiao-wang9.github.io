---
id: 202512170425
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - censorship
  - bgp
  - history
related_topics: []
created: 2025-12-17
title: "Connectivity Disruption via BGP (Egypt Case Study)"
---

# Connectivity Disruption via BGP (Egypt Case Study)

## 💡 The Core Idea
The most blunt form of censorship is "pulling the plug" by withdrawing BGP prefixes from the global routing table, making the country's networks physically unreachable from the outside world.

## 🧠 Case Study: Egypt (2011)
* **Event:** During political protests, the government ordered a shutdown.
* **Mechanism:** ISPs withdrew BGP routes. The number of visible IP prefixes dropped from 2,500 to fewer than 500 in minutes.
* **Detection:** Observed via global BGP monitors (Route Views) and a sudden drop in darknet traffic.

## 🔗 Connections
- **Source:** Source - Internet Surveillance and Censorship