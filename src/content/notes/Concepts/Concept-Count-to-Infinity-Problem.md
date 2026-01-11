---
id: 202512170139
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - networks
  - failure-mode
related_topics: "[[Concept - Distance Vector Routing]]"
created: 2025-12-17
---

# Count-to-Infinity Problem & Poison Reverse

## 💡 The Core Idea
A major flaw in Distance Vector routing. "Good news" (cost decrease) propagates fast, but "Bad news" (link failure or cost increase) propagates slowly, causing routing loops where nodes bounce packets back and forth indefinitely (or until they count to infinity).

## 🧠 Mechanism
If link $X-Y$ fails (cost jumps to 60):
1.  $Y$ might switch to route through $Z$, because $Z$ *was* routing through $Y$.
2.  $Y$ thinks $Z$ has a path, and $Z$ thinks $Y$ has a path.
3.  They keep incrementing their costs back and forth, assuming the other still has a valid route.

### The Fix: Poison Reverse
If node $Z$ routes through $Y$ to get to destination $X$, $Z$ will lie to $Y$ and say its distance to $X$ is **Infinity**.
* This prevents $Y$ from trying to route through $Z$ if the direct link fails, effectively breaking the loop immediately.
* *Limitation:* Does not solve loops involving 3+ nodes.

## 🔗 Connections
- **Source:** [[Source - Intradomain Routing]]