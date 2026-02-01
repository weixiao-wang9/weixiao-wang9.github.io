---
id: 202512170433
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - cdn
  - architecture
related_topics:
  - - Concept - Server Selection Strategies
created: 2025-12-17
---

# CDN Server Placement Strategies

## 💡 The Core Idea
CDNs must decide where to place their server clusters within the vast Internet topology to balance **performance** (latency) against **management complexity**.



## 🧠 Strategies

### 1. Enter Deep (e.g., Akamai)
* **Philosophy:** Get as close to the user as possible.
* **Deployment:** Place many small server clusters deep inside **Access Networks** (ISPs) worldwide.
* **Pros:** Lowest latency, high throughput.
* **Cons:** High maintenance complexity (updating thousands of clusters).

### 2. Bring Home (e.g., Google/Limelight)
* **Philosophy:** Centralize slightly to simplify operations.
* **Deployment:** Place fewer, larger clusters at key **IXPs** (Internet Exchange Points).
* **Pros:** Easier management and maintenance.
* **Cons:** Higher delay compared to "Enter Deep".

## 🔗 Connections
- **Source:** [[Source - CDNs and Overlay Networks]]