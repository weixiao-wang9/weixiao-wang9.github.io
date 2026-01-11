---
id: 202512170406
type: atom
course: "[[Computer Networks (CS401)]]"
tags: [concept, algorithms, optimization]
related_topics: [[Concept - Trie-Based Lookups]]
created: 2025-12-17
---

# Packet Classification Algorithms (2D)

## 💡 The Core Idea
Packet classification requires matching packets against multiple fields (e.g., Source IP *and* Destination IP) simultaneously. Specialized algorithms are required to balance **Memory usage** vs. **Lookup Speed**.



## 🧠 The Approaches

### 1. Set-Pruning Tries (Fast but Heavy)
* **Mechanism:** Construct a Destination Trie. For every leaf node, hang a Source Trie containing compatible rules.
* **Pros:** Fast lookup (match Dest, then match Source).
* **Cons:** **Memory Explosion**. A rule like `Dest=0*` must be copied into the source tries for `00*`, `01*`, etc..

### 2. Backtracking (Light but Slow)
* **Mechanism:** Destination nodes point to source tries. If a search in a specific source trie fails, the algorithm backtracks up the destination trie to check ancestors.
* **Pros:** Low memory (rules stored once).
* **Cons:** High time cost due to traversing back up the tree.

### 3. Grid of Tries (The Optimization)
* **Mechanism:** Improves backtracking by adding **Switch Pointers**.
* **Shortcut:** If a match fails in a source trie, the switch pointer jumps directly to the next relevant source trie, skipping the backtracking steps.

## 🔗 Connections
- **Source:** [[Source - Packet Classification and QoS]]