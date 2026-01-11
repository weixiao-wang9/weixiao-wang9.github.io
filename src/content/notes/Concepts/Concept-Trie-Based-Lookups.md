---
id: 202512170336
type: atom
course: "[[Computer Networks (CS401)]]"
tags:
  - concept
  - algorithm
  - data-structure
related_topics: "[[Concept - Longest Prefix Match]]"
created: 2025-12-17
---

# Trie-Based Lookups (Unibit & Multibit)

## 💡 The Core Idea
A **Trie** (Prefix Tree) is the data structure used to solve the Longest Prefix Match problem efficiently. It treats IP addresses as a sequence of bits (0 = Left, 1 = Right).



## 🧠 Variations

### Unibit Trie
* **Stride:** 1 bit per step.
* **Pros:** Memory efficient.
* **Cons:** Slow. A 32-bit IP requires 32 memory accesses.

### Multibit Trie (Fixed/Variable Stride)
* **Stride:** Checks $k$ bits per step (e.g., 3 bits).
* **Pros:** Faster. Reduces tree height, requiring fewer memory accesses.
* **Cons:** **Prefix Expansion**. To check 3 bits at a time, a 1-bit prefix (e.g., `1*`) must be expanded into multiple 3-bit entries (`100`, `101`, `110`, `111`). This trades **Memory** for **Speed**.

## 🔗 Connections
- **Source:** [[Source - Router Architecture and Lookups]]