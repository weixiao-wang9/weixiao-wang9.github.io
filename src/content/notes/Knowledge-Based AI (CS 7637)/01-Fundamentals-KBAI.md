---
type: note
course: "[[Knowledge-Based AI (CS 7637)]]"
date: 2026-02-02
topic: Fundamentals
lessons: 1-3
prerequisites:
  - Basic understanding of AI concepts
  - Familiarity with problem-solving
---

# Fundamentals of Knowledge-Based AI

## Prerequisites

- Basic understanding of artificial intelligence
- Familiarity with computational thinking
- Interest in cognitive science and human-like AI

## Learning Goals

After completing this module, you will be able to:

1. Define knowledge-based AI and distinguish it from other AI approaches
2. Understand the fundamental conundrums and characteristics of AI
3. Describe cognitive system architecture with its three layers
4. Create and use semantic networks as knowledge representations
5. Apply the "represent and reason" paradigm to solve problems
6. Evaluate knowledge representations based on key criteria

---

## 1. Introduction to Knowledge-Based AI

### The Five Fundamental Conundrums of AI

Knowledge-Based AI must address five core challenges that define the field:

**Conundrum 1: Computational Resources vs. Complexity**
- AI agents have limited computational resources (processing speed, memory)
- Most interesting AI problems are computationally intractable
- **Challenge:** How can AI agents deliver near real-time performance on complex problems?

**Conundrum 2: Local Computation vs. Global Constraints**
- All computation is local (happens in specific places/times)
- Most AI problems have global constraints (affect entire system)
- **Challenge:** How can AI agents address global problems using only local computation?

**Conundrum 3: Deductive Logic vs. Abductive/Inductive Problems**
- Computational logic is fundamentally deductive
- Many AI problems require abduction (hypothesis generation) or induction (generalization)
- **Challenge:** How can AI agents solve abductive and inductive problems?

**Conundrum 4: Limited Knowledge vs. Novel Problems**
- The world is dynamic and knowledge is limited
- AI agents must always begin with what they already know
- **Challenge:** How can AI agents address completely new problems?

**Conundrum 5: Complexity of Explanation**
- Problem solving and reasoning are already complex
- Explanation and justification add additional complexity
- **Challenge:** How can AI agents explain or justify their decisions?

### Characteristics of AI Agents

AI agents operate under bounded rationality with these inherent limitations:

1. **Limited Computing Power:** Constrained processing speed and memory
2. **Limited Sensors:** Cannot perceive everything in the world
3. **Limited Attention:** Cannot focus on everything simultaneously
4. **Deductive Logic:** Fundamentally deductive computational systems
5. **Incomplete Knowledge:** World knowledge is partial relative to the full world

**Central Question:** How can AI agents with bounded rationality address open-ended problems in complex, dynamic environments?

---

## 2. The Four Schools of AI

AI approaches can be categorized along two dimensions, creating four distinct schools:

### The Two Dimensions

**Dimension 1: Thinking vs. Acting**
- **Thinking:** Internal reasoning processes, cognition, decision-making
- **Acting:** External behaviors, actions in the world, performance

**Dimension 2: Optimally vs. Human-like**
- **Optimally:** Mathematically ideal, provably correct, efficiency-focused
- **Human-like:** Models human cognition, exhibits human behaviors

### The Four Quadrants

```
                    Optimally                Human-like
               ┌─────────────────┬──────────────────────┐
   Thinking    │  Optimal        │  Human-like          │
               │  Reasoning      │  Thinking            │
               │                 │  (Cognitive          │
               │                 │   Modeling)          │
               ├─────────────────┼──────────────────────┤
   Acting      │  Optimal        │  Human-like          │
               │  Behavior       │  Acting              │
               │  (Rational      │                      │
               │   Agents)       │                      │
               └─────────────────┴──────────────────────┘
```

**Examples:**
- **Watson (IBM):** Optimal reasoning - designed to win at Jeopardy through computational power
- **Self-driving cars:** Optimal behavior - rational agents acting in the world
- **Cognitive architectures:** Human-like thinking - modeling human reasoning
- **Social robots:** Human-like acting - exhibiting recognizable human behaviors

**KBAI Position:** Knowledge-Based AI primarily occupies the right side of the spectrum (human-like), focusing on human-level, human-like intelligence. KBAI is concerned with both thinking and acting, but always from a cognitive perspective.

---

## 3. Cognitive Systems

### Definition

**Cognitive Systems** are systems that exhibit human-level, human-like intelligence through interaction among components like learning, reasoning, and memory.

- **Cognitive:** Dealing with human-like intelligence; ultimate goal is human-level AI
- **Systems:** Multiple interacting components working together

### Three-Layer Cognitive Architecture

Cognitive systems map percepts (inputs from the world) to actions (outputs to the world) through three distinct layers:

```
        Percepts from World
               ↓
    ┌──────────────────────────┐
    │    METACOGNITION         │  ← Reasoning about reasoning
    │  (Thinking about         │    Self-reflection and
    │   thinking)              │    strategy adjustment
    ├──────────────────────────┤
    │    DELIBERATION          │  ← Goal-driven reasoning
    │  ┌──────────────────┐   │    Planning and problem-solving
    │  │    Learning      │←──┼──┐
    │  │       ↕          │   │  │
    │  │    Reasoning ←───┼───┼──┤ Tightly coupled
    │  │       ↕          │   │  │ processes
    │  │     Memory       │←──┼──┘
    │  └──────────────────┘   │
    ├──────────────────────────┤
    │    REACTION              │  ← Direct percept-action mapping
    │  (Reflexive response)    │    Fast, automatic responses
    └──────────────────────────┘
               ↓
        Actions on World
```

**Layer 1: Reaction**
- Direct mapping of percepts to actions
- Fast, automatic responses (e.g., brake lights → press brakes)
- No deliberation or planning involved
- Handles immediate, time-critical situations

**Layer 2: Deliberation**
- Goal-driven reasoning and planning
- Core of knowledge-based AI
- Three intimately connected processes:
  - **Learning:** Acquiring knowledge from experience
  - **Reasoning:** Using knowledge to solve problems
  - **Memory:** Storing and retrieving knowledge
- Example: Planning a lane change while driving

**Layer 3: Metacognition**
- Reasoning about internal mental processes
- Reflects on deliberation and reaction
- Enables self-improvement and strategy adjustment
- Example: After a poor lane change, evaluating your decision-making process and adjusting future behavior

### The Deliberation Trinity

The three processes of deliberation form a unified, interdependent system:

```
         Learning ←→ Reasoning
             ↕           ↕
           Memory ←──────┘
```

**Interconnections:**
- We **learn** so we can **reason**
- The results of **reasoning** often lead to additional **learning**
- Once we **learn**, we store knowledge in **memory**
- We need knowledge from **memory** to **learn** (the more we know, the more we can learn)
- **Reasoning** requires knowledge that **memory** provides
- Results of **reasoning** can be stored in **memory**

**Key Principle:** KBAI develops unified theories that integrate reasoning, learning, and memory rather than treating them separately.

---

## 4. Semantic Networks

### What Are Semantic Networks?

**Semantic networks** are structured knowledge representations that explicitly capture objects, relationships, and transformations using nodes and labeled links.

### Structure of Semantic Networks

**Lexicon (Vocabulary):**
- **Nodes** represent objects, concepts, or entities (e.g., X, Y, Z)

**Structure (Composition):**
- **Links** with directions connect nodes
- Links capture relationships between objects
- Enable composition of nodes into complex representations

**Semantics (Inference):**
- **Labels** on links specify relationship types
- Labels enable drawing inferences and reasoning
- Support systematic problem-solving

### Example: Ravens Progressive Matrices (2×1)

Consider a simple visual analogy problem: A is to B as C is to D

**Image A:**
```
Diamond  ──inside──▶  Circle

Diamond  ──size=small
```

**Semantic Network for A:**
```
        inside

Diamond ───────▶ Circle

   │

 size=small
```

**Image B:**
```
Diamond  ──outside──▶  Circle

Diamond  ──size=large
```

**Semantic Network for B:**
```
        outside

Diamond ───────▶ Circle

   │

 size=large
```

**Transformation A → B:**
- Y: inside(X) → above(X)
- Y: size(small) → size(large)
- Relationship changed: inside → above
- Property changed: expanded

```
inside  →  outside

small   →  large
```
### Characteristics of Good Representations

A good knowledge representation exhibits these properties:

**1. Makes Relationships Explicit**
- All objects, properties, and relationships are clearly visible
- No hidden or implicit information
- Example: Semantic networks show "inside" and "outside" relationships explicitly

**2. Exposes Natural Constraints**
- Problem constraints become visible in the representation
- Makes illegal or impossible states obvious
- Helps guide problem-solving

**3. Right Level of Abstraction**
- Captures everything needed for the problem
- Removes unnecessary details
- Balance between completeness and simplicity

**4. Transparent and Concise**
- Easy to understand and interpret
- Captures only what's needed, nothing more
- Complete: Contains all necessary information

**5. Computationally Efficient**
- Fast processing due to appropriate abstraction
- No extraneous details to slow computation
- Enables real-time or near-real-time performance

**6. Computable**
- Allows drawing necessary inferences
- Supports the reasoning required for the problem
- Enables systematic problem-solving algorithms

**Key Insight:** "If you have the right knowledge representation, problem solving becomes very easy."

---

## 5. Guards and Prisoners Problem

### Problem Statement

A classic AI problem that illustrates semantic networks in action:

**Setup:**
- 3 guards and 3 prisoners on the left bank of a river
- Must all cross to the right bank
- One boat available, holds 1-2 people maximum
- Boat cannot travel alone (needs at least 1 person)

**Constraint:**
- Prisoners can never outnumber guards on either bank
- If prisoners outnumber guards, they will overpower them
- Prisoners won't run away if left alone
- But they will attack guards if they have numerical advantage

**Goal:** Find a sequence of boat trips that safely transports everyone to the right bank.

### Semantic Network Representation

**Node Structure:**
Each node represents a complete state of the problem:

```
State Node:
┌─────────────────────────────┐
│  Left Bank: G G P P         │
│  Boat Position: Right       │
│  Right Bank: G P            │
└─────────────────────────────┘
```

**Link Labels:**
Links between nodes show transitions (boat trips):
- Icons or descriptions of who moved
- Direction of travel (left → right or right → left)

**Example Transition:**
```
Initial State              After First Move
┌─────────────┐           ┌─────────────┐
│ L: GGG PPP  │ ──────→   │ L: GG PP    │
│ Boat: Left  │   Move     │ Boat: Right │
│ R: (empty)  │   G + P    │ R: G P      │
└─────────────┘           └─────────────┘
```

### Problem-Solving with Semantic Networks

The semantic network representation enables:

1. **Systematic State Generation**
   - From any state, generate all possible next states
   - Each boat trip creates a new state node

2. **Constraint Checking**
   - Immediately identify illegal states (prisoners > guards)
   - Remove illegal states from consideration

3. **Path Finding**
   - Search for path from initial state to goal state
   - Avoid cycles (returning to previously visited states)
   - Track productive vs. unproductive moves

4. **Solution Visualization**
   - The complete solution is a path through the network
   - Each node on the path represents a safe state
   - Each link represents a legal boat trip

**Sample Solution Sequence:**
```
(3G,3P,L) → (2G,2P,L)+(G,P,R) → (3G,2P,L)+(P,R) →
(3G,L)+(2P,R) → (3G,1P,L)+(P,R) → (1G,1P,L)+(2G,2P,R) →
(2G,2P,L)+(G,P,R) → (2G,1P,L)+(G,P,R) → (2G,L)+(G,3P,R) →
(3G,L)+(3P,R) → (1G,L)+(2G,3P,R) → (1G,1P,L)+(2G,2P,R) →
(R)+(3G,3P,R) ✓
```

---

## 6. Represent and Reason Paradigm

### Core Concept

The **represent and reason** paradigm is fundamental to all knowledge-based AI:

1. **Represent:** Create an explicit knowledge representation of the problem
2. **Reason:** Use that representation to solve the problem

This two-step approach underlies virtually all KBAI methods.

### Application to Ravens Problems

**Problem:** A is to B as C is to ?  (Choose from options 1-6)

**Step 1: Represent**
- Build semantic networks for A, B, and C
- Build semantic networks for each answer option (1-6)
- Identify transformations between related images

**Step 2: Reason**
- Compare transformation from A → B with transformation from C → each option
- Match transformations to find the best fit
- The option with the most similar transformation is the answer

**Example Reasoning:**
```
Transformation A → B:
  - Object moved: inside → above
  - Object expanded: small → large

Test C → Option 5:
  - Object moved: inside → above  ✓ Match
  - Object expanded: small → large ✓ Match

Option 5 is likely correct!
```

### Weighted Matching

Sometimes multiple answers partially match. Use weighted comparison:

**Criteria for Matching:**
1. **Exact matches:** Transformation is identical (highest weight)
2. **Partial matches:** Some aspects match, others differ (medium weight)
3. **Unchanged matches:** Property stays same in both (low weight)
4. **Mismatches:** Transformations are different (negative weight)

**Example:**
```
Option 2: 2 matching properties (weight +2)
Option 4: 1 matching property (weight +1)
Option 5: 3 matching properties (weight +3) ← Best choice
```

The option with the highest total weight is selected as the answer.

---

## 7. Cognitive Connections

### Semantic Networks and Human Cognition

**Connection 1: Knowledge Representation in Mind**
- Humans represent problems and knowledge mentally
- Mental representations enable problem-solving
- The form of representation affects solution ease
- KBAI insight: Representation is key to intelligence

**Connection 2: Spreading Activation Networks**
Semantic networks relate to spreading activation theory of human memory:

**Example: Story Understanding**
```
Story: "John wanted to become rich. He got a gun."

Your inference: John will rob a bank

How?
1. "Rich" node activates in memory
2. "Gun" node activates in memory
3. Activation spreads through connected concepts
4. Paths merge at "rob bank" concept
5. Nodes along the merged path become active
6. These activated concepts form your understanding
```

This explains how humans draw inferences from incomplete information - concepts activate related concepts through spreading activation.

**Connection 3: Structured Representations**
- Human memory is structured, not random
- Related concepts are connected
- Retrieval follows associative paths
- Semantic networks model this structure

---

## Summary

### Key Takeaways

1. **Knowledge-Based AI** focuses on human-like intelligence through structured knowledge representations, distinguished from other AI approaches by its cognitive orientation.

2. **Five AI Conundrums** define the fundamental challenges: limited resources vs. intractable problems, local computation vs. global constraints, deductive logic vs. abductive/inductive reasoning, limited knowledge vs. novel problems, and complexity of explanation.

3. **Cognitive Systems Architecture** has three layers:
   - Reaction (direct percept-action mapping)
   - Deliberation (reasoning + learning + memory, tightly integrated)
   - Metacognition (reasoning about reasoning)

4. **Semantic Networks** are structured knowledge representations using nodes (objects), links (relationships), and labels (semantics) that make knowledge explicit and support reasoning.

5. **Good Representations** are explicit, expose constraints, work at the right abstraction level, are transparent and concise, enable fast computation, and support necessary inferences.

6. **Represent and Reason** is the fundamental paradigm: create explicit representations, then reason over them to solve problems.

7. **Cognitive Connection:** KBAI methods mirror human cognition, particularly in knowledge representation and spreading activation in memory.

### Essential Principles

- Intelligence arises from the interaction of reasoning, learning, and memory
- The right representation makes problems easier to solve
- Knowledge-based AI explicitly captures and uses structured knowledge
- Human cognition provides both inspiration and validation for AI designs

---

## See Also

- [[02-Core-Reasoning-Strategies|Core Reasoning Strategies]] - Learn problem-solving methods like Generate & Test
- [[05-Language-and-Commonsense|Language and Common Sense]] - Explore frames, another knowledge representation
- [[00-README|Course Overview]] - Navigate the full course structure

---

*Knowledge representations are at the heart of KBAI. The representation you choose determines how easy or hard problem-solving becomes.*
