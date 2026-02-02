---
type: index
course: "[[Knowledge-Based AI (CS 7637)]]"
date: 2026-02-02
---

# Knowledge-Based AI (CS 7637) - Course Navigation

## Overview

Knowledge-Based AI (KBAI) is the study of artificial intelligence systems that use structured knowledge representations to solve problems through human-like reasoning, learning, and memory. This course explores cognitive systems that exhibit human-level intelligence through the interaction of these three fundamental components.

**Core Philosophy:** KBAI represents a unified theory where reasoning, learning, and memory are intimately connected. We learn so we can reason; the results of reasoning lead to additional learning; and memory provides the knowledge foundation for both.

## Course Organization

This course is organized into 9 comprehensive modules covering the fundamental topics, reasoning strategies, learning methods, and advanced AI concepts.

### Learning Path

```
Fundamentals → Core Reasoning → Learning Methods → Logic & Planning
     ↓              ↓                 ↓                    ↓
Language & Common Sense → Advanced Reasoning → Applied Problem Solving
     ↓                          ↓                          ↓
          Metacognition & Advanced Topics
```

## Module Guide

### 1. [[01-Fundamentals-KBAI|Fundamentals of KBAI]]
**Lessons 1-3** | Introduction, Cognitive Systems, Semantic Networks

Core concepts: Knowledge representations, cognitive architectures, semantic networks as knowledge structures, characteristics of AI agents and problems.

**Start here** if you're new to KBAI or want to understand the foundational principles.

**Key Topics:**
- Four schools of AI: Acting/Thinking, Human-like/Optimally
- Cognitive system architecture: Reaction, Deliberation, Metacognition
- Semantic networks: Nodes, links, and structured representations
- Guards & Prisoners problem

### 2. [[02-Core-Reasoning-Strategies|Core Reasoning Strategies]]
**Lessons 4-6** | Generate & Test, Means-Ends Analysis, Production Systems

Problem-solving methods that form the basis of AI reasoning. Learn how to map percepts to actions through systematic approaches.

**Start here** if you understand representations and want to learn reasoning methods.

**Key Topics:**
- Generate & Test: Smart generators vs. smart testers
- Means-Ends Analysis: Goal-driven problem solving
- Problem Reduction: Decomposing complex problems
- Production Systems: Cognitive architectures with rules
- Chunking: Learning from impasses

### 3. [[03-Learning-Methods|Learning Methods]]
**Lessons 8-11, 19, 23** | Case-based Reasoning, Classification, Version Spaces

How AI agents learn from experience through recording cases, concept learning, and systematic generalization/specialization.

**Start here** if you want to understand learning mechanisms in KBAI.

**Key Topics:**
- Learning by Recording Cases: k-nearest neighbor
- Case-Based Reasoning: Retrieve, adapt, evaluate, store
- Incremental Concept Learning: Variabilization, specialization, generalization
- Classification: Prototype vs. exemplar concepts
- Version Spaces: Hypothesis refinement
- Learning by Correcting Mistakes: Error detection and repair

### 4. [[04-Logic-and-Planning|Logic and Planning]]
**Lessons 12-13** | Formal Logic, Planning Strategies

Logical representations and planning methods for goal achievement using operators and state spaces.

**Start here** for systematic approaches to goal-driven problem solving.

**Key Topics:**
- Predicate logic: Conjunctions, disjunctions, implications
- Truth tables and logical equivalences
- Rules of inference: Modus ponens, resolution
- Partial order planning: Open preconditions, conflict detection
- State space search

### 5. [[05-Language-and-Commonsense|Language and Common Sense Reasoning]]
**Lessons 7, 14-16** | Frames, Understanding, Scripts

How AI systems understand language and reason about everyday situations using structured knowledge.

**Start here** for natural language understanding and common sense reasoning.

**Key Topics:**
- Frames: Slots, fillers, default values, stereotypes
- Thematic role systems: Agent, object, location, time
- Common sense reasoning: Primitive actions, implied actions
- Scripts: Stereotypical event sequences with tracks
- Story understanding

### 6. [[06-Advanced-Reasoning|Advanced Reasoning]]
**Lessons 17-18** | Explanation-Based Learning, Analogical Reasoning

Advanced learning and reasoning methods including cross-domain analogy and explanation-driven learning.

**Start here** for sophisticated reasoning techniques.

**Key Topics:**
- Explanation-Based Learning: Abstraction and transfer
- Analogical Reasoning: Retrieval, mapping, transfer, evaluation
- Spectrum of similarity: Semantic, pragmatic, structural
- Design by analogy
- Cross-domain analogy

### 7. [[07-Applied-Problem-Solving|Applied Problem Solving]]
**Lessons 20-22** | Constraint Propagation, Configuration, Diagnosis

Practical AI applications using constraints, configuration, and diagnostic reasoning.

**Start here** for real-world AI applications.

**Key Topics:**
- Constraint propagation: Visual reasoning from 2D to 3D
- Configuration: Design through constraint satisfaction
- Diagnosis as abduction: Hypothesis generation and testing
- Connection to classification and planning

### 8. [[08-Metacognition-and-Advanced|Metacognition and Advanced Topics]]
**Lessons 24-26** | Meta-reasoning, Visuospatial Reasoning, Design & Creativity

Advanced topics including reasoning about reasoning, visual problem solving, and creative AI systems.

**Start here** for cutting-edge KBAI topics.

**Key Topics:**
- Meta-reasoning: Reasoning about deliberation and reaction
- Strategy selection and knowledge gaps
- Visuospatial reasoning: Ravens Progressive Matrices
- Design and creativity
- Symbol grounding problem
- Systems thinking

## Key Projects & Examples

Throughout the course, several recurring examples illustrate KBAI concepts:

### Ravens Progressive Matrices
The primary project: Building AI agents that solve visual analogy problems from intelligence tests. This project integrates knowledge representation, reasoning, and learning.

**Complexity progression:**
- 2×1 matrices: Simple transformations
- 2×2 matrices: Pattern completion
- 3×3 matrices: Complex rule systems

### Classic AI Problems
- **Guards & Prisoners:** State space search, semantic networks
- **Blocks World:** Means-ends analysis, planning, problem reduction
- **Baseball Pitcher:** Production systems, action selection, chunking

## Recommended Learning Paths

### Path 1: Foundation-First (Recommended for beginners)
1. Fundamentals → 2. Core Reasoning → 3. Learning Methods → 4. Logic & Planning → 5. Language & Common Sense → 6. Advanced Reasoning → 7. Applied → 8. Metacognition

### Path 2: Problem-Solving Focus
1. Fundamentals → 2. Core Reasoning → 4. Logic & Planning → 7. Applied → 3. Learning Methods → 6. Advanced Reasoning

### Path 3: Learning-Centric
1. Fundamentals → 3. Learning Methods → 6. Advanced Reasoning → 2. Core Reasoning → 4. Logic & Planning

### Path 4: Applications-First
1. Fundamentals → 7. Applied → 2. Core Reasoning → 5. Language & Common Sense → 3. Learning Methods

## Unifying Principles

Seven key principles run throughout this course:

1. **Knowledge Representations** are central to KBAI
2. **Reasoning, Learning, Memory** are intimately connected
3. **Cognitive architectures** separate content from behavior
4. **Analogy and abstraction** enable transfer and generalization
5. **Generate and test** underlies many AI methods
6. **Meta-reasoning** enables self-improvement
7. **Human cognition** inspires and validates AI design

## Core Concepts Cross-Reference

| Concept | Primary Module | Also Appears In |
|---------|---------------|-----------------|
| Semantic Networks | 1. Fundamentals | 5. Language, 6. Advanced |
| Generate & Test | 2. Core Reasoning | 3. Learning, 7. Applied |
| Production Systems | 2. Core Reasoning | 5. Language |
| Frames | 5. Language | 1. Fundamentals, 2. Core |
| Case-Based Reasoning | 3. Learning | 6. Advanced, 7. Applied |
| Planning | 4. Logic & Planning | 2. Core, 7. Applied |
| Chunking | 2. Core Reasoning | 3. Learning |
| Analogical Reasoning | 6. Advanced | 3. Learning, 8. Metacognition |
| Constraint Propagation | 7. Applied | 8. Metacognition |
| Meta-reasoning | 8. Metacognition | Throughout course |

## Study Tips

1. **Follow the cognitive connection sections** - They link AI techniques to human cognition
2. **Practice with Ravens problems** - The project integrates all concepts
3. **Compare knowledge representations** - Understand when to use semantic networks vs. frames vs. production rules
4. **Connect reasoning and learning** - Notice how reasoning drives what/when/why to learn
5. **Build incrementally** - Start with simple problems, add complexity gradually

## Prerequisites

- Basic programming skills (for projects)
- Familiarity with data structures
- Logical thinking and problem-solving ability
- Interest in cognitive science helpful but not required

## Course Resources

- **Main Project:** Raven's Progressive Matrices AI agent
- **Classic Problems:** Guards & Prisoners, Blocks World, Baseball Pitcher
- **Key Readings:** Spread throughout lessons on cognitive architectures, chunking, scripts, analogical reasoning

## See Also

- [[01-Fundamentals-KBAI|Start with Fundamentals]]
- [[02-Core-Reasoning-Strategies|Begin Problem Solving]]
- [[03-Learning-Methods|Explore Learning]]
- [[08-Metacognition-and-Advanced|Jump to Advanced Topics]]

---

*This course explores how knowledge-based AI can achieve human-level intelligence through the unified interaction of reasoning, learning, and memory.*
