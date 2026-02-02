---
type: note
course: "[[Knowledge-Based AI (CS 7637)]]"
date: 2026-02-02
topic: Applied Problem Solving
lessons: 20-22
prerequisites:
  - "[[01-Fundamentals-KBAI|Fundamentals of KBAI]]"
  - "[[02-Core-Reasoning-Strategies|Core Reasoning Strategies]]"
  - "[[04-Logic-and-Planning|Logic and Planning]]"
---

# Applied Problem Solving: Constraint Propagation, Configuration, and Diagnosis

## Prerequisites

- Understanding of semantic networks and knowledge representations
- Familiarity with problem-solving methods
- Knowledge of planning and state spaces
- Completed Fundamentals and Core Reasoning modules

## Learning Goals

After completing this module, you will be able to:

1. Apply constraint propagation to visual reasoning and interpretation
2. Use constraints to reduce ambiguity in under-determined problems
3. Design configuration systems through constraint satisfaction
4. Understand diagnosis as abductive reasoning
5. Apply hypothesis generation and testing for diagnostic problems
6. Recognize connections between classification, planning, and diagnosis

---

## 1. Constraint Propagation

### Core Concept

**Constraint propagation** uses local constraints to determine global properties, resolving ambiguities through iterative constraint satisfaction.

**Key Idea:** Information from one part of a problem constrains possibilities in other parts. Propagate these constraints until solution emerges.

### The 3D Figure Recognition Problem

**Input: 2D Line Drawing**
```
      A
     / \
    /   \
   B─────C
   │     │
   │     │
   D─────E
```

**Task:** Interpret as 3D object (cube? pyramid? ambiguous polyhedron?)

**Challenge:** 2D projection is ambiguous
- Same drawing could represent multiple 3D objects
- Need additional information to resolve ambiguity

### Edge and Vertex Labels

**Huffman-Clowes Labeling:**

**Edge Types:**
```
+ : Convex edge (mountain)
- : Concave edge (valley)
→ : Occluding edge (one surface hides another)
```

**Vertex Types (Junctions):**
```
Y-junction: 3 edges meet
L-junction: 2 edges meet (90°)
T-junction: 3 edges meet (one terminates)
Arrow: 3 edges meet (arrow shape)
```

**Constraint:** Each junction can only have certain legal edge label combinations

### Legal Labelings for Junctions

**Y-Junction (3 convex edges meet):**
```
Possible labelings:
1. All three edges +  (convex corner pointing out)
2. All three edges -  (concave corner pointing in)
3. Two + and one → (corner with occlusion)
[Other combinations physically impossible for solid objects]
```

**L-Junction (two edges, 90°):**
```
Possible labelings:
1. Both edges +  (convex edge)
2. Both edges -  (concave edge)
3. One +, one → (edge with occlusion)
4. One -, one → (occluded concave edge)
```

**T-Junction (one edge terminates):**
```
Always has structure:
  │  (vertical edge)
──┼── (horizontal bar occludes vertical)

One edge must be → (occlusion boundary)
```

### Constraint Propagation Process

**Step 1: Initialize**
```
For each edge: List all possible labels (+, -, →)
For each vertex: List all legal junction labelings
```

**Step 2: Propagate Constraints**
```
For each vertex:
  Remove labelings inconsistent with adjacent edges

For each edge:
  Remove labels inconsistent with both endpoint vertices

Iterate until no more changes
```

**Step 3: Solution**
```
If unique labeling remains: 3D interpretation found!
If multiple labelings remain: Still ambiguous
If no labelings remain: Impossible object (like Penrose triangle)
```

### Example: Cube Recognition

**Input Drawing:**
```
    A───B
   /│  /│
  / │ / │
 D───C  │
 │  F──┤G
 │ /   │/
 E─────H
```

**Initial (all edges unknown):**
```
Each edge: {+, -, →}
```

**Apply Constraints:**

**Vertex A (Y-junction):**
```
Edges: A-B, A-D, A-F
Legal combinations for Y-junction:
- All convex (+, +, +)
- All concave (-, -, -)
- Mixed with occlusion

Assume we're viewing convex cube:
A-B: +
A-D: +
A-F: +
```

**Propagate to Vertex B (Y-junction):**
```
Edge A-B already labeled: +
Edges: B-A, B-C, B-G
If B-A is +, then B-C and B-G must be + (convex corner)

B-C: +
B-G: +
```

**Continue propagation...**

**Final Labeling:**
```
All visible edges: + (convex)
All hidden edges behind: → (occluded)

Interpretation: Cube with front-top-right vertex at A
```

### Resolving Ambiguity

**Ambiguous Drawing:**
```
Necker Cube (can flip interpretation)

    ┌───┐
   /│  /│
  / │ / │
 └───┘  │
 │  └───┘
 │ /   /
 └───┘
```

**Two Valid Interpretations:**

**Interpretation 1:**
```
Front face: top-left
Back face: bottom-right
(cube appears to extend back-right)
```

**Interpretation 2:**
```
Front face: bottom-right
Back face: top-left
(cube appears to extend forward-left)
```

**Constraint propagation finds BOTH interpretations**
- Cannot resolve without additional information
- Human vision also flips between these!

### Gibberish Sentences and Constraint Propagation

**Similar principle applies to language:**

**Input:** "Colorless green ideas sleep furiously"

**Constraints:**
- Syntactic: Grammatically correct (noun phrase + verb phrase)
- Semantic: Semantically anomalous (contradictions)

**Propagation:**
```
"colorless green" → Contradiction (can't be both)
"ideas sleep" → Category error (abstract ≠ biological)
"sleep furiously" → Modifier mismatch (manner ≠ state)

Constraints identify this as semantically impossible
```

**Well-formed:** "Colorful green parrots sleep quietly"
```
All constraints satisfied:
- Syntax: ✓
- Semantics: ✓
- Pragmatics: ✓
```

### Applications Beyond Vision

**Sudoku:**
```
Constraints:
- Each row contains 1-9 (no repeats)
- Each column contains 1-9 (no repeats)
- Each 3×3 box contains 1-9 (no repeats)

Propagation:
Fill one cell → Constraints propagate to row, column, box
→ Eliminates possibilities in related cells
→ Iteratively solve
```

**Map Coloring:**
```
Constraints:
- Adjacent regions must have different colors
- Minimize total colors used

Propagation:
Color one region → Adjacent regions can't use that color
→ Propagates through entire map
```

**Scheduling:**
```
Constraints:
- No person in two places at once
- Required sequences (A before B)
- Resource limitations

Propagation:
Schedule one event → Constrains available times
→ Narrows possibilities for dependent events
```

---

## 2. Configuration

### What is Configuration?

**Configuration** is the design task of arranging components to satisfy requirements and constraints.

**Examples:**
- Computer system configuration (CPU, RAM, storage, etc.)
- Office layout configuration
- Network topology configuration
- Manufacturing workflow configuration

**Key Properties:**
- Fixed set of component types
- Known constraints between components
- Goal: Find valid arrangement
- Multiple solutions often exist

### Configuration vs. Design

**Design:**
- Create new components
- Invent new structures
- Open-ended creativity
- Few constraints initially

**Configuration:**
- Select from existing components
- Arrange within constraints
- Well-defined problem space
- Many constraints throughout

**Configuration is constrained design** - a special case.

### The Basement Configuration Problem

**Task:** Configure a basement with multiple purposes

**Requirements:**
```
- Home office (needs quiet, lighting, power)
- Gym equipment (needs space, floor reinforcement)
- Storage area (needs accessibility)
- Workshop (needs ventilation, power)
```

**Components Available:**
```
- Rooms/partitions
- Windows (light + noise)
- Doors (access + sound transmission)
- Electrical outlets
- Ventilation ducts
- Flooring types
```

**Constraints:**
```
- Load-bearing walls cannot be removed
- Plumbing locations fixed
- Total area fixed
- Budget constraints
- Building codes
```

### The Configuration Process

**Step 1: Specify Requirements**
```
Office: Area ≥ 100 sq ft, Quiet ≥ 7/10, Light ≥ 8/10, Power ≥ 4 outlets
Gym: Area ≥ 150 sq ft, Floor-strength ≥ high, Ceiling-height ≥ 8 ft
Storage: Area ≥ 80 sq ft, Access = easy
Workshop: Area ≥ 120 sq ft, Ventilation ≥ good, Power ≥ 6 outlets
```

**Step 2: Initialize Variables with Ranges**
```
Office-area: [0, total-area]
Office-quiet: [0, 10]
Office-light: [0, 10]
...
```

**Step 3: Apply Constraints**
```
Constraint: Office-area + Gym-area + Storage-area + Workshop-area ≤ Total-area

Propagate:
If Office-area = 100 sq ft
And Gym-area = 150 sq ft
And Total-area = 500 sq ft
Then: Storage-area + Workshop-area ≤ 250 sq ft

Refine ranges:
Storage-area: [80, 250]
Workshop-area: [120, 250]
```

**Step 4: Refine Iteratively**
```
Apply constraint: Office requires quiet → Place away from gym/workshop
Propagate: Office-location = {north-corner} (furthest from noisy areas)

Apply constraint: Workshop needs ventilation → Near window/duct
Propagate: Workshop-location = {east-side} (has existing window)

Apply constraint: Gym needs floor-strength → Concrete floor area
Propagate: Gym-location = {south-section} (has concrete)

Continue until all constraints satisfied...
```

**Step 5: Verify Solution**
```
Check all requirements met:
✓ Office: 110 sq ft, quiet, well-lit, 4 outlets
✓ Gym: 160 sq ft, reinforced floor, 9 ft ceiling
✓ Storage: 90 sq ft, direct access from stairs
✓ Workshop: 130 sq ft, ventilated, 6 outlets

Check all constraints satisfied:
✓ Total area = 490 ≤ 500 sq ft
✓ No conflicting placements
✓ Building codes met
✓ Budget not exceeded
```

### Connection to Other Methods

**Configuration as Classification:**
```
Task: Classify each component into location/configuration category

Example:
- Classify office location: {north-corner, south-corner, ...}
- Classify gym flooring: {concrete, reinforced-wood, ...}
- Classify outlet placement: {wall-1, wall-2, ...}

Use classification methods (concept hierarchies, prototypes)
```

**Configuration as Planning:**
```
Initial state: Empty basement
Goal state: Configured basement meeting requirements
Operators: Place component, add feature, install utility

Planning finds sequence:
1. Partition into rooms
2. Install reinforced flooring for gym
3. Add electrical outlets
4. Install ventilation
5. Place doors and windows
```

**Configuration as Constraint Satisfaction:**
```
Variables: Location, size, features of each component
Domains: Possible values for each variable
Constraints: Requirements and limitations

Use constraint propagation to narrow domains
Use search to find consistent assignment
```

### Advantages of Configuration Approach

**1. Systematic:**
- All constraints explicitly represented
- Guaranteed to satisfy requirements (if solution exists)
- Can prove no solution exists

**2. Flexible:**
- Easy to add/remove constraints
- Can handle alternative components
- Can optimize (cost, space, efficiency)

**3. Explainable:**
- Can trace why decisions made
- Can explain constraint violations
- Can suggest modifications if over-constrained

---

## 3. Diagnosis

### Diagnosis as a Task

**Diagnosis** identifies the cause of observed symptoms or failures.

**Examples:**
- Medical diagnosis (symptoms → disease)
- Mechanical diagnosis (malfunction → broken part)
- Software debugging (errors → faulty code)
- Network troubleshooting (connectivity issues → problem node)

### Data Space vs. Hypothesis Space

**Data Space:** Observations, symptoms, measurements
```
Patient symptoms:
- Fever: 102°F
- Cough: Yes
- Fatigue: High
- Sore throat: Yes
```

**Hypothesis Space:** Possible explanations
```
Possible diagnoses:
- Flu
- Cold
- COVID-19
- Strep throat
- Pneumonia
```

**Diagnosis bridges the two spaces:** Data → Hypothesis

### Problems with Diagnosis as Classification

**Classification assumes:** One-to-one or many-to-one mapping
```
Features → Category
Symptoms → Disease
```

**Problems:**

**1. Multiple Causes:**
```
Fever can be caused by:
- Flu, cold, COVID, strep, pneumonia, appendicitis, ...

Symptoms don't uniquely determine cause
```

**2. Incomplete Data:**
```
Observed: Fever, cough
Not observed: Many other possible symptoms

Cannot definitively classify with partial information
```

**3. Multiple Simultaneous Issues:**
```
Patient could have:
- Flu AND strep (two diseases)
- Flu AND broken thermometer (disease + measurement error)

Classification assumes single category
```

**4. Novel Problems:**
```
New disease (COVID in early 2020)
Not in training data
Classification fails

But diagnosis can reason about mechanisms
```

### Diagnosis as Abduction

**Three Types of Inference:**

**Deduction (Certain):**
```
Rule: All birds have feathers
Fact: Tweety is a bird
Conclusion: Tweety has feathers (certain)
```

**Induction (Generalization):**
```
Observations: Tweety (bird) has feathers
              Robins (birds) have feathers
              Eagles (birds) have feathers
Generalization: All birds have feathers (probable)
```

**Abduction (Hypothesis Generation):**
```
Observation: Tweety has feathers
Rule: Birds have feathers
Hypothesis: Tweety is a bird (possible explanation)

Not certain! (Tweety could be costume, plucked from bird, etc.)
```

**Diagnosis is Abduction:**
```
Observation: Patient has fever, cough
Rule: Flu causes fever and cough
Hypothesis: Patient has flu (possible explanation)

Must evaluate among competing hypotheses
```

### Criteria for Choosing Hypotheses

**Multiple hypotheses often explain data. How to choose?**

**Criterion 1: Coverage**
```
How many symptoms does hypothesis explain?

Flu explains: Fever, cough, fatigue ✓✓✓
Cold explains: Cough, fatigue ✓✓ (not high fever)

Flu has better coverage
```

**Criterion 2: Parsimony (Occam's Razor)**
```
Prefer simpler explanations (fewer causes)

Hypothesis 1: Patient has flu (single cause)
Hypothesis 2: Patient has cold + food poisoning + insomnia (three causes)

Both explain symptoms, but Hypothesis 1 simpler
```

**Criterion 3: Probability**
```
How likely is this hypothesis?

Flu: Common, 10% of population in flu season
Rare tropical disease: 0.001% of population

Even if both explain symptoms, flu more probable
```

**Criterion 4: Testability**
```
Can hypothesis be confirmed/refuted?

Flu: Test with rapid flu test (verifiable)
Vague "imbalance": No test available (not testable)

Prefer testable hypotheses
```

**Criterion 5: Consistency with Prior Knowledge**
```
Does hypothesis fit with what we know?

Patient recently traveled to flu-endemic area: ✓ Consistent
Patient in isolated cabin for 6 months: ✗ Inconsistent

Context matters
```

### The Diagnostic Process

**Step 1: Gather Data**
```
Symptoms, test results, observations
Create complete picture of problem
```

**Step 2: Generate Hypotheses**
```
Use causal models:
"What could cause these symptoms?"

Medical: Use disease models
Mechanical: Use component failure models
Software: Use bug pattern knowledge

Generate multiple candidate hypotheses
```

**Step 3: Evaluate Hypotheses**
```
Apply criteria:
- Coverage: How much data explained?
- Parsimony: How simple?
- Probability: How likely?
- Testability: Can we verify?
- Consistency: Fits context?

Rank hypotheses by these criteria
```

**Step 4: Test Hypotheses**
```
Design discriminating tests:
- Tests that differentiate top hypotheses
- Tests that confirm/refute predictions

Execute tests, gather new data
```

**Step 5: Refine**
```
New data → Update hypothesis probabilities
Eliminate refuted hypotheses
Generate new hypotheses if needed
Iterate until confident diagnosis
```

### Example: Car Won't Start

**Data:**
```
- Engine won't turn over
- Lights work
- Starter makes clicking sound
```

**Generate Hypotheses:**
```
H1: Dead battery
H2: Faulty starter motor
H3: Bad alternator (battery drained)
H4: Loose/corroded battery cables
H5: Out of gas (unlikely given symptoms)
```

**Evaluate:**

```
H1 (Dead battery):
- Coverage: ✓✓ (explains clicking, no turnover)
- But: Lights work (battery has some charge)
- Probability: High
- Parsimony: Simple (single component)

H2 (Faulty starter):
- Coverage: ✓✓✓ (explains all symptoms perfectly)
- Probability: Moderate
- Parsimony: Simple

H3 (Bad alternator):
- Coverage: ✓ (explains dead battery)
- But: Requires two failures (alternator + drained battery)
- Parsimony: Less simple

H4 (Loose cables):
- Coverage: ✓✓✓ (poor connection → no power to starter)
- Probability: Moderate
- Parsimony: Simple
- Testability: Easy to check!
```

**Test:**
```
Action: Tighten and clean battery cables

Result: Car starts! ✓

Diagnosis: H4 (loose cables) confirmed
```

**Learning:**
```
Store case:
- Symptoms: No start, clicking, lights work
- Cause: Loose battery cables
- Solution: Clean and tighten cables

Future similar cases → Retrieve this diagnosis
```

---

## 4. Integration and Connections

### Unified View of Configuration and Diagnosis

**Configuration: Synthesis**
```
Requirements → Design → Components + Arrangement
Forward reasoning: Given goals, create solution
```

**Diagnosis: Analysis**
```
Symptoms ← Problem ← Failed Components
Backward reasoning: Given symptoms, find causes
```

**Relationship:**
```
Configuration and Diagnosis are inverses!

Configuration: Goals → Design
Diagnosis: Symptoms ← Failure ← Design

Understanding how things work (configuration)
helps diagnose how they break (diagnosis)
```

### Connection to Planning

**Planning:**
```
Initial state → Operators → Goal state
Find sequence of actions to achieve goal
```

**Configuration:**
```
Empty space + Requirements → Components + Constraints → Valid configuration
Find arrangement that satisfies constraints
```

**Diagnosis:**
```
Normal function + Failure modes → Symptoms
Find failure that explains symptoms
Inverse of planning (what went wrong?)
```

### Connection to Classification

**Classification:**
```
Features → Category
Assign to predefined class
```

**Configuration:**
```
Requirements → Component Selection (classify components)
→ Arrangement (spatial classification)
```

**Diagnosis:**
```
Symptoms → Disease/Fault (classify problem)
But more complex: Abduction, not just pattern matching
```

### Constraint Propagation Across Tasks

**Vision (Line Labeling):**
```
2D lines + Legal 3D junctions → Propagate → 3D interpretation
```

**Configuration:**
```
Components + Requirements → Propagate → Valid arrangement
```

**Diagnosis:**
```
Symptoms + Causal models → Propagate → Likely causes
```

**Common pattern:**
- Local constraints
- Propagation to global solution
- Iterative refinement

---

## Summary

### Key Takeaways

1. **Constraint Propagation** uses local constraints (legal junction labelings) to determine global properties (3D object interpretation). Iteratively propagates constraints until ambiguity resolved or multiple interpretations identified.

2. **Visual Reasoning** through Huffman-Clowes labeling: Each edge labeled (+, -, →), each junction has legal label combinations, propagation finds consistent global labeling, resolves ambiguous 2D drawings into 3D interpretations.

3. **Configuration** is constrained design: Select from existing components, arrange to satisfy requirements, apply constraints iteratively, refine ranges until valid solution found. Related to classification, planning, and constraint satisfaction.

4. **Diagnosis as Abduction** generates hypotheses to explain observations. Not deduction (certain) or induction (generalization), but hypothesis generation (possible explanations). Multiple hypotheses often explain data.

5. **Hypothesis Selection Criteria:** Coverage (explains symptoms), parsimony (simpler better), probability (more likely better), testability (can verify), consistency (fits context). Combine criteria to rank hypotheses.

6. **Integration:** Configuration (synthesis) and diagnosis (analysis) are inverse tasks. Both use constraints. Planning relates to configuration. Classification underlies parts of all three. Constraint propagation is common pattern.

### Essential Principles

- **Constraints reduce ambiguity:** Local rules → Global conclusions
- **Propagation is iterative:** Apply constraints repeatedly until convergence
- **Multiple solutions common:** Constraints may not uniquely determine answer
- **Abduction generates hypotheses:** Work backward from observations to causes
- **Parsimony guides selection:** Prefer simpler explanations (Occam's Razor)
- **Testing refines diagnosis:** Discriminating tests distinguish hypotheses

### Task Comparison

| Task | Direction | Reasoning | Constraints | Output |
|------|-----------|-----------|-------------|--------|
| Configuration | Forward | Synthesis | Requirements | Design |
| Diagnosis | Backward | Analysis | Causal models | Cause |
| Planning | Forward | Search | State/operators | Action sequence |
| Classification | Forward | Recognition | Features | Category |

---

## See Also

- [[02-Core-Reasoning-Strategies|Core Reasoning]] - Problem-solving methods underlying these tasks
- [[04-Logic-and-Planning|Logic & Planning]] - Planning connections to configuration
- [[03-Learning-Methods|Learning Methods]] - Classification underlying diagnosis
- [[00-README|Course Overview]] - Navigate the full course structure

---

*Applied problem solving demonstrates how fundamental KBAI concepts—knowledge representation, constraint satisfaction, reasoning—combine to address real-world tasks like interpretation, design, and diagnosis.*
