---
type: note
course: "[[Knowledge-Based AI (CS 7637)]]"
date: 2026-02-02
topic: Learning Methods
lessons: 8-11, 19, 23
prerequisites:
  - "[[01-Fundamentals-KBAI|Fundamentals of KBAI]]"
  - "[[02-Core-Reasoning-Strategies|Core Reasoning Strategies]]"
  - Understanding of knowledge representations
---

# Learning Methods in Knowledge-Based AI

## Prerequisites

- Understanding of semantic networks and frames
- Familiarity with production systems and chunking
- Knowledge of problem-solving strategies
- Completed Fundamentals and Core Reasoning modules

## Learning Goals

After completing this module, you will be able to:

1. Implement k-nearest neighbor learning for case-based reasoning
2. Design complete case-based reasoning systems with retrieval, adaptation, evaluation, and storage
3. Apply incremental concept learning through variabilization, specialization, and generalization
4. Classify objects using prototype and exemplar concepts
5. Use version spaces for hypothesis refinement
6. Implement learning by correcting mistakes with error detection and explanation

---

## 1. Learning by Recording Cases

### Core Concept

**Learning by recording cases** is the simplest form of learning: store experiences and retrieve similar ones when facing new problems.

**Key Principle:** Past experiences (cases) are valuable for solving new, similar problems.

### The Blocks World Example

**Setup:** Six colored blocks with varying properties

```
Block 1: Red, Square, Large
Block 2: Blue, Circle, Small
Block 3: Green, Triangle, Medium
Block 4: Red, Circle, Large
Block 5: Blue, Square, Small
Block 6: Green, Circle, Medium
```

**Problem:** Given a new block with width=0.8 and height=0.8, what color is it?

### Nearest Neighbor Method

**Step 1: Represent Cases**
Plot existing cases in feature space:

```
    Height
      ↑
  1.0 │   B1●(R)
      │
  0.8 │        B4●(R)
      │
  0.6 │             B6●(G)
      │
  0.4 │   B5●(B)      B3●(G)
      │
  0.2 │        B2●(B)
      │
  0.0 └─────────────────────→ Width
      0   0.2  0.4  0.6  0.8  1.0
```

**Step 2: Place New Problem**
```
New block: (width=0.8, height=0.8) = Q
```

**Step 3: Find Nearest Neighbor**
Calculate Euclidean distance from Q to each case:

```
Distance to B1: √[(0.8-0.3)² + (0.8-1.0)²] = 0.54
Distance to B2: √[(0.8-0.5)² + (0.8-0.2)²] = 0.67
Distance to B3: √[(0.8-0.7)² + (0.8-0.4)²] = 0.41
Distance to B4: √[(0.8-0.8)² + (0.8-0.8)²] = 0.00 ← Nearest!
Distance to B5: √[(0.8-0.3)² + (0.8-0.4)²] = 0.64
Distance to B6: √[(0.8-0.7)² + (0.8-0.6)²] = 0.22
```

**Step 4: Transfer Property**
```
Nearest neighbor: B4 (Red, Circle, Large)
Therefore, new block is: RED
```

### Distance Metrics

**Euclidean Distance (L2 norm):**
```
d = √[(x₁-x₂)² + (y₁-y₂)² + ... + (zₙ-zₙ)²]
```

**Manhattan Distance (L1 norm):**
```
d = |x₁-x₂| + |y₁-y₂| + ... + |zₙ-zₙ|
```

**Choice depends on problem domain:**
- Euclidean: "As the crow flies" distance
- Manhattan: "City block" distance
- Others: Cosine similarity, Mahalanobis distance

### k-Nearest Neighbors (k-NN)

Instead of using just the nearest neighbor, use k nearest neighbors:

**Algorithm:**
1. Find k nearest cases to the new problem
2. Have them "vote" on the answer
3. Use majority vote or weighted average

**Example with k=3:**
```
New block at (0.7, 0.6):
- 1st nearest: B6 (Green) - distance 0.1
- 2nd nearest: B3 (Green) - distance 0.2
- 3rd nearest: B4 (Red) - distance 0.25

Vote: Green=2, Red=1
Prediction: GREEN
```

**Advantages of k-NN:**
- More robust to outliers
- Considers local neighborhood
- Can weight by distance (closer cases have more influence)

**Disadvantage:**
- Must choose k appropriately
- Too small k: Sensitive to noise
- Too large k: Includes irrelevant cases

### Multi-Dimensional Spaces

Real-world problems rarely have just 2 dimensions:

**Example: Route Planning**
```
Features:
- Origin location (x₁, y₁)
- Destination location (x₂, y₂)
- Time of day
- Day of week
- Weather conditions
- Traffic density
...potentially 10+ dimensions
```

**Challenge:** Distance calculation in high-dimensional space

**Solution:**
```
d = √[w₁(x₁-x₁')² + w₂(y₁-y₁')² + ... + wₙ(xₙ-xₙ')²]

where wᵢ = weight for dimension i
```

**Feature Weighting:**
- Not all dimensions equally important
- Learn weights from data or use domain knowledge
- Example: Destination location more important than weather

### Application to Medical Diagnosis

**Cases:** Past patients with symptoms and diagnoses

```
Patient 1: Fever=102°F, Cough=Yes, Fatigue=High → Flu
Patient 2: Fever=99°F, Cough=No, Fatigue=Low → Cold
Patient 3: Fever=103°F, Cough=Yes, Fatigue=High → Flu
...
```

**New Patient:** Fever=101°F, Cough=Yes, Fatigue=Medium

**Nearest Neighbors:**
1. Patient 1 (Flu) - small distance
2. Patient 3 (Flu) - small distance

**Diagnosis:** Likely Flu (majority vote)

**Note:** In real medical diagnosis, much more sophisticated than simple k-NN, but illustrates the principle.

---

## 2. Case-Based Reasoning

### From Recording Cases to CBR

**Learning by recording cases** focuses on storage and retrieval.

**Case-Based Reasoning (CBR)** is a complete methodology:
1. **Retrieve:** Find similar past cases
2. **Adapt:** Modify past solution for new problem
3. **Evaluate:** Test adapted solution
4. **Store:** Save new case for future use

### The CBR Cycle

```
        ┌─────────────┐
        │  New Problem│
        └──────┬──────┘
               ↓
        ┌─────────────┐
        │  RETRIEVE   │ ← Find similar cases
        │  similar    │   from memory
        │  cases      │
        └──────┬──────┘
               ↓
        ┌─────────────┐
        │   ADAPT     │ ← Modify solution
        │   solution  │   for current problem
        └──────┬──────┘
               ↓
        ┌─────────────┐
        │  EVALUATE   │ ← Test solution
        │  solution   │   (success? failure?)
        └──────┬──────┘
               ↓
        ┌─────────────┐
        │   STORE     │ ← Save new case
        │   new case  │   Update memory
        └─────────────┘
```

### Assumptions of CBR

Five key assumptions underlie case-based reasoning:

**1. Patterns exist in the world**
- Similar problems have similar solutions
- Past experiences are relevant to future situations

**2. Similar problems have similar solutions**
- Can transfer solutions across similar contexts
- Similarity in problem → Similarity in solution

**3. The world is largely predictable**
- Patterns repeat
- Past is generally indicative of future

**4. Solutions are often reusable**
- Don't need to solve from scratch every time
- Adaptation is cheaper than creation

**5. Knowledge is in the form of cases**
- Concrete experiences rather than abstract rules
- Episodic memory is primary knowledge source

### Case Adaptation

**Simple Adaptation: Parameter Substitution**

```
Old Case: Restaurant A
- Location: Downtown
- Cuisine: Italian
- Price: $$
- Parking: Street parking
- Solution: Take taxi (avoid parking hassle)

New Problem: Restaurant B
- Location: Downtown
- Cuisine: French
- Price: $$$
- Parking: Street parking
- Adaptation: Take taxi (same reasoning applies)
```

**Complex Adaptation: Rule-Based Modification**

```
Old Case: Route planning
- Origin: Home
- Destination: Airport
- Time: 8am weekday
- Route: Highway 85 → 280
- Duration: 45 minutes

New Problem: Similar trip
- Origin: Home
- Destination: Airport
- Time: 5pm weekday (DIFFERENT!)

Adaptation Rules:
- IF time=rush-hour THEN add 20 minutes
- IF time=rush-hour THEN prefer alternate routes

Adapted Solution:
- Route: Side roads → 280 (avoid 85 in rush hour)
- Duration: 65 minutes
```

**Model-Based Adaptation:**

Use domain model to transform solution:

```
Old Case: Bridge Design
- Span: 100m
- Load: 1000 tons
- Materials: Steel beams X, Y, Z
- Solution: Truss design D1

New Problem: Similar bridge
- Span: 150m (50% longer)
- Load: 1500 tons (50% heavier)

Adaptation via structural model:
- Scale beam dimensions by factor 1.5
- Add support column at midpoint
- Recalculate stress loads
→ Adapted design D2
```

### Case Evaluation

After adapting a solution, evaluate whether it works:

**Methods:**
1. **Execution:** Try it in the real world
2. **Simulation:** Test in simulated environment
3. **Human Expert:** Ask domain expert to evaluate
4. **Formal Verification:** Prove correctness mathematically
5. **Criteria Checking:** Compare against requirements

**Outcomes:**
- **Success:** Solution works, store case with positive marker
- **Failure:** Solution fails, explain why, store failure case
- **Partial Success:** Works but suboptimal, store with notes

**Feedback Loop:**
Evaluation results inform future retrieval and adaptation:
- Successful cases retrieved more often
- Failed cases help avoid mistakes
- Patterns in failures → Better adaptation rules

### Case Storage

**Question:** Store every single case?

**Issues with storing everything:**
- Memory grows without bound
- Redundant cases waste space
- Retrieval slows down with too many cases

**Storage Strategies:**

**Strategy 1: Redundancy Removal**
```
If new case is very similar to existing case:
  → Don't store new case (redundant)
  → Or merge with existing case
```

**Strategy 2: Prototypical Cases**
```
Store only representative cases:
  → Central examples of each category
  → Discard outliers or noise
```

**Strategy 3: Failure-Driven Storage**
```
Prioritize storing:
  → Cases where initial retrieval failed
  → Cases with surprising outcomes
  → Cases that taught something new
```

### Case Indexing

**Problem:** With thousands of cases, how to efficiently retrieve relevant ones?

**Solution:** Index cases by important features

**Discrimination Tree Example:**

```
                    All Cases
                       │
            ┌──────────┴──────────┐
         Origin?                   │
    ┌───────┴───────┐              │
  Home           Work            Other
    │               │              │
 Time?           Time?          Time?
  ┌─┴─┐          ┌─┴─┐         ┌─┴─┐
  AM  PM         AM  PM        AM  PM
  │   │          │   │         │   │
 [Cases]      [Cases]       [Cases]
```

**Index Selection:**
- Use features that discriminate well
- Order by importance (most discriminating first)
- Balance tree depth vs. breadth

**Retrieval with Index:**
1. Start at root
2. Follow branches based on new problem's features
3. Reach leaf node with small set of similar cases
4. Apply k-NN within that subset

**Advantage:** Fast retrieval - O(log n) instead of O(n)

### Advanced CBR: Adaptation by Analogy

**Cross-Domain Adaptation:**

```
Source Case: Heat flow in metal rod
- Problem: Calculate temperature distribution
- Solution: Fourier's law, differential equations

Target Problem: Traffic flow on highway
- Problem: Calculate traffic density distribution
- Adaptation: Map heat→cars, temperature→density
- Solution: Adapted flow equations

Analogical Mapping:
- Heat ≈ Cars
- Temperature ≈ Density
- Thermal conductivity ≈ Road capacity
- Heat source ≈ On-ramp
→ Transfer solution structure across domains
```

This connects CBR to **analogical reasoning** (covered in Advanced Reasoning module).

---

## 3. Incremental Concept Learning

### The "Foo" Problem

**Task:** Learn the concept of "Foo" from examples

**Example 1:** (Positive example - IS a Foo)
```
┌─────┐
│ ○ ○ │  Small circle above small circle
│  ○  │
└─────┘
```

**Initial Concept:**
```
Foo:
- Object1: Circle, Small
- Object2: Circle, Small
- Relation: Object1 above Object2
```

**Example 2:** (Positive example - IS a Foo)
```
┌─────┐
│ □ □ │  Small square above small square
│  □  │
└─────┘
```

**Updated Concept:** Must generalize

```
Foo (generalized):
- Object1: Shape=?, Size=Small
- Object2: Shape=?, Size=Small
- Relation: Object1 above Object2
```

The shape is no longer specified (generalized), but size remains small.

### Three Key Operations

**1. Variabilization (Generalization)**

Replace specific value with variable when examples differ:

```
Before: Shape=Circle
Examples show: Circle, Square, Triangle
After: Shape=? (any shape)
```

**2. Specialization**

Add requirements when negative example appears:

```
Negative Example: Large circle above large circle (NOT a Foo)

Current: Size=?
After specialization: Size=Small (must be small)
```

**3. Abstraction**

Move to higher-level concept:

```
Before: Shape=Circle OR Shape=Square OR Shape=Triangle
After: Shape=Polygon (abstract category)
```

### Incremental Learning Process

**Initial State:** No concept of Foo

**Example 1 (Positive):** Small red circle above small red circle
```
Concept: Exactly match this example
```

**Example 2 (Positive):** Small blue circle above small blue circle
```
Difference: Color (red vs. blue)
Action: Variabilize color
New Concept: Small circle above small circle (any color)
```

**Example 3 (Positive):** Small circle above small square
```
Difference: Bottom shape (circle vs. square)
Action: Variabilize bottom shape
New Concept: Small circle above small shape (any bottom shape)
```

**Example 4 (Negative):** Large circle above small circle (NOT Foo)
```
Must exclude this case
Action: Specialize to require top=small
New Concept: Small circle above small shape, top must be small
```

**Example 5 (Positive):** Medium circle above medium triangle
```
Difference: Size (small vs. medium)
Current concept excludes this!
Action: Generalize size to medium-or-small
Or: Abstract to "similar-size objects"
```

### Heuristics for Concept Learning

**When to Generalize:**
- Positive example differs in one feature
- That feature is currently specific
- Generalize just that feature (minimal change)

**When to Specialize:**
- Negative example matches current concept
- Add constraint to exclude it
- Require feature to take specific value(s)
- Or: Forbid feature from taking specific value

**When to Abstract:**
- Multiple generalizations pile up (Circle OR Square OR...)
- Higher-level concept encompasses variations
- Replace OR list with abstract category

**Require vs. Forbid:**
- **Require:** Feature MUST have certain value(s)
- **Forbid:** Feature MUST NOT have certain value(s)
- **Example:**
  - Require(size, small): Size must be small
  - Forbid(size, large): Size cannot be large

### Final Concept of Foo

After seeing many examples:

```
Foo consists of:
- Two objects
- Both objects same size (small or medium, not large)
- Top object is same shape as bottom object OR
  Top object is different shape from bottom object
  (shape can vary)
- Top object is above bottom object
- Objects can be any color
```

This concept has been learned incrementally from examples!

### Connection to Version Spaces

Concept learning maintains two boundaries:

**Specific Boundary (S):**
- Most specific concept consistent with positives
- Excludes negatives

**General Boundary (G):**
- Most general concept consistent with positives
- Excludes negatives

**Version Space:**
- All concepts between S and G
- Each new example narrows the space
- Converge on target concept

---

## 4. Classification

### From Concept Learning to Classification

**Concept Learning:** Determining the definition of a concept

**Classification:** Assigning objects to already-defined concepts

**Relationship:**
- First learn concepts (what is a "bird"?)
- Then classify new objects (is this a bird?)

### Equivalence Classes

Group objects into categories based on properties:

**Example: Classifying Birds**

```
Class: Birds
Members: Robin, Sparrow, Penguin, Ostrich, Eagle

Common Properties:
- Has feathers
- Lays eggs
- Has beak
- Has wings

Distinguishing Properties:
- Can fly (most, but not penguin/ostrich)
- Size (varies widely)
- Color (varies widely)
```

### Concept Hierarchies

Organize concepts in taxonomies:

```
              Animal
                │
        ┌───────┴───────┐
      Bird            Mammal
        │                │
    ┌───┴───┐        ┌───┴───┐
  Penguin Eagle      Dog   Cat
    │       │         │     │
 Emperor  Bald      Beagle Persian
```

**Inheritance:**
- Lower levels inherit properties from upper levels
- Eagle inherits "has feathers" from Bird
- Can override properties (Penguin: can-fly=false)

**Classification Strategy:**
1. Start at top of hierarchy
2. Check properties to descend
3. Continue until reaching specific category

### Types of Concepts

**1. Axiomatic Concepts (Classical)**
- Defined by necessary and sufficient conditions
- Clear boundaries
- Example: Triangle = closed figure with exactly 3 sides

```
Triangle Definition:
- MUST have 3 sides (necessary)
- 3 sides is SUFFICIENT
- No ambiguity
```

**2. Prototype Concepts**
- Defined by typical example (prototype)
- Graded membership (more/less typical)
- Fuzzy boundaries

```
Bird Prototype:
- Flies ✓
- Small ✓
- Sings ✓
- Perches in trees ✓

Robin: Very typical bird (close to prototype)
Penguin: Atypical bird (far from prototype)
```

**3. Exemplar Concepts**
- Defined by collection of examples
- Classify by similarity to stored exemplars
- No single prototype

```
"Chair" Exemplars:
- Office chair (example 1)
- Kitchen chair (example 2)
- Bean bag chair (example 3)
- Rocking chair (example 4)

New object: Compare to all exemplars, classify if similar enough
```

### Prototype vs. Exemplar

**Prototype Approach:**
```
Concept: Bird
Prototype: {flies=yes, size=small, sings=yes}

New object: Compare to prototype
  Similarity to prototype → Confidence in classification
```

**Advantages:**
- Efficient (compare to one prototype)
- Captures central tendency
- Easy to understand

**Disadvantages:**
- May not represent diversity
- Hard to define prototype for heterogeneous categories

**Exemplar Approach:**
```
Concept: Bird
Exemplars: {robin, eagle, penguin, ostrich, ...}

New object: Compare to ALL exemplars
  Average similarity → Confidence in classification
```

**Advantages:**
- Captures diversity within category
- Handles atypical members well
- No information loss

**Disadvantages:**
- Computationally expensive
- Requires storing all examples
- May overfit to noise

### Order of Concepts

What's the relationship between subconcepts and superconcepts?

**Bottom-Up Classification:**
```
Observe: Four legs, fur, barks
  ↓
Classify as: Dog
  ↓
Infer: Therefore, Mammal
  ↓
Infer: Therefore, Animal
```

**Top-Down Classification:**
```
Observe: Unknown object
  ↓
Check: Is it Animal? Yes (moves, eats)
  ↓
Check: Is it Mammal? Yes (fur, warm-blooded)
  ↓
Check: Is it Dog? Yes (barks, friendly)
```

**Which is better?**
- Bottom-up: Efficient when low-level features are distinctive
- Top-down: Efficient when high-level categories constrain search
- **Reality:** Combination of both (bidirectional)

### Classification in Practice

**Medical Diagnosis:**
```
Symptoms: Fever, Cough, Fatigue
  ↓
Top-down: Is it infectious? Yes (fever)
           Is it respiratory? Yes (cough)
           Is it viral? Check white blood cell count
  ↓
Bottom-up: Specific symptom pattern matches Flu exemplars
  ↓
Classification: Influenza (high confidence)
```

**Image Recognition:**
```
Pixels → Edge detection → Shape recognition
  ↓
Bottom-up: Collection of features (fur, whiskers, pointed ears)
Top-down: Context suggests "animal"
  ↓
Classification: Cat
```

---

## 5. Version Spaces

### The Concept

**Version Space:** The set of all concept definitions consistent with observed examples.

**Goal:** Narrow down to the single correct concept through systematic refinement.

### Boundaries of Version Space

**General Boundary (G):**
- Most general hypotheses consistent with examples
- Accepts all positives, rejects all negatives

**Specific Boundary (S):**
- Most specific hypotheses consistent with examples
- Accepts all positives, rejects all negatives

**Version Space = All hypotheses between S and G**

### Example: Food Allergies

**Problem:** Determine what food causes allergic reaction

**Features:**
- Main ingredient: Beef, Chicken, Fish
- Sauce: Tomato, Cream, None
- Vegetable: Carrots, Peas, Broccoli

**Example 1 (Positive - caused reaction):**
```
Meal: Beef, Tomato sauce, Carrots → Reaction

Initial S: {Beef, Tomato, Carrots}
  (Most specific: exactly this meal)

Initial G: {?, ?, ?}
  (Most general: any meal)
```

**Example 2 (Negative - no reaction):**
```
Meal: Chicken, Cream, Peas → No reaction

Must exclude this from concept!

Updated G:
  {Beef, ?, ?} OR {?, Tomato, ?} OR {?, ?, Carrots}
  (Must differ in at least one feature)

S unchanged: {Beef, Tomato, Carrots}
```

**Example 3 (Positive - caused reaction):**
```
Meal: Beef, Cream, Peas → Reaction

Common with Example 1: Beef
Different: Sauce, Vegetable

Updated S: {Beef, ?, ?}
  (Generalize: Only Beef is common to all positives)

Updated G: {Beef, ?, ?}
  (All hypotheses in G must be consistent with this positive)
```

**Convergence:**
```
S and G have converged: {Beef, ?, ?}
Conclusion: Beef causes the allergic reaction!
```

### Version Space Algorithm

**Initialize:**
```
S = most specific hypothesis
G = most general hypothesis
```

**For each positive example:**
```
Generalize S to include example (if needed)
Remove from G any hypothesis that excludes example
```

**For each negative example:**
```
Specialize G to exclude example (if needed)
Remove from S any hypothesis that includes example
```

**Termination:**
```
When S = G: Unique concept found
When S and G converge: Concept learned
When S and G become inconsistent: No consistent concept exists
```

### Optimal Question Ordering

**Problem:** Which example to show next?

**Strategy:** Maximum discrimination

**Choose example that:**
- Splits version space in half (or as evenly as possible)
- Maximizes information gain
- Minimizes expected number of examples needed

**Identification Trees:**
```
Like decision trees, but for concept learning:
- Each node asks about a feature
- Branches based on values
- Leaves are concepts
- Optimal tree minimizes average depth
```

### Advantages of Version Spaces

1. **Systematic:** Guarantees finding consistent concept
2. **Efficient:** Represents whole space with just S and G boundaries
3. **Query Optimization:** Can choose best questions to ask
4. **Correctness:** Provably correct when enough examples provided

### Limitations

1. **Feature Representation:** Requires features known in advance
2. **Noise:** Sensitive to mislabeled examples
3. **Complexity:** Boundaries can grow exponentially
4. **Single Concept:** Assumes single target concept exists

---

## 6. Learning by Correcting Mistakes

### The Error-Driven Learning Cycle

**Philosophy:** Mistakes are learning opportunities

**Process:**
```
1. Make a decision
2. Execute action
3. Observe outcome
4. Detect error (if any)
5. Explain why error occurred
6. Correct knowledge to prevent future errors
```

### Example: Identifying Cups

**Case 1: Successful Classification**
```
Object: Cylindrical, Handle, Holds liquid
Agent: "This is a cup" ✓
Outcome: Correct
Action: Reinforce concept
```

**Case 2: Error**
```
Object: Cylindrical, Handle, Holds liquid, Very large (bucket)
Agent: "This is a cup" ✗
Feedback: "No, it's too large to be a cup"
Error detected!
```

### Error Detection

**How to detect errors:**

**1. Explicit Feedback:**
- Teacher corrects agent
- User provides negative feedback
- System reports failure

**2. Expectation Violation:**
```
Agent expects: Cup can be lifted with one hand
Observation: Cannot lift this "cup"
Conclusion: Classification error
```

**3. Goal Failure:**
```
Goal: Pour coffee into cup
Action: Try to pour into bucket
Outcome: Coffee overflows (too fast)
Conclusion: Bucket behaves differently than cup
```

### Explaining Mistakes

**Why did the error occur?**

**Types of Explanations:**

**1. Missing Feature:**
```
Current concept: {cylindrical, has-handle, holds-liquid}
Error: Classified bucket as cup

Explanation: Missing feature "size=small"
Should have checked size!
```

**2. Incorrect Feature:**
```
Current concept: "All birds can fly"
Error: Classified penguin as non-bird

Explanation: Feature "can-fly" is incorrect
Some birds cannot fly
```

**3. Overgeneralization:**
```
Current concept: "All metal things are magnetic"
Error: Copper is metal but not magnetic

Explanation: Overgeneralized from iron
Should be "Some metal things are magnetic"
```

**4. Wrong Abstraction Level:**
```
Current concept: "Fruit" includes tomatoes
Context: Making fruit salad
Error: Tomatoes don't taste good in fruit salad

Explanation: Right botanically, wrong culinarily
Need context-specific categories
```

### Correcting Mistakes

**Correction Strategies:**

**1. Add Feature Requirements:**
```
Before: Cup = {cylindrical, has-handle}
Error: Bucket incorrectly classified
After: Cup = {cylindrical, has-handle, size=small}
```

**2. Relax Feature Requirements:**
```
Before: Bird = {has-feathers, can-fly}
Error: Penguin incorrectly rejected
After: Bird = {has-feathers}
  (Remove "can-fly" requirement)
```

**3. Add Exceptions:**
```
Concept: Birds can fly
Exception: Penguins, Ostriches cannot fly
Rule: Bird → can-fly, EXCEPT {Penguin, Ostrich}
```

**4. Create Subcategories:**
```
Before: Single "Bird" category
After:
  - Flying birds (Robin, Eagle)
  - Flightless birds (Penguin, Ostrich)
Each subcategory has appropriate properties
```

**5. Adjust Feature Weights:**
```
Before: All features equally important
Error: Shape matched but function didn't
After: Increase weight of functional features
  (Function more important than shape for cups)
```

### The Knowledge Gap Model

**Three Types of Knowledge Gaps:**

**1. Missing Knowledge:**
```
Don't know: Cups must be small-sized
Result: Classify buckets as cups
Fix: Add size constraint
```

**2. Incorrect Knowledge:**
```
Believe: All birds fly
Truth: Some birds don't fly
Fix: Correct rule or add exceptions
```

**3. Unusable Knowledge:**
```
Know: Cups are drinking vessels
But: Don't know how to check "drinking vessel" property
Fix: Operationalize concept (add measurable features)
```

### Integration with Other Learning Methods

**Connection to Case-Based Reasoning:**
```
Error case → Stored as failure case
Future retrieval → Avoid similar mistakes
Adaptation rules → Adjusted based on errors
```

**Connection to Explanation-Based Learning:**
```
Error → Triggers explanation process
Explanation → Identifies knowledge gap
Gap-filling → Learning happens
(Covered in Advanced Reasoning module)
```

**Connection to Chunking:**
```
Error → Creates impasse in production system
Impasse → Triggers chunking
New rule → Prevents future error
```

### Cognitive Connection

**Human Learning:**
- Children learn from mistakes constantly
- Errors trigger re-evaluation of concepts
- Explanation-seeking is natural human response
- "Teachable moments" are often after mistakes

**AI Learning:**
- Error-driven learning mirrors human learning
- Makes AI systems that improve from experience
- Enables graceful degradation (fail, learn, improve)
- Critical for real-world deployment

---

## Summary

### Key Takeaways

1. **Learning by Recording Cases** uses k-nearest neighbor in multi-dimensional feature spaces to classify new problems based on similarity to stored experiences. Distance metrics and feature weighting are critical for effectiveness.

2. **Case-Based Reasoning** extends simple case retrieval with a complete cycle: Retrieve similar cases, Adapt solutions to new context, Evaluate results, and Store new experiences. Indexing structures enable efficient retrieval from large case libraries.

3. **Incremental Concept Learning** builds concepts from examples through three operations: Variabilization (generalize when positive examples differ), Specialization (add constraints when negative examples match), and Abstraction (move to higher-level concepts).

4. **Classification** assigns objects to predefined concepts using prototype (compare to typical example) or exemplar (compare to all examples) approaches. Concept hierarchies enable inheritance and efficient classification.

5. **Version Spaces** systematically narrow the space of consistent hypotheses by maintaining Specific and General boundaries, converging toward the target concept through positive and negative examples.

6. **Learning by Correcting Mistakes** detects errors, explains why they occurred, and corrects knowledge to prevent recurrence. Three types of knowledge gaps: missing, incorrect, and unusable knowledge.

### Essential Principles

- **Experience is valuable:** Past cases inform future decisions
- **Similarity enables transfer:** Similar problems → Similar solutions
- **Learning is incremental:** Concepts refined gradually through examples
- **Errors are opportunities:** Mistakes reveal knowledge gaps
- **Multiple representations:** Prototypes, exemplars, rules, cases all useful
- **Integration is key:** Learning methods work together in cognitive systems

### Method Comparison

| Method | Strength | Weakness | Best For |
|--------|----------|----------|----------|
| k-NN | Simple, no training | Slow retrieval | Small datasets |
| CBR | Flexible, explainable | Adaptation complexity | Complex domains |
| Concept Learning | Systematic, provable | Needs good features | Well-defined concepts |
| Classification | Fast, hierarchical | Rigid categories | Taxonomic domains |
| Version Spaces | Optimal, minimal examples | Noise sensitive | Clean data |
| Error Correction | Self-improving | Needs feedback | Interactive settings |

### Connections Across Course

```
Case-Based Reasoning
    ↓
Analogical Reasoning (Module 6)
    ↓
Explanation-Based Learning (Module 6)
    ↓
Meta-Reasoning (Module 8)
```

All learning methods share common theme: **Using knowledge to acquire more knowledge**

---

## See Also

- [[02-Core-Reasoning-Strategies|Core Reasoning]] - Chunking as learning mechanism
- [[06-Advanced-Reasoning|Advanced Reasoning]] - Analogical reasoning and explanation-based learning
- [[08-Metacognition-and-Advanced|Metacognition]] - Learning about learning
- [[00-README|Course Overview]] - Navigate the full course structure

---

*Learning is not separate from reasoning and memory - they form a unified, interdependent system where each enables and enhances the others.*
