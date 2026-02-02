---
type: note
course: "[[Knowledge-Based AI (CS 7637)]]"
date: 2026-02-02
topic: Logic and Planning
lessons: 12-13
prerequisites:
  - "[[01-Fundamentals-KBAI|Fundamentals of KBAI]]"
  - "[[02-Core-Reasoning-Strategies|Core Reasoning Strategies]]"
  - Understanding of means-ends analysis and problem reduction
---

# Logic and Planning

## Prerequisites

- Understanding of semantic networks and knowledge representations
- Familiarity with means-ends analysis and problem reduction
- Knowledge of state spaces and operators
- Completed Fundamentals and Core Reasoning modules

## Learning Goals

After completing this module, you will be able to:

1. Express knowledge using predicate logic with conjunctions, disjunctions, and implications
2. Construct and interpret truth tables for logical expressions
3. Apply rules of inference including modus ponens and resolution
4. Define planning problems with states, goals, and operators
5. Use partial order planning to construct flexible plans
6. Detect and resolve conflicts in partially ordered plans
7. Handle open preconditions systematically

---

## 1. Formal Logic

### Why Logic in KBAI?

**Logic provides:**
- Precise, unambiguous knowledge representation
- Formal rules for drawing inferences
- Foundation for automated reasoning
- Connection between representation and reasoning

**Role in Planning:**
- Express states as logical formulas
- Express goals as logical formulas
- Express operators as logical rules
- Reasoning determines action sequences

### Predicates

**Predicate:** A function that returns true or false

**Example:**
```
Bird(Tweety) → "Tweety is a bird" (True or False)
CanFly(Tweety) → "Tweety can fly" (True or False)
Color(Tweety, Yellow) → "Tweety is yellow" (True or False)
```

**Anatomy:**
```
Predicate(argument1, argument2, ...)
    ↑           ↑
  Name       Arguments
```

**Common Predicates:**
```
IsA(X, Y) → "X is a Y"
Has(X, Y) → "X has Y"
On(X, Y) → "X is on Y"
Loves(X, Y) → "X loves Y"
```

### Logical Connectives

**Conjunction (AND): ∧**
```
Bird(Tweety) ∧ CanFly(Tweety)
→ "Tweety is a bird AND Tweety can fly"

True only if BOTH parts are true
```

**Disjunction (OR): ∨**
```
Bird(X) ∨ Mammal(X)
→ "X is a bird OR X is a mammal"

True if EITHER part is true (or both)
```

**Negation (NOT): ¬**
```
¬CanFly(Penguin)
→ "Penguin cannot fly"

Reverses truth value
```

**Implication (IF-THEN): →**
```
Bird(X) → HasFeathers(X)
→ "If X is a bird, then X has feathers"

False only when antecedent true but consequent false
```

**Biconditional (IF AND ONLY IF): ↔**
```
Triangle(X) ↔ HasThreeSides(X)
→ "X is a triangle if and only if X has three sides"

True when both have same truth value
```

### Complex Expressions

**Combining Connectives:**
```
Bird(X) ∧ ¬CanFly(X) → Flightless(X)
→ "If X is a bird AND cannot fly, then X is flightless"

Example: Bird(Penguin)=T, CanFly(Penguin)=F
         → ¬CanFly(Penguin)=T
         → Bird(Penguin) ∧ ¬CanFly(Penguin)=T
         → Flightless(Penguin)=T
```

**Nested Implications:**
```
Raining(outside) → (HaveUmbrella(person) → StayDry(person))
→ "If it's raining, then if person has umbrella, then person stays dry"
```

### Truth Tables

**Truth table:** Shows truth value of expression for all possible input combinations

**Basic Connectives:**

**AND (∧):**
```
 A  │  B  │ A ∧ B
────┼─────┼───────
 T  │  T  │   T
 T  │  F  │   F
 F  │  T  │   F
 F  │  F  │   F
```

**OR (∨):**
```
 A  │  B  │ A ∨ B
────┼─────┼───────
 T  │  T  │   T
 T  │  F  │   T
 F  │  T  │   T
 F  │  F  │   F
```

**NOT (¬):**
```
 A  │ ¬A
────┼────
 T  │  F
 F  │  T
```

**Implication (→):**
```
 A  │  B  │ A → B
────┼─────┼───────
 T  │  T  │   T
 T  │  F  │   F    ← Only false case!
 F  │  T  │   T    ← Vacuously true
 F  │  F  │   T    ← Vacuously true
```

**Key Insight:** A → B is false ONLY when A is true but B is false.

### Logical Equivalences

**De Morgan's Laws:**
```
¬(A ∧ B) ≡ (¬A ∨ ¬B)
¬(A ∨ B) ≡ (¬A ∧ ¬B)

Example: ¬(Raining ∧ Cold) ≡ (¬Raining ∨ ¬Cold)
         "Not (raining and cold)" = "Not raining or not cold"
```

**Implication Elimination:**
```
A → B ≡ ¬A ∨ B

Example: Bird(X) → CanFly(X) ≡ ¬Bird(X) ∨ CanFly(X)
```

**Commutative Property:**
```
A ∧ B ≡ B ∧ A
A ∨ B ≡ B ∨ A

Order doesn't matter for AND/OR
```

**Associative Property:**
```
(A ∧ B) ∧ C ≡ A ∧ (B ∧ C)
(A ∨ B) ∨ C ≡ A ∨ (B ∨ C)

Grouping doesn't matter
```

**Distributive Property:**
```
A ∧ (B ∨ C) ≡ (A ∧ B) ∨ (A ∧ C)
A ∨ (B ∧ C) ≡ (A ∨ B) ∧ (A ∨ C)
```

### Rules of Inference

**Modus Ponens:**
```
Premise 1: A → B
Premise 2: A
Conclusion: B

Example:
  Bird(Tweety) → CanFly(Tweety)
  Bird(Tweety)
  ∴ CanFly(Tweety)
```

**Modus Tollens:**
```
Premise 1: A → B
Premise 2: ¬B
Conclusion: ¬A

Example:
  Bird(X) → HasFeathers(X)
  ¬HasFeathers(Penguin)
  ∴ ¬Bird(Penguin)
```

**Hypothetical Syllogism:**
```
Premise 1: A → B
Premise 2: B → C
Conclusion: A → C

Example:
  Bird(X) → Animal(X)
  Animal(X) → LivingThing(X)
  ∴ Bird(X) → LivingThing(X)
```

**Resolution:**
```
Premise 1: A ∨ B
Premise 2: ¬B ∨ C
Conclusion: A ∨ C

Example:
  Bird(X) ∨ Mammal(X)
  ¬Mammal(X) ∨ HasFur(X)
  ∴ Bird(X) ∨ HasFur(X)
```

### Proof Example

**Goal:** Prove Harry is a bird

**Knowledge Base:**
```
1. Feathers(Harry)                    [Given]
2. ∀x: Feathers(x) → Bird(x)         [Rule]
3. ∀x: Bird(x) → Animal(x)           [Rule]
```

**Proof:**
```
Step 1: Feathers(Harry)               [Premise 1]
Step 2: Feathers(Harry) → Bird(Harry) [Universal instantiation of Rule 2]
Step 3: Bird(Harry)                   [Modus Ponens on Steps 1, 2]
∴ Harry is a bird ✓
```

**Extended Proof:**
```
Step 4: Bird(Harry) → Animal(Harry)  [Universal instantiation of Rule 3]
Step 5: Animal(Harry)                [Modus Ponens on Steps 3, 4]
∴ Harry is an animal ✓
```

---

## 2. Planning Fundamentals

### What is Planning?

**Planning** is the problem-solving activity whose goal is to come up with a sequence of operators (actions) that will achieve one or more goals.

**Key Components:**
1. **Initial State:** Where we start
2. **Goal State:** Where we want to be
3. **Operators:** Actions that change state
4. **Plan:** Sequence of operators that achieves goal

### States

**State:** Complete description of the world at a point in time

**Blocks World Example:**
```
State S1:
  On(A, Table)
  On(B, Table)
  On(C, A)
  Clear(C)
  Clear(B)
  HandEmpty
```

**Properties of States:**
- **Complete:** Includes all relevant facts
- **Consistent:** No contradictions
- **Propositional:** Can be expressed in logic

### Goals

**Goal:** Desired properties of final state

**Examples:**
```
Goal G1: On(A, B) ∧ On(B, C)
         "A on B, and B on C"

Goal G2: Clear(A) ∧ On(A, Table)
         "A is clear and on table"
```

**Goals vs. States:**
- State: Complete description (all facts)
- Goal: Partial description (only desired facts)
- Many states can satisfy a goal

### Operators

**Operator:** Action that transforms one state into another

**Structure:**
```
Operator:
  Name: Move(X, Y)
  Preconditions: What must be true before
  Effects: What becomes true after
```

**Blocks World Operators:**

**Operator 1: Move block to table**
```
Name: MoveToTable(X)
Preconditions:
  - Clear(X)       [Nothing on X]
  - On(X, Y)       [X is on something]
  - HandEmpty      [Hand is free]
Effects:
  - Add: On(X, Table), Clear(Y)
  - Delete: On(X, Y), HandEmpty
  - Add: Holding(X)
```

**Operator 2: Stack block on another**
```
Name: Stack(X, Y)
Preconditions:
  - Holding(X)     [Holding X]
  - Clear(Y)       [Y has nothing on it]
Effects:
  - Add: On(X, Y), HandEmpty
  - Delete: Holding(X), Clear(Y)
  - Add: Clear(X)
```

### Planning as State Space Search

**State Space:**
- Nodes = States
- Edges = Operators
- Path = Plan

```
Initial State → Op1 → State2 → Op2 → State3 → ... → Goal State
```

**Example:**
```
S0: C on A, A on Table, B on Table
    ↓ MoveToTable(C)
S1: C on Table, A on Table, B on Table
    ↓ Stack(A, B)
S2: C on Table, A on B, B on Table
    ↓ Stack(C, A)
S3: C on A, A on B, B on Table ← Goal!
```

### The Painting a Ceiling Problem

**Scenario:** Paint a ceiling with a ladder

**Initial State:**
```
OnFloor(person)
OnFloor(ladder)
CleanHands(person)
DryFloor
UnpaintedCeiling
```

**Goal:**
```
PaintedCeiling
CleanHands(person)
```

**Operators:**

**ClimbLadder:**
```
Preconditions: OnFloor(person), OnFloor(ladder)
Effects:
  Add: OnLadder(person)
  Delete: OnFloor(person)
```

**PaintCeiling:**
```
Preconditions: OnLadder(person), DryFloor
Effects:
  Add: PaintedCeiling, DirtyHands(person)
  Delete: CleanHands(person), UnpaintedCeiling
```

**Descend:**
```
Preconditions: OnLadder(person)
Effects:
  Add: OnFloor(person), WetFloor
  Delete: OnLadder(person), DryFloor
```

**Simple Plan:**
```
1. ClimbLadder
2. PaintCeiling
3. Descend
```

**Problem:** Hands are dirty at end! Doesn't satisfy goal.

**Better Plan:**
```
1. ClimbLadder
2. PaintCeiling
3. Descend
4. WashHands ← Need additional operator
```

---

## 3. Partial Order Planning

### Motivation

**Total Order Planning:**
```
Must specify: Do A, then B, then C, then D
Very rigid, commits early
```

**Partial Order Planning:**
```
Specify only necessary orderings:
  - A before B
  - C before D
  - B and C can be in any order
More flexible, commits late
```

### Key Concepts

**Open Precondition:**
- Precondition of an operator not yet satisfied
- Needs to be achieved by some earlier operator

**Causal Link:**
```
Op1 ──[condition C]──> Op2

"Op1 achieves condition C for Op2"
```

**Conflict (Threat):**
```
Op1 ──[C]──> Op3
  ↓
 Op2 deletes C

Op2 threatens the causal link
```

### Partial Order Planning Algorithm

**Initialize:**
```
Plan = {Start, Finish}
Start has no preconditions, adds initial state facts
Finish requires goal conditions
```

**Repeat until no open preconditions:**

**Step 1: Select open precondition**
```
Choose some precondition P of operator Op that's not satisfied
```

**Step 2: Find operator to achieve it**
```
Option A: Use existing operator that adds P
Option B: Add new operator that adds P
Create causal link: Achiever ─[P]→ Op
```

**Step 3: Order operators**
```
Ensure Achiever comes before Op
Update partial ordering constraints
```

**Step 4: Resolve conflicts**
```
If any operator threatens a causal link:
  Option A: Order threat before achiever (promotion)
  Option B: Order threat after consumer (demotion)
  Option C: Add constraint to make threat not delete condition
```

### Example: Three-Block Stacking

**Initial State:**
```
  C
  A     B
──────────
  Table
```

**Goal:**
```
  A
  B
  C
──────────
  Table
```

**Planning Process:**

**Initial Plan:**
```
Start → Finish

Finish requires:
  - On(A, B)      [open]
  - On(B, C)      [open]
  - On(C, Table)  [open]
```

**Iteration 1: Achieve On(A, B)**
```
Add operator: Stack(A, B)
Preconditions:
  - Holding(A)    [open]
  - Clear(B)      [open]

Plan: Start → Stack(A, B) → Finish
         Causal link: Stack(A,B) ─[On(A,B)]→ Finish
```

**Iteration 2: Achieve Holding(A)**
```
Add operator: PickUp(A)
Precondition:
  - Clear(A)      [open]
  - OnTable(A)    [from Start, satisfied]

Causal link: PickUp(A) ─[Holding(A)]→ Stack(A,B)
Ordering: PickUp(A) before Stack(A,B)
```

**Iteration 3: Achieve Clear(A)**
```
Need to remove C from A
Add operator: Unstack(C, A)

Causal link: Unstack(C,A) ─[Clear(A)]→ PickUp(A)
Ordering: Unstack(C,A) before PickUp(A)
```

**Continue until all preconditions satisfied...**

### Conflict Resolution Example

**Scenario:**
```
Op1 ──[On(A,B)]──> Op3 (needs A on B)
  ↓
Op2 (moves A off B, deleting On(A,B))

Conflict! Op2 threatens causal link
```

**Resolution Options:**

**Promotion:**
```
Start → Op1 → Op3 → Op2 → Finish
         [On(A,B)]
Move Op2 after Op3 (promotes Op3)
```

**Demotion:**
```
Start → Op2 → Op1 → Op3 → Finish
                     [On(A,B)]
Move Op2 before Op1 (demotes Op2)
```

**Choice depends on:**
- Other constraints in plan
- Which resolution maintains feasibility
- Efficiency considerations

### Advantages of Partial Order Planning

**1. Flexibility:**
```
Can execute independent actions in any order
Allows parallelization
```

**2. Least Commitment:**
```
Don't decide ordering until necessary
Keeps options open
Easier to modify plan
```

**3. Conflict Detection:**
```
Systematically finds and resolves conflicts
Ensures plan correctness
```

**4. Incremental:**
```
Build plan step by step
Can stop when "good enough"
Can continue refining
```

### Comparison with Other Planning Approaches

**Total Order Planning:**
```
Pros: Simpler, easier to execute
Cons: Overcommits, less flexible
```

**Partial Order Planning:**
```
Pros: Flexible, optimal ordering
Cons: More complex, needs conflict resolution
```

**Hierarchical Planning:**
```
Pros: Handles complexity through abstraction
Cons: Needs domain-specific decomposition
```

**Reactive Planning:**
```
Pros: Handles dynamics, no advance planning needed
Cons: May not find optimal solutions
```

---

## 4. Integration with Other KBAI Concepts

### Connection to Means-Ends Analysis

**Means-Ends Analysis:**
```
1. Identify difference between current and goal
2. Find operator to reduce difference
3. Apply operator
4. Repeat
```

**Planning:**
```
1. Identify open preconditions (differences)
2. Find operators to achieve them
3. Order operators appropriately
4. Repeat until all satisfied
```

**Key Similarity:** Both work backward from goal to find relevant operators.

### Connection to Problem Reduction

**Problem Reduction:**
```
Decompose complex goal into subgoals
Solve each subgoal independently
Compose solutions
```

**Partial Order Planning:**
```
Decompose goal into multiple conditions
Achieve each condition with operators
Order operators to avoid conflicts
```

**Integration:**
```
Use problem reduction to identify subgoals
Use partial order planning to order actions
Hierarchical planning combines both
```

### Connection to Logic

**Planning uses logic for:**

**State Representation:**
```
State = Conjunction of facts
S1 = On(A,B) ∧ On(B,Table) ∧ Clear(A)
```

**Goal Specification:**
```
Goal = Logical formula to satisfy
G = On(C,B) ∧ On(B,A) ∧ On(A,Table)
```

**Precondition Checking:**
```
Can apply operator if:
  CurrentState → Preconditions
  (Logical implication)
```

**Effect Application:**
```
NewState = (OldState - DeleteList) ∪ AddList
(Set operations based on logic)
```

---

## Summary

### Key Takeaways

1. **Formal Logic** provides precise knowledge representation through predicates, logical connectives (∧, ∨, ¬, →), and rules of inference (modus ponens, resolution). Truth tables systematically enumerate all possibilities.

2. **Logical Equivalences** enable transformation of expressions: De Morgan's laws, implication elimination, commutative, associative, and distributive properties allow rewriting while preserving meaning.

3. **Planning Problems** consist of initial state (where we start), goal state (where we want to be), and operators (actions that transform states). Plans are sequences of operators achieving goals.

4. **Operators** have preconditions (what must be true to apply) and effects (what becomes true/false after application). Both expressed as logical formulas.

5. **Partial Order Planning** builds flexible plans by specifying only necessary operator orderings, using causal links to track dependencies and resolving conflicts through promotion or demotion.

6. **Open Preconditions** drive planning process - select unsatisfied precondition, find operator to achieve it, order appropriately, resolve any conflicts. Repeat until all preconditions satisfied.

7. **Integration:** Planning connects means-ends analysis (goal-driven search), problem reduction (decomposition), and logic (formal representation and reasoning).

### Essential Principles

- **Formal representation enables automated reasoning:** Logic provides unambiguous semantics
- **Explicit preconditions and effects:** Make operator applicability and consequences clear
- **Least commitment:** Don't decide orderings until necessary (flexibility)
- **Causal links:** Track dependencies between operators
- **Conflict resolution:** Systematic methods (promotion/demotion) ensure plan correctness
- **Logic as foundation:** State, goals, and operators all expressed logically

### Planning Strategies

| Strategy | When to Use | Advantage | Limitation |
|----------|------------|-----------|------------|
| Forward Search | Clear path to goal | Efficient if branching low | Explores irrelevant states |
| Backward Search | Complex initial state | Focuses on goal | May generate impossible states |
| Partial Order | Need flexibility | Optimal ordering | Complex conflict resolution |
| Hierarchical | Very complex problems | Manages complexity | Needs abstraction hierarchy |

---

## See Also

- [[02-Core-Reasoning-Strategies|Core Reasoning]] - Means-ends analysis and problem reduction
- [[05-Language-and-Commonsense|Language & Common Sense]] - Frames as knowledge structures
- [[07-Applied-Problem-Solving|Applied Problem Solving]] - Configuration and diagnosis
- [[00-README|Course Overview]] - Navigate the full course structure

---

*Planning bridges the gap between knowledge representation (logic) and problem-solving (action sequences). Formal logic provides the precision; planning provides the methodology.*
