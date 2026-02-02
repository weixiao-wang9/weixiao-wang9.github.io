---
type: note
course: "[[Knowledge-Based AI (CS 7637)]]"
date: 2026-02-02
topic: Reasoning Strategies
lessons: 4-6
prerequisites:
  - "[[01-Fundamentals-KBAI|Fundamentals of KBAI]]"
  - Understanding of semantic networks
  - State space concepts
---

# Core Reasoning Strategies

## Prerequisites

- Understanding of semantic networks and knowledge representations
- Familiarity with state spaces and problem representations
- Completed Fundamentals module

## Learning Goals

After completing this module, you will be able to:

1. Apply the Generate & Test method to systematically explore solution spaces
2. Balance responsibility between generators and testers
3. Use Means-Ends Analysis to reduce differences between current and goal states
4. Apply Problem Reduction to decompose complex problems
5. Understand Production Systems as cognitive architectures
6. Implement learning through chunking when impasses occur
7. Design knowledge-based AI systems using rule-based reasoning

---

## 1. Generate & Test

### Core Concept

**Generate & Test** is a fundamental problem-solving method with two components:

1. **Generator:** Creates possible solutions or successor states
2. **Tester:** Evaluates generated solutions and filters out invalid ones

This method systematically explores solution spaces through iterative generation and evaluation.

### The Guards and Prisoners Problem Revisited

**Initial State:**
- Left bank: 3 guards, 3 prisoners, boat
- Right bank: empty
- Goal: Get everyone to right bank safely

**Generation Phase:**
From initial state, generator creates all possible moves:

```
Possible Moves:
1. Move 1 guard →
2. Move 1 guard + 1 prisoner →
3. Move 2 guards →
4. Move 2 prisoners →
5. Move 1 prisoner →
```

**Testing Phase:**
Tester removes illegal states (prisoners > guards):

```
Move 1: G G | G P P P  ← ILLEGAL (2G vs 3P on left)
Move 2: G G P P | G P  ← LEGAL
Move 3: G | G G P P P  ← ILLEGAL (1G vs 3P on left)
Move 4: G G G P | P P  ← LEGAL
Move 5: G G G P P | P  ← LEGAL but UNPRODUCTIVE
```

Result: 2 legal and productive moves remain (moves 2 and 4).

### Dumb Generators vs. Smart Generators

**Dumb Generator:**
- Generates ALL possible successor states
- No intelligence about which states are productive
- Maximum generation, minimal filtering during generation

**Smart Generator:**
- Filters during generation
- Doesn't generate obviously bad states
- Example: Won't generate states identical to already-visited states
- More selective, fewer states to test

### Dumb Testers vs. Smart Testers

**Dumb Tester:**
- Only checks basic problem constraints
- Example: Only removes illegal states (prisoners > guards)
- Doesn't detect repeated states or unproductive paths

**Smart Tester:**
- Checks multiple criteria:
  - Legal states (basic constraints)
  - Previously visited states (avoid cycles)
  - Productive vs. unproductive moves
  - Distance from goal state

### Balancing Responsibility

The key design decision: How smart should each component be?

```
     Generator Intelligence
            ↕
        Trade-off
            ↕
      Tester Intelligence
```

**Option 1: Smart Generator + Dumb Tester**
- Generator does most filtering
- Generates only promising states
- Tester only checks basic legality
- Advantage: Fewer states generated
- Disadvantage: Complex generator logic

**Option 2: Dumb Generator + Smart Tester**
- Generator creates all possibilities
- Tester does sophisticated filtering
- Advantage: Simple generator, flexible testing
- Disadvantage: Many states to test

**Option 3: Both Smart (Recommended)**
- Generator avoids obvious bad states
- Tester performs sophisticated evaluation
- Advantage: Efficient overall
- Disadvantage: More complex system

**Practical Consideration:**
The choice depends on:
- Problem complexity
- Size of state space
- Computational resources
- Knowledge available about problem structure

### Combinatorial Explosion

Without intelligent generation/testing, state spaces grow exponentially:

```
Level 0: 1 state (initial)
Level 1: 5 states
Level 2: 25 states
Level 3: 125 states
...
```

**Problem:** Even small problems create millions of states.

**Solution:** Smart generators and testers prune aggressively:

```
Level 0: 1 state
Level 1: 2 legal productive states
Level 2: 4 legal productive states
Level 3: 3 legal productive states (some paths merged)
...
```

### Application to Ravens Matrices

Ravens problems present additional complexity:

**Discrete vs. Continuous Spaces:**

Guards & Prisoners: Discrete
- Move 1 guard, 2 guards, 1 prisoner, etc.
- Finite, countable options

Ravens Matrices: Continuous
- Object can be placed anywhere (infinite positions)
- Object can be any size (infinite sizes)
- Transformations have infinite variations

**Challenge:** Continuous spaces require:
- Smarter generators (constrain to likely transformations)
- Smarter testers (evaluate similarity measures)
- Heuristics to limit search space

**Example:**
```
Transformation A → B: Circle inside → Circle outside (larger)

Applying to C → ?:
- How far outside? (1 pixel? 10 pixels? 50 pixels?)
- How much larger? (10%? 50%? 100%?)
- Solution: Generate candidates based on observed pattern,
            test similarity to original transformation
```

---

## 2. Means-Ends Analysis

### Core Concept

**Means-Ends Analysis** is a goal-driven problem-solving method that:

1. Identifies differences between current state and goal state
2. Selects operators to reduce those differences
3. Iteratively applies operators until goal is reached

**Key Insight:** Focus on reducing the distance to the goal at each step.

### The Blocks World Problem

**Setup:**
- Table with blocks of different colors and shapes
- Can stack blocks or place them on table
- Goal: Arrange blocks in specific configuration

**Constraints:**
- Move only one block at a time
- Can only move a block with nothing on top of it

**Operators:**
- `Move(X, Y)`: Move block X onto block Y
- `Move(X, Table)`: Move block X to table

### Simple Example

**Initial State:**
```
    C
    A       B
  ─────────────
     Table
```

**Goal State:**
```
    A
    B
    C
  ─────────────
     Table
```

**Means-Ends Analysis Solution:**

**Step 1: Identify Differences**
```
Current:  C on A,  A on Table,  B on Table
Goal:     A on B,  B on C,      C on Table

Differences:
- C should be on Table (not on A)
- B should be on C (not on Table)
- A should be on B (not on Table)
```

**Step 2: Select Operator to Reduce Difference**
First difference: C should be on Table
→ Operator: `Move(C, Table)`

**Step 3: Apply Operator**
```
New State:
    A       B       C
  ─────────────────────
        Table
```

**Step 4: Repeat**
```
Differences now:
- B should be on C (not on Table)
- A should be on B (not on Table)

Select: Move(B, C)

New State:
    A       B
            C
  ─────────────
     Table

Differences now:
- A should be on B (not on Table)

Select: Move(A, B)

Final State:
    A
    B
    C
  ─────────────
     Table  ✓ GOAL REACHED
```

### Complex Example: Four Blocks

**Initial State:**
```
A  B  C
   D
─────────
  Table
```

**Goal State:**
```
A
B
C
D
─────────
  Table
```

**Step 1: Calculate Differences**
```
Current Goal State:
A: on Table → should be on B (1 difference)
B: on D      → should be on C (1 difference)
C: on Table  → should be on D (1 difference)
D: on Table  → should be on Table (0 differences)

Total: 3 differences
```

**Step 2: Generate Possible Moves**
```
Can move: A (nothing on top)
          B (nothing on top)
          C (nothing on top)

Possible next states:
1. Move A onto B
2. Move A onto C
3. Move A onto D
4. Move B onto A
5. Move B onto C
6. Move C onto A
7. Move C onto B
8. Move C onto D
```

**Step 3: Evaluate Each Move (Count Remaining Differences)**
```
Move A onto D:  Still 3 differences
Move B onto C:  Only 2 differences remaining ✓ Best choice
Move C onto D:  Only 2 differences remaining ✓ Also good
```

**Step 4: Select Best Operator**
Choose `Move(B, C)` - reduces differences from 3 to 2

**Iteration:** Continue until differences = 0 (goal reached)

### Hitting an Obstacle

Sometimes means-ends analysis gets stuck:

**Problem Scenario:**
```
Initial: A on B, B on C, C on Table
Goal:    C on B, B on A, A on Table

Direct path blocked: Can't move A onto Table
                     (would temporarily increase differences)
```

**Solution:** Sometimes you must make moves that temporarily don't reduce differences, or even increase them, to reach the goal.

This is where **Problem Reduction** becomes necessary.

---

## 3. Problem Reduction

### Core Concept

**Problem Reduction** decomposes a complex problem into simpler subproblems:

1. Break hard problem into multiple easier problems
2. Solve each subproblem independently
3. Compose subsolutions into a solution for the whole problem

**Key Insight:** The right decomposition, guided by knowledge, makes complex problems tractable.

### Application to Blocks World

**Stuck Situation:**
```
Initial:
    D
A   B   C
─────────
  Table

Goal:
A
B
C
D
─────────
  Table
```

**Using Means-Ends Analysis alone:**
- All moves seem to increase differences or make no progress
- Need to temporarily worsen state to eventually improve it
- Hard to know which "worsening" moves are productive

**Using Problem Reduction:**

**Step 1: Decompose Goal**
Break goal into subgoals in reverse order:

```
Subgoal 1: Get D on Table
Subgoal 2: Get C on D
Subgoal 3: Get B on C
Subgoal 4: Get A on B
```

**Step 2: Solve Each Subgoal**

```
Subgoal 1: Get D on Table
  Current: D on B
  Action: Move(D, Table)

State after subgoal 1:
A   B   C   D
─────────────
    Table

Subgoal 2: Get C on D
  Current: C on Table
  Action: Move(C, D)

State after subgoal 2:
        C
A   B   D
─────────
    Table

Subgoal 3: Get B on C
  Current: B on Table
  Action: Move(B, C)

State after subgoal 3:
        B
        C
A       D
─────────
    Table

Subgoal 4: Get A on B
  Current: A on Table
  Action: Move(A, B)

Final State:
        A
        B
        C
        D
─────────────
    Table  ✓ GOAL REACHED
```

**Step 3: Compose Solution**
Complete sequence: Move(D,Table), Move(C,D), Move(B,C), Move(A,B)

### Why Problem Reduction Works

**Advantages:**
1. **Simplification:** Each subproblem is easier than the original
2. **Clear Progress:** Each subgoal provides measurable progress
3. **Knowledge-Driven:** Decomposition uses domain knowledge
4. **Avoids Local Minima:** Subgoals guide through temporarily worse states

**Requirements:**
1. **Good Decomposition:** Subproblems must be truly simpler
2. **Independence:** Solving one subproblem shouldn't undo others (or know how to handle it)
3. **Composability:** Subsolutions must combine into a complete solution

### Integration with Means-Ends Analysis

**Combined Approach:**
1. Use Means-Ends Analysis to make progress when possible
2. When stuck (no move reduces differences), use Problem Reduction
3. Decompose the problem into subgoals
4. Apply Means-Ends Analysis to each subgoal
5. Compose the subsolutions

This creates a powerful, hierarchical problem-solving method.

---

## 4. Production Systems

### What is a Cognitive Architecture?

A **cognitive architecture** is a fixed system structure that processes variable knowledge content to produce behavior:

```
Architecture + Content = Behavior
```

**Key Principle:** Keep architecture constant, change content (knowledge) to get different behaviors.

**Analogy:** Computer architecture:
- Fixed hardware (CPU, memory, buses)
- Variable software (programs, data)
- Different programs → Different behaviors

**For Cognitive Systems:**
- Fixed cognitive architecture
- Variable knowledge (rules, facts, memories)
- Different knowledge → Different behaviors

### Fundamental Assumptions

Cognitive architectures assume:

1. **Goal-Oriented:** Agents have goals and take actions to achieve them
2. **Rich Environment:** Agents operate in complex, dynamic environments
3. **Knowledge-Based:** Agents use knowledge of the world to pursue goals
4. **Abstraction:** Knowledge is at an appropriate abstraction level
5. **Symbolic:** Knowledge is captured in symbols at that abstraction
6. **Flexible:** Behavior depends on and adapts to the environment
7. **Learning:** Agents constantly learn from experience

### Structure of Production Systems

A production system consists of:

**1. Working Memory**
- Current state information (facts, percepts, goals)
- Active information being processed
- Changes rapidly as new information arrives

**2. Long-Term Memory**
Contains three types of knowledge:

**a) Procedural Knowledge (Production Rules)**
```
IF <conditions>
THEN <actions>
```

Example:
```
IF batter is left-handed
   AND first base is empty
   AND second base has runner
THEN pitch inside
```

**b) Semantic Knowledge (Conceptual Knowledge)**
- Facts about the world
- Concepts and their properties
- Organized in frames or networks

**c) Episodic Knowledge (Experiences)**
- Specific past events
- Personal experiences
- Stored as cases or episodes

### The Baseball Pitcher Example

**Situation:**
```
- 7th inning, top half
- Score: Tie game
- Runners on 2nd and 3rd base
- 2 outs
- Batter: Parra (left-handed)
- First base: Empty
```

**Working Memory (Current State):**
```
┌─────────────────────────────┐
│ inning: 7th                 │
│ half: top                   │
│ runners: 2nd, 3rd          │
│ outs: 2                     │
│ batter: Parra              │
│ batter-hand: left          │
│ first-base: empty          │
│ score: tied                │
│ goal: escape-inning        │
└─────────────────────────────┘
```

**Production Rules (Procedural Knowledge):**
```
Rule 1:
IF runners on 2nd AND 3rd
   AND first base empty
   AND batter is strong
THEN suggest walk-batter

Rule 2:
IF runners on 2nd AND 3rd
   AND batter is strong
THEN suggest pitch

Rule 3:
IF batter is left-handed
   AND pitch suggested
THEN dismiss throw-fastball

Rule 4:
IF throw-fastball dismissed
   AND pitch suggested
THEN suggest throw-curveball
```

### Production System Execution Cycle

```
1. Match Rules to Working Memory
   ↓
2. Select Rule(s) to Fire
   ↓
3. Execute Rule Actions
   ↓
4. Update Working Memory
   ↓
5. Repeat (or stop if goal achieved)
```

**Detailed Example:**

**Cycle 1:**
- Match: Rules 1 and 2 match current working memory
- Select: Both fire
- Execute: Add "walk-batter" and "pitch" to working memory
- Update: Working memory now contains two suggested operators

**Cycle 2:**
- Match: Rules 3 and 4 might match
- Issue: Two operators suggested, but which to choose?
- **IMPASSE:** Cannot decide between walk-batter and pitch

### Action Selection

When multiple operators are suggested:

**Option 1: Meta-Rules**
- Additional rules to choose among operators
- Example: "IF tied game THEN prefer walk-batter"

**Option 2: Priorities**
- Rules have assigned priorities
- Higher priority rules fire first

**Option 3: Specificity**
- More specific rules take precedence
- Example: Specific to this batter vs. general pitching strategy

**Option 4: Learning (Chunking)**
- When impasse occurs, learn a new rule
- Use past experience to break impasse

---

## 5. Chunking: Learning in Production Systems

### What is Chunking?

**Chunking** is a learning mechanism that creates new production rules when the system reaches an impasse.

**Impasse:** Situation where the system cannot make a decision because:
- No rules match current state, OR
- Multiple rules suggest conflicting actions

### How Chunking Works

**Step 1: Detect Impasse**
```
Working Memory shows:
- Operator 1 suggested: walk-batter
- Operator 2 suggested: pitch (throw-fastball or throw-curveball)
- Cannot choose between them
```

**Step 2: Search Episodic Memory**
Look for past experiences similar to current situation:

```
Episodic Memory:
┌──────────────────────────────────────────┐
│ Episode: Previous Game (5th inning)     │
│ - Weather: windy                         │
│ - Batter: Parra (left-handed)           │
│ - Situation: Similar (runners on base)  │
│ - Action: Threw fastball                │
│ - Result: HOME RUN (bad outcome)        │
└──────────────────────────────────────────┘
```

**Step 3: Extract Relevant Features**
Identify what's relevant from the episode:
- Batter identity: Parra ✓
- Batter handedness: Left ✓
- Pitch type: Fastball ✓
- Outcome: Negative ✓

Weather, inning → Less relevant

**Step 4: Create New Rule (Chunk)**
```
New Rule (Learned):
IF two operators suggested
   AND throw-fastball is one operator
   AND batter is Parra
THEN dismiss throw-fastball
```

**Step 5: Store in Procedural Memory**
New rule becomes part of permanent knowledge

**Step 6: Apply New Rule**
```
Re-run production system:
- Rules 1 & 2 fire → Suggest walk-batter and pitch
- New learned rule fires → Dismiss throw-fastball
- Rule 4 fires → Suggest throw-curveball
- Decision: Throw curveball (impasse resolved!)
```

### Characteristics of Chunking

**1. Impasse-Driven**
- Learning only when necessary
- Triggered by inability to decide
- Goal-directed learning

**2. Experience-Based**
- Uses episodic memory of past situations
- Learns from successes and failures
- Transfers knowledge across similar situations

**3. Knowledge Compilation**
- Converts episodic knowledge → procedural knowledge
- Encapsulates experience into reusable rules
- Speeds up future decision-making

**4. Incremental**
- One new rule at a time
- Gradually builds expertise
- Each impasse is a learning opportunity

### Reasoning, Learning, and Memory Integration

Chunking demonstrates the intimate connection between:

**Reasoning:**
- Production system tries to select action
- Reaches impasse during reasoning

**Learning:**
- Impasse triggers learning mechanism
- Creates new rule to prevent future impasses

**Memory:**
- Uses episodic memory to inform learning
- Stores new rule in procedural memory
- Memory provides foundation for both reasoning and learning

**Cycle:**
```
Reason → Reach Impasse → Learn from Episodes →
Store New Rule → Reason Better → (repeat)
```

This is exactly the unified theory of cognition that KBAI aims for!

---

## 6. Application to Ravens Progressive Matrices

### Using Production Systems

**Approach 1: Generic Production Rules**
Create rules that work for any Ravens problem:

```
Rule: IF two figures differ by rotation
      THEN apply same rotation to test figure

Rule: IF object moves from inside to outside
      THEN apply same transformation to answer
```

**Advantages:**
- Works across many problems
- Captures general visual reasoning

**Disadvantages:**
- Hard to enumerate all possible rules
- May not handle novel patterns

**Approach 2: Problem-Specific Rule Induction**
For each new problem, induce rules from the problem itself:

```
1. Receive problem (e.g., A, B, C and options)
2. Analyze A → B transformation
3. Create temporary rules encoding that transformation
4. Apply rules to C to generate expected D
5. Match against provided options
```

**Advantages:**
- Flexible, handles novel patterns
- Learns from each problem

**Disadvantages:**
- Must induce rules on-the-fly
- Requires good transformation detection

### Learning Component

Both approaches benefit from learning:

**Approach 1:** Learn new generic rules from problems solved:
```
After solving many "rotation" problems:
  → Learn: "Rotation patterns are common in Ravens tests"
  → Prioritize checking for rotations
```

**Approach 2:** Reuse induced rules when patterns repeat:
```
If previously encountered "inside → outside, expand":
  → Store as case
  → Retrieve when similar problem appears
  → Adapt rule to new context
```

The key insight: **Production rules can be learned, not just hand-coded**, making the system adaptive and improving with experience.

---

## Summary

### Key Takeaways

1. **Generate & Test** systematically explores solution spaces through generation of candidates and testing for validity. Balance intelligence between generator (creating candidates) and tester (evaluating them) based on problem characteristics.

2. **Means-Ends Analysis** is a goal-driven method that identifies differences between current and goal states, selects operators to reduce differences, and iteratively applies them. Most effective for problems with clear distance metrics.

3. **Problem Reduction** decomposes complex problems into simpler subproblems, solves each independently, and composes subsolutions. Essential when means-ends analysis alone gets stuck in local minima.

4. **Production Systems** are cognitive architectures with working memory (current state) and long-term memory (procedural, semantic, episodic knowledge). They execute through match-select-execute cycles using IF-THEN rules.

5. **Chunking** is a learning mechanism that creates new production rules when impasses occur, converting episodic experiences into procedural knowledge. Demonstrates the integration of reasoning, learning, and memory.

6. **Architecture + Content = Behavior** principle means fixing the architecture allows us to change behavior by changing knowledge content, simplifying both design and understanding of intelligent systems.

### Essential Principles

- Problem-solving methods couple with knowledge representations
- Balance computational effort between generation and testing
- Sometimes must temporarily increase distance to goal to ultimately reach it
- Learning is triggered by reasoning demands (impasse-driven)
- Episodic memory → Procedural memory transformation enables improvement
- Production systems provide unified framework for reasoning and learning

### Integration of Concepts

```
Semantic Networks (Representation)
         ↓
Generate & Test (Search)
         ↓
Means-Ends Analysis (Goal-Driven)
         ↓
Problem Reduction (Decomposition)
         ↓
Production Systems (Architecture)
         ↓
Chunking (Learning)
```

Each concept builds on previous ones, creating a complete cognitive system capable of reasoning and learning.

---

## See Also

- [[01-Fundamentals-KBAI|Fundamentals]] - Knowledge representations underlying these strategies
- [[03-Learning-Methods|Learning Methods]] - More sophisticated learning mechanisms
- [[04-Logic-and-Planning|Logic and Planning]] - Formal approaches to problem-solving
- [[00-README|Course Overview]] - Navigate the full course structure

---

*Problem-solving methods are most powerful when coupled with appropriate knowledge representations. The architecture provides the structure; knowledge provides the behavior.*
