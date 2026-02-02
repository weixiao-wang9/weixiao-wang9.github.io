---
type: note
course: "[[Knowledge-Based AI (CS 7637)]]"
date: 2026-02-02
topic: Language and Common Sense
lessons: 7, 14-16
prerequisites:
  - "[[01-Fundamentals-KBAI|Fundamentals of KBAI]]"
  - "[[02-Core-Reasoning-Strategies|Core Reasoning Strategies]]"
  - Understanding of semantic networks and frames
---

# Language Understanding and Common Sense Reasoning

## Prerequisites

- Understanding of semantic networks and knowledge representations
- Familiarity with frames as structured knowledge
- Knowledge of production systems
- Completed Fundamentals module

## Learning Goals

After completing this module, you will be able to:

1. Use frames with slots and fillers to represent stereotypical knowledge
2. Apply thematic role systems to understand sentence meaning
3. Perform common sense reasoning about actions and events
4. Identify primitive actions and implied actions in narratives
5. Use scripts to represent and reason about stereotypical event sequences
6. Generate expectations from scripts and handle track variations

---

## 1. Frames Representation

### What Are Frames?

**Frames** are structured knowledge representations that capture stereotypical situations with slots (properties) and fillers (values).

**Key Characteristics:**
- **Structured:** Multiple related pieces of information organized together
- **Stereotypical:** Represent common, typical situations
- **Default Values:** Slots have expected default fillers
- **Inheritance:** Can inherit from more general frames

**Comparison:**
- **Production Rules:** Atomic units (individual IF-THEN rules)
- **Frames:** Molecular units (structured packets of related knowledge)

### Frame Structure

```
Frame: EAT
├─ Agent: [Entity doing the eating]
├─ Object: [Thing being eaten]
├─ Location: [Where eating occurs]
├─ Time: [When eating occurs]
├─ Utensil: [Tool used for eating]
├─ Object-Is: [alive=false, location=inside-body]
└─ Mood: [happy (after eating)]
```

### Example: "Ashok ate a frog"

**Frame Instantiation:**
```
Frame: EAT (instance)
├─ Agent: Ashok
├─ Object: Frog
├─ Location: [unknown]
├─ Time: [past]
├─ Utensil: [unknown, default: fork/hands]
├─ Object-Is:
│  ├─ Alive: false (frog not alive when eaten)
│  └─ Location: inside-body (frog inside Ashok)
└─ Mood: [unknown, default: happy]
```

**What We Understand:**
From the sentence "Ashok ate a frog," frames help us infer:
- Ashok is the one who ate (agent role)
- The frog is what was eaten (object role)
- The frog was probably not alive when eaten (default expectation)
- The frog is now inside Ashok's body (consequence of eating)
- Ashok might be happier now (default mood change)

### Slots and Fillers

**Types of Fillers:**

**1. From Sentence (Explicit):**
```
"Ashok ate a frog" →
  Agent: Ashok (from subject)
  Object: Frog (from object)
```

**2. Default Values (Implicit):**
```
Object-Is alive: false (typical for human eating)
Mood: happy (people typically happier after eating)
```

**3. Inferred from Context:**
```
"Ashok ate a frog at home" →
  Location: home (from sentence)
  Time: mealtime (inferred from context)
```

**4. Unknown (To be filled):**
```
Utensil: ? (could be fork, spoon, hands, chopsticks)
```

### Multiple Frame Instances

**"David ate a pizza at home":**

```
Frame: EAT (instance 2)
├─ Agent: David
├─ Object: Pizza
├─ Location: home (explicitly stated)
├─ Time: [past]
├─ Utensil: [hands/fork, default for pizza]
├─ Object-Is:
│  ├─ Alive: false
│  └─ Location: inside-body
└─ Mood: satisfied
```

**Comparison:**
- Same frame structure (EAT)
- Different fillers (David vs. Ashok, pizza vs. frog)
- Some same defaults (alive=false, inside-body)
- Location explicit this time (home)

### Frames and Semantic Networks

**Equivalence:** Frames can be represented as semantic networks

**Semantic Network View:**
```
    X (circle) ←─── inside ───┐
        ↓                      │
      shape: circle         Y (square)
      size: small              ↓
      filled: yes          shape: square
                           size: large
```

**Frame View:**
```
Frame X:
├─ Shape: circle
├─ Size: small
├─ Filled: yes
└─ Inside: Y

Frame Y:
├─ Shape: square
├─ Size: large
└─ Contains: X
```

**Relationships Between Frames:**
```
Frame Y ←── inside ─── Frame X

Or within Frame X:
├─ Inside-of: Y  (points to another frame)
```

---

## 2. Understanding: Thematic Role Systems

### What is Understanding?

**Understanding** goes beyond parsing syntax to extracting meaning - identifying who did what to whom, where, when, and why.

**Key Challenge:** Natural language is ambiguous
- Same words, different meanings
- Different structures, same meaning
- Context-dependent interpretation

### Thematic Roles

**Thematic roles** identify the function each entity plays in an event:

**Core Roles:**
```
Agent: Who/what performs the action
Object/Theme: Who/what is affected by action
Recipient/Beneficiary: Who receives/benefits
Instrument: Tool used to perform action
Location: Where action occurs
Time: When action occurs
Source: Starting point
Destination: Ending point
```

### Example: "John gave Mary a book"

**Thematic Role Analysis:**
```
Event: GIVE
├─ Agent: John (giver)
├─ Recipient: Mary (receiver)
├─ Theme: book (thing given)
├─ Source: [John's possession]
├─ Destination: [Mary's possession]
└─ Time: [past]
```

**Understanding Achieved:**
- John initiated the transfer (agent)
- Mary received the object (recipient)
- The book changed possession (theme)
- Ownership transferred from John to Mary

### Resolving Ambiguity

**Ambiguous Verb Example:** "Run"

**Sentence 1:** "John ran the marathon"
```
Event: RUN (physical activity)
├─ Agent: John
├─ Theme: marathon (activity performed)
└─ Manner: [athletic, sustained]
```

**Sentence 2:** "John ran the company"
```
Event: RUN (manage/operate)
├─ Agent: John
├─ Theme: company (entity managed)
└─ Manner: [administrative, leadership]
```

**How to Resolve:**
1. Check selectional restrictions (marathons are run physically, companies are run administratively)
2. Look at object type (marathon vs. company)
3. Use context and world knowledge
4. Frames for different senses of "run"

### Constraints on Thematic Roles

**Selectional Restrictions:**

```
Frame: EAT
├─ Agent: MUST be animate entity
│         (animals, people, not rocks)
├─ Object: MUST be edible
│         (food, not concepts like "justice")
└─ Location: MUST be physical place
           (restaurant, not "happiness")
```

**Example Violations:**
```
"The rock ate happiness" ✗
  - Rock is not animate (agent violation)
  - Happiness is not edible (object violation)

"John ate the table" ✗
  - Table not typically edible (object violation)
  - But could be metaphorical or creative language
```

### The Earthquake Sentences

**Challenging Example:**

**Sentence 1:** "The earthquake destroyed the city"
```
Event: DESTROY
├─ Agent: earthquake (natural force)
├─ Theme: city (affected entity)
└─ Manner: physical destruction
```

**Sentence 2:** "The city was destroyed by the earthquake"
```
Event: DESTROY (passive voice)
├─ Theme: city (still the affected entity)
├─ Agent: earthquake (by-phrase indicates agent)
└─ Manner: physical destruction
```

**Understanding:**
- Same thematic roles despite different syntactic structures
- Passive voice changes word order but not roles
- "The city" is theme in both (receives action)
- "Earthquake" is agent in both (causes action)

**Lesson:** Syntax ≠ Semantics
- Surface form varies
- Deep meaning (thematic roles) remains constant

---

## 3. Common Sense Reasoning

### What is Common Sense Reasoning?

**Common sense reasoning** draws inferences about everyday situations that are obvious to humans but not explicitly stated.

**Example:**
```
Input: "John gave Mary a book"
Not stated: Mary now has the book
            John no longer has the book
Common sense: Ownership transferred
```

### Primitive Actions

**Primitive actions** are basic, atomic actions that compose complex events:

**For "Eating":**
```
Primitive Actions:
1. MOVE (hand to food)
2. GRASP (food with hand/utensil)
3. MOVE (food to mouth)
4. INGEST (food into mouth)
5. CHEW (break down food)
6. SWALLOW (move to stomach)
```

**Complex Action = Sequence of Primitives:**
```
EAT = MOVE + GRASP + MOVE + INGEST + CHEW + SWALLOW
```

**Why Primitives Matter:**
- Enable detailed understanding
- Support reasoning about action feasibility
- Allow partial action recognition
- Connect actions to physical constraints

### Implied Actions

**Implied actions** are actions not explicitly mentioned but necessary for stated events:

**Example:** "Ashok ate a frog"

**Stated Action:** Ate

**Implied Actions:**
```
Before eating:
- ACQUIRE (Ashok obtained a frog somehow)
- PREPARE (Ashok cooked/prepared the frog)
- MOVE-TO (Ashok brought frog to eating location)

During eating:
- All primitive actions of eating

After eating:
- DIGEST (food processing occurs)
- DISPOSE (waste elimination)
```

**Common Sense Inference:**
All these actions must have occurred even though not mentioned!

### Actions and Subactions

**Hierarchical Decomposition:**

```
EAT (complex action)
├─ ACQUIRE food
│  ├─ LOCATE food
│  ├─ MOVE to food
│  └─ TAKE food
├─ PREPARE food
│  ├─ COOK food
│  └─ SERVE food
├─ CONSUME food
│  ├─ MOVE to mouth
│  ├─ CHEW
│  └─ SWALLOW
└─ DIGEST food (automatic)
```

**Different Abstraction Levels:**
- High level: "Ashok ate"
- Medium level: "Ashok consumed prepared frog"
- Low level: "Ashok moved frog to mouth, chewed, swallowed..."

### State Changes from Actions

**Actions cause predictable state changes:**

**Action: GIVE(John, Mary, book)**

**State Changes:**
```
Before:
├─ Possess(John, book) = true
├─ Possess(Mary, book) = false
└─ Location(book) = with-John

After:
├─ Possess(John, book) = false
├─ Possess(Mary, book) = true
└─ Location(book) = with-Mary
```

**Action: EAT(Ashok, frog)**

**State Changes:**
```
Before:
├─ Alive(frog) = might be true
├─ Location(frog) = external to Ashok
└─ Hunger(Ashok) = high

After:
├─ Alive(frog) = false
├─ Location(frog) = inside Ashok
├─ Hunger(Ashok) = low
└─ Mood(Ashok) = satisfied
```

### Resultant Actions

**Some actions automatically trigger other actions:**

**Causal Chain:**
```
Action: PUSH(person, glass, edge-of-table)
  ↓
Resultant: FALL(glass)
  ↓
Resultant: BREAK(glass)
  ↓
Resultant: SCATTER(glass-pieces)
```

**Common Sense:**
- We infer the entire chain from "pushed glass off table"
- Don't need each step explicitly stated
- Physical laws and typical outcomes drive inference

---

## 4. Scripts

### What Are Scripts?

**Scripts** are structured representations of stereotypical event sequences - what typically happens in common situations.

**Components:**
- **Entry Conditions:** What must be true to start
- **Roles:** Actors participating in events
- **Props:** Objects used in events
- **Scenes:** Sequence of events/actions
- **Exit Conditions:** What's true when script ends

### Example: Restaurant Script

**Script: RESTAURANT**

**Entry Conditions:**
```
- Customer is hungry
- Customer has money
- Restaurant is open
```

**Roles:**
```
- Customer (C)
- Waiter (W)
- Cook (K)
- Cashier (Ca)
```

**Props:**
```
- Menu
- Table
- Food
- Check
- Money
```

**Scenes:**

**Scene 1: ENTERING**
```
1. C enters restaurant
2. C finds table
3. C sits at table
```

**Scene 2: ORDERING**
```
4. W brings menu to C
5. C reads menu
6. C decides on food
7. C tells W the order
8. W gives order to K
```

**Scene 3: EATING**
```
9. K prepares food
10. W brings food to C
11. C eats food
```

**Scene 4: EXITING**
```
12. W brings check to C
13. C goes to Ca
14. C pays Ca
15. C leaves restaurant
```

**Exit Conditions:**
```
- Customer no longer hungry
- Customer has less money
- Restaurant has customer's money
```

### Scripts Generate Expectations

**Story:** "John went to a restaurant. He ordered a burger. He left a tip."

**What We Understand (via script):**
```
Explicitly stated:
- John went to restaurant
- John ordered burger
- John left tip

Inferred via script:
- John was hungry (entry condition)
- John sat at a table (scene 1)
- Waiter took order to cook (scene 2)
- Cook prepared burger (scene 2)
- Waiter brought burger (scene 3)
- John ate the burger (scene 3)
- John paid for meal (scene 4)
- John is no longer hungry (exit condition)
```

**Power of Scripts:**
- Fill in missing details
- Generate expectations
- Enable understanding from minimal text
- Detect anomalies (deviations from script)

### Tracks: Script Variations

**Tracks** are variations of the same script for different contexts:

**Restaurant Script Tracks:**

**Track 1: FANCY RESTAURANT**
```
- Maître d' seats customers
- Multiple waiters
- Multiple courses
- Wine selection
- Extended duration
- Higher cost
```

**Track 2: FAST FOOD**
```
- Customer orders at counter
- No waiters
- Pick up own food
- Paper plates/plastic utensils
- Quick duration
- Lower cost
```

**Track 3: CAFETERIA**
```
- Customer carries tray
- Self-service food selection
- Pay before eating
- Communal tables
- Medium duration
```

**Same Core Script:**
- All involve ordering food, eating, paying
- Different specific sequences and props
- Different roles (waiter vs. no waiter)

### Learning Scripts

**How to learn scripts from stories:**

**Story 1:** "John went to a restaurant. He ordered pasta. He paid and left."

**Initial Script (Specific):**
```
1. Go to restaurant
2. Order pasta
3. Pay
4. Leave
```

**Story 2:** "Mary went to a restaurant. She ordered pizza. She paid and left."

**Generalized Script:**
```
1. Go to restaurant
2. Order <food>  (generalize: not always pasta)
3. Pay
4. Leave
```

**Story 3:** "Bob went to a restaurant. He ordered a salad. He read the menu first. He paid and left."

**Enhanced Script:**
```
1. Go to restaurant
2. [Read menu] (optional step discovered)
3. Order <food>
4. Pay
5. Leave
```

**Continued exposure → Rich, detailed scripts with:**
- Required steps
- Optional steps
- Alternative paths (tracks)
- Typical orderings
- Exception handling

### Using Scripts in AI

**Story Understanding:**
```
Input: "John went to a restaurant. Afterward, he felt full."

Script Activation: RESTAURANT script
Inference: John must have eaten (even though not stated)
Explanation: Eating is part of restaurant script,
            eating causes fullness
```

**Anomaly Detection:**
```
Input: "John went to a restaurant. He never ordered food. He left happy."

Script Violation: Ordering food is expected
Inference: Something unusual happened
Possible explanations:
- John met someone there (social, not eating)
- John works at restaurant
- Story is incomplete or has error
```

**Question Answering:**
```
Story: "Mary went to a restaurant. She ordered soup."
Question: "Did Mary eat?"
Answer: Yes (inferred via script)

Question: "Did a waiter take her order?"
Answer: Probably yes (typical in restaurant script)
```

---

## 5. Integration and Cognitive Connections

### Frames + Scripts

**Frames** represent stereotypical situations (static knowledge)
**Scripts** represent stereotypical event sequences (dynamic knowledge)

**Integration:**
```
Restaurant Script uses Frames:
- EAT frame (for eating scene)
- PAY frame (for payment scene)
- ORDER frame (for ordering scene)

Each scene in script can be represented as frames
Scripts organize frames into temporal sequences
```

### Common Sense Reasoning + Scripts

**Scripts embody common sense:**
- What typically happens in situations
- Normal orderings of events
- Expected outcomes
- Typical roles and props

**Common sense reasoning fills script gaps:**
- Implied actions between script steps
- Causal connections between events
- Reasoning about deviations and exceptions

### Thematic Roles + Frames

**Frames use thematic roles:**
```
Frame: EAT
├─ Agent: [corresponds to Agent role]
├─ Object: [corresponds to Theme role]
├─ Location: [corresponds to Location role]
└─ Utensil: [corresponds to Instrument role]
```

**Unified representation:**
- Thematic roles provide semantic categories
- Frames provide structured organization
- Together enable deep language understanding

### Top-Down and Bottom-Up Processing

**Bottom-Up (Data-Driven):**
```
Words → Parse → Identify verb → Activate frame → Fill slots
```

**Top-Down (Knowledge-Driven):**
```
Context → Activate script → Generate expectations → Interpret words
```

**Combined Processing:**
```
"John went to a restaurant"
  ↓ (bottom-up: parse, identify "restaurant")
RESTAURANT script activated
  ↓ (top-down: expect ordering, eating, paying)
"He ordered..."
  ↓ (matches expectation - confirms script)
Script guides interpretation of remaining story
```

### Cognitive Efficiency

**Why Scripts and Frames are Cognitively Efficient:**

**1. Chunking Information:**
- Don't reason about every detail
- Activate pre-packaged knowledge
- Process at higher abstraction level

**2. Default Reasoning:**
- Assume typical values unless contradicted
- Don't need everything stated explicitly
- Fill gaps automatically

**3. Fast Expectation Generation:**
- Scripts predict what comes next
- Enables anticipation and preparation
- Faster comprehension

**4. Error Detection:**
- Deviations from scripts/frames are salient
- Unusual events stand out
- Signals need for deeper processing

---

## Summary

### Key Takeaways

1. **Frames** are structured knowledge representations with slots (properties) and fillers (values) that capture stereotypical situations with default values. They organize related information into coherent, molecular units compared to atomic production rules.

2. **Thematic Role Systems** identify functional roles (agent, object, recipient, instrument, location, time) that entities play in events, enabling understanding beyond syntax. They resolve ambiguity through selectional restrictions and context.

3. **Common Sense Reasoning** infers unstated but obvious information about actions and events. Primitive actions compose complex actions; implied actions fill gaps; state changes follow from actions; and resultant actions cascade from initial actions.

4. **Scripts** represent stereotypical event sequences with entry conditions, roles, props, scenes, and exit conditions. They generate expectations, fill gaps in narratives, enable understanding from minimal text, and detect anomalies.

5. **Tracks** are variations of scripts for different contexts (fancy restaurant vs. fast food). Same core structure adapts to specific situations while maintaining fundamental event sequence.

6. **Integration:** Frames provide static stereotypical knowledge; scripts provide dynamic event sequences; thematic roles provide semantic categories; common sense reasoning fills gaps between all of them.

### Essential Principles

- **Stereotypes enable efficiency:** Pre-packaged knowledge avoids reasoning from scratch
- **Default values are cognitively cheap:** Assume typical unless contradicted
- **Structure matters:** Organized knowledge (frames/scripts) more useful than isolated facts
- **Expectation-driven:** Top-down processing guides bottom-up interpretation
- **Gaps are normal:** Language/narratives omit obvious information
- **Context resolves ambiguity:** Scripts and frames provide interpretive context

### Representations Hierarchy

```
Production Rules (Atoms)
    ↓
Frames (Molecules)
    ↓
Scripts (Sequences of Frames)
    ↓
Event Hierarchies (Networks of Scripts)
```

Each level adds structure and organization, enabling more sophisticated reasoning.

---

## See Also

- [[01-Fundamentals-KBAI|Fundamentals]] - Semantic networks as related representations
- [[02-Core-Reasoning-Strategies|Core Reasoning]] - Production systems and working memory
- [[06-Advanced-Reasoning|Advanced Reasoning]] - Explanation-based learning using scripts
- [[00-README|Course Overview]] - Navigate the full course structure

---

*Language understanding requires bridging syntax and semantics through structured knowledge. Frames and scripts capture the stereotypical knowledge that enables humans to understand with minimal explicit information.*
