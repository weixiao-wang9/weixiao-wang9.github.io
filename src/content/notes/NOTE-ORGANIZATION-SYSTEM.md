---
type: meta
date: 2026-02-01
---

# Universal Note Organization System

A flexible framework for organizing course notes and knowledge bases. This system can be adapted to any subject area from computer science to humanities.

> **Case Study**: This framework was applied to reorganize Operating System notes, but the principles work for any course.

---

## Core Philosophy

**The Problem**: Notes often grow organically and become:
- Too fragmented (dozens of tiny files)
- Too monolithic (one giant file)
- Poorly structured (no clear learning path)
- Hard to navigate (inconsistent naming, unclear relationships)

**The Solution**: Apply consistent organizational principles that balance:
- **Accessibility**: Easy to find information
- **Coherence**: Related concepts stay together
- **Scalability**: Structure works for 10 files or 100
- **Flexibility**: Adapts to different subjects

---

## Universal Principles

### Principle 1: Identify the Natural Structure

Every subject has an inherent organization. Discover it by asking:

#### For Sequential/Cumulative Subjects
**Questions**:
- What are the prerequisites? (What must be learned first?)
- What builds on what?
- What's the typical course progression?

**Examples**:
- **Programming**: Syntax → Control Flow → Functions → OOP → Design Patterns
- **Calculus**: Limits → Derivatives → Integrals → Series
- **Chemistry**: Atoms → Bonding → Reactions → Thermodynamics
- **OS**: Fundamentals → Processes → Threads → Scheduling → Memory

**Structure**: Use numbered sequential files (01, 02, 03...)

#### For Topic-Based/Modular Subjects
**Questions**:
- What are the major independent topics?
- How do they relate to each other (if at all)?
- Are there natural categories?

**Examples**:
- **History**: By period, by region, by theme
- **Psychology**: Cognitive, Social, Developmental, Clinical
- **Data Structures**: Arrays, Trees, Graphs, Hash Tables
- **World Literature**: By culture, by period, by genre

**Structure**: Use descriptive names with optional category prefixes
```
Cognitive-Psychology/
  Attention.md
  Memory.md
  Language.md

Social-Psychology/
  Group-Dynamics.md
  Persuasion.md
```

#### For Reference/Lookup Material
**Questions**:
- Will this be accessed randomly or sequentially?
- Is it meant for quick lookup?
- Does order matter?

**Examples**:
- API documentation
- Language syntax references
- Formula sheets
- Command cheatsheets

**Structure**: Alphabetical or categorical, optimized for search

---

### Principle 2: Balance Granularity

**The Goldilocks Rule**: Files should be neither too large nor too small.

#### Size Guidelines

| File Size | Status | Action |
|-----------|--------|--------|
| < 2KB | Too small (fragmented) | Consider consolidating |
| 2-5KB | Small but acceptable | OK if truly standalone |
| 5-20KB | Ideal range | Perfect |
| 20-30KB | Getting large | Consider splitting |
| > 30KB | Too large | Definitely split |

**Adjust these thresholds based on**:
- Density of content (code vs. prose)
- Topic complexity
- Personal preference

#### Consolidation Triggers

Merge multiple small files when:
- ✅ Files are always studied together
- ✅ One concept depends on another
- ✅ They're variations of the same theme
- ✅ Constant switching between files is annoying

**Example** (OS):
```
Before: 11 separate scheduling algorithm files (0.5-2KB each)
After: One scheduling file with sections (13KB)
Reason: Algorithms are compared against each other
```

**Example** (Data Structures):
```
Before: Binary-Tree.md, AVL-Tree.md, Red-Black-Tree.md, B-Tree.md
After: Trees.md with sections for each type
Reason: All share common tree concepts
```

#### Splitting Triggers

Split a large file when:
- ✅ File exceeds 25-30KB
- ✅ Multiple distinct topics in one file
- ✅ Different abstraction levels (basics vs. advanced)
- ✅ You find yourself scrolling excessively

**Example** (OS):
```
Before: GIOS.md (29KB) - OS basics, processes, threads, sync
After:
  - 01-Fundamentals.md (8KB)
  - 02-Processes-and-Threads.md (16KB)
  - 03-Synchronization.md (8KB)
Reason: Each topic is substantial and independent
```

**Example** (Calculus):
```
Before: Derivatives.md (35KB) - rules, applications, optimization
After:
  - 03-Derivatives-Basics.md (12KB)
  - 04-Derivatives-Applications.md (14KB)
  - 05-Optimization.md (10KB)
Reason: Natural breakpoints, different skill levels
```

---

### Principle 3: One Clear Purpose Per File

Each file should answer a specific question or cover a specific topic.

#### Test Your File Focus

Ask: "What is this file about?"
- ✅ Good: "How CPU scheduling works"
- ✅ Good: "French verb conjugation rules"
- ❌ Bad: "Computer stuff"
- ❌ Bad: "Week 3 notes"

#### File Naming Formula

```
[Prefix-]Descriptive-Topic-Name.md
```

**Components**:
- **Prefix**: Number (01-99) for sequential content, category for topics
- **Descriptive**: Clear indication of content
- **Separators**: Use hyphens (more readable than underscores or spaces)
- **Capitalization**: Title Case or kebab-case (be consistent)

**Examples**:

Sequential (numbered):
- `01-Introduction-to-Python.md`
- `02-Control-Flow.md`
- `03-Functions.md`

Categorical:
- `Grammar-Articles.md`
- `Grammar-Pronouns.md`
- `Vocab-Medical-Terms.md`

Standalone:
- `Quick-Reference.md`
- `Common-Mistakes.md`
- `Practice-Problems.md`

---

### Principle 4: Create Clear Navigation

**Readers should never be lost.**

#### Essential Navigation Elements

**1. README/Index File** (name it `00-README.md` or `INDEX.md`)

Must include:
- **Learning path**: Where to start, what order to follow
- **File descriptions**: Brief summary of each file
- **Quick reference**: Key concepts, formulas, or terms
- **Context**: How files relate to each other

**Template**:
```markdown
# [Subject] Study Guide

## Learning Path

### Prerequisites
Before starting, you should know: [X, Y, Z]

### Recommended Study Order
1. [File 1] - [What it covers]
2. [File 2] - [What it covers]
...

### Independent Topics
These can be studied in any order:
- [File A] - [Description]
- [File B] - [Description]

## Quick Reference
- **Key Concept 1**: Brief definition
- **Key Concept 2**: Brief definition

## File Descriptions

### [File 1 Name]
**Topics**: [Topic list]
**Key Learning Goals**: What you'll understand after reading

### [File 2 Name]
...
```

**2. Internal Cross-References**

Link related concepts:
```markdown
See also: [[Related-Topic]]
Prerequisites: [[Foundation-Topic]]
Next: [[Advanced-Topic]]
```

**3. Section Headers**

Use consistent hierarchy:
```markdown
# Main Topic (H1) - File title, once per file
## Major Section (H2) - Main divisions
### Subsection (H3) - Detailed topics
#### Detail (H4) - Use sparingly
```

---

### Principle 5: Optimize for Your Learning Style

**The structure should serve YOU.**

#### Different Learning Styles → Different Structures

**Visual/Spatial Learners**:
- Use folders/subdirectories for major categories
- Include diagrams and visual aids in files
- Create concept maps in README

**Sequential/Linear Learners**:
- Number files strictly
- Clear "previous/next" links
- Learning path is critical

**Global/Holistic Learners**:
- Comprehensive README with overview
- Show connections between topics
- Include summary sections

**Hands-on/Kinesthetic Learners**:
- Separate theory from practice
- Create dedicated practice/examples folders
- Include exercises within theory files

---

## Reorganization Workflow

Use this process for any subject area:

### Phase 1: Audit (Don't Edit Yet!)

**Step 1: Inventory**
```bash
# List all files with sizes
ls -lh *.md | awk '{print $5, $9}'

# Or count lines in each file
wc -l *.md
```

**Step 2: Identify Problems**

Check for:
- [ ] Files > 25KB (too large)
- [ ] Files < 2KB (too small/fragmented)
- [ ] Index-only files (no actual content)
- [ ] Inconsistent naming
- [ ] Unclear organization
- [ ] Duplicate content

**Step 3: Map Content**

Create a visual map:
```
Topic A (fundamental)
  ├── Subtopic 1
  └── Subtopic 2

Topic B (requires Topic A)
  ├── Subtopic 3
  └── Subtopic 4

Topic C (independent)
  └── Subtopic 5
```

Ask yourself:
- What are the 5-10 major topics?
- What are the dependencies?
- What belongs together?
- What's independent?

**Step 4: Choose Organization Type**

Based on your subject's nature:

| Subject Type | Organization | Example |
|--------------|--------------|---------|
| Sequential/cumulative | Numbered files | Math, Programming, OS |
| Topic-based modules | Category-prefixed | Psychology, History |
| Reference/lookup | Alphabetical | API docs, Vocabulary |
| Mixed | Hybrid (numbered main + topic appendices) | Many courses |

---

### Phase 2: Design

**Step 5: Create Structure Plan**

Write out your target structure:

**For Sequential Course**:
```
00-README.md (navigation)
01-Fundamentals.md
  - Concept A
  - Concept B
02-Intermediate.md
  - Concept C (requires A, B)
  - Concept D
03-Advanced.md
  - Concept E (requires C, D)

Appendix-Examples.md (unordered)
Appendix-Quick-Reference.md
```

**For Topic-Based Course**:
```
00-README.md
Topic-A/
  Overview.md
  Subtopic-1.md
  Subtopic-2.md
Topic-B/
  Overview.md
  Subtopic-3.md
Reference/
  Glossary.md
  Formulas.md
```

**Step 6: Estimate Sizes**

For each planned file, estimate:
```
01-Fundamentals.md
  Current: 3 files totaling 12KB
  Target: ~12KB (consolidate into one)

02-Applications.md
  Current: Part of 35KB monolithic file
  Target: ~15KB (extract and organize)
```

**Step 7: Plan Content Migration**

Map old files to new files:
```
Old File → New Location
-------------------------
GIOS.md (lines 1-162) → 01-Fundamentals.md
GIOS.md (lines 163-300) → 02-Processes.md
GIOS.md (lines 301-570) → 02-Processes.md (Threads section)

FCFS.md → 04-Scheduling.md (Section 2.1)
RR.md → 04-Scheduling.md (Section 2.2)
Priority.md → 04-Scheduling.md (Section 2.3)
```

---

### Phase 3: Execute

**Step 8: Create New Files**

⚠️ **Don't delete old files yet!**

For each new file:
1. Create file with proper naming
2. Add frontmatter/metadata
3. Copy relevant content from old files
4. Reorganize with clear sections
5. Add internal navigation (See also, Prerequisites)
6. Ensure smooth flow

**Step 9: Create README**

Write your navigation file:
- Learning path
- File descriptions
- Quick reference
- Study tips

**Step 10: Verify Migration**

Checklist:
- [ ] All content from old files accounted for?
- [ ] No broken internal links?
- [ ] File sizes in ideal range (5-20KB)?
- [ ] Clear progression/organization?
- [ ] Consistent formatting?
- [ ] README is helpful?

**Step 11: Clean Up**

Only after verification:
- Archive old files (move to `_archive/` folder)
- Or delete old files (if backed up elsewhere)
- Update any external references

---

### Phase 4: Maintain

**When Adding New Content**

| New Content Type | Action |
|-----------------|--------|
| New concept in existing topic | Add to appropriate file, maintain section order |
| New subtopic | Add new section to existing file, update README |
| New major topic | Create new numbered file, insert in sequence |
| Examples/exercises | Add inline if short, or create dedicated Examples/ folder |

**When Content Grows**

File approaching size limit (20-25KB):
1. Identify natural split point
2. Create new file
3. Move content
4. Add cross-references
5. Update README

**Regular Maintenance** (every 3-6 months):
- Check file sizes
- Look for orphaned content
- Verify all links work
- Update README if structure changed
- Archive outdated content

---

## Subject-Specific Adaptations

### Computer Science Courses

**Structure**:
```
00-README.md
01-Fundamentals.md
02-Core-Concept-1.md
03-Core-Concept-2.md
...
XX-Advanced-Topics.md
Examples/
  example-1.py
  example-2.py
Exercises/
  problem-set-1.md
  problem-set-2.md
```

**Special considerations**:
- Separate code examples into files
- Include algorithmic complexity
- Link to external resources (documentation)

---

### Mathematics Courses

**Structure**:
```
00-README.md
01-Foundations.md (definitions, axioms)
02-Theory-1.md (theorems, proofs)
03-Theory-2.md
04-Applications.md (problem-solving)
Formula-Sheet.md (quick reference)
Practice-Problems.md
```

**Special considerations**:
- Separate proofs from intuition
- Include worked examples
- Formula sheet for quick lookup

---

### Language Courses

**Structure**:
```
00-README.md
Grammar/
  01-Articles.md
  02-Verbs.md
  03-Pronouns.md
Vocabulary/
  Food.md
  Travel.md
  Business.md
Practice/
  Exercises-Level-1.md
  Exercises-Level-2.md
```

**Special considerations**:
- Separate grammar from vocabulary
- Organize vocab by theme
- Include practice sections

---

### History/Humanities Courses

**Structure Options**:

**Chronological**:
```
00-README.md
01-Ancient-Period.md
02-Medieval-Period.md
03-Modern-Period.md
Themes/
  Economic-History.md
  Social-History.md
```

**Thematic**:
```
00-README.md
Political-History/
  Democracy.md
  Revolution.md
Economic-History/
  Trade.md
  Industrialization.md
```

**Special considerations**:
- Include timelines
- Cross-reference between themes and periods
- Maps and context

---

### Science Courses (Physics, Chemistry, Biology)

**Structure**:
```
00-README.md
01-Fundamental-Concepts.md
02-Laws-and-Principles.md
03-Applications.md
04-Lab-Techniques.md
Appendix-Formulas.md
Appendix-Constants.md
```

**Special considerations**:
- Separate theory from application
- Include diagrams
- Unit conversions and constants

---

## Common Patterns and Templates

### Pattern 1: Layered Structure (Simple → Advanced)

Use when: Subject has clear difficulty progression

```
01-Basics.md (definitions, simple examples)
02-Intermediate.md (builds on basics)
03-Advanced.md (complex topics)
04-Expert.md (cutting-edge, special cases)
```

---

### Pattern 2: Categorical Structure

Use when: Independent topics that can be studied in any order

```
Category-A/
  Topic-1.md
  Topic-2.md
Category-B/
  Topic-3.md
  Topic-4.md
```

---

### Pattern 3: Hybrid Structure

Use when: Some prerequisites, but also independent topics

```
Core/ (sequential)
  01-Foundations.md
  02-Building-Blocks.md
  03-Integration.md

Topics/ (any order)
  Topic-A.md
  Topic-B.md
  Topic-C.md

Reference/
  Quick-Reference.md
  Glossary.md
```

---

### Pattern 4: Problem-Solution Structure

Use when: Subject is problem-focused

```
00-README.md
Problems/
  Problem-Type-1.md
  Problem-Type-2.md
Techniques/
  Technique-A.md
  Technique-B.md
Solutions/
  Worked-Examples.md
  Practice-Solutions.md
```

---

## File Content Template

### For Sequential Topic Files

```markdown
---
type: source
course: "[[Course Name]]"
prerequisites: "[[Previous-Topic]]"
---

# [Topic Name]

> **Prerequisites**: [[Topic A]], [[Topic B]]
> **Learning Goals**: After reading this, you will understand X, Y, Z

## Introduction
Brief overview of what this topic is and why it matters

## Fundamental Concepts

### Concept 1
- Definition
- Key properties
- Simple examples

### Concept 2
...

## How It Works

### Mechanism
Detailed explanation of the process

### Examples
Concrete examples applying the concepts

## Advanced Topics

### Special Cases
Edge cases, exceptions, nuances

### Common Mistakes
Pitfalls to avoid

## Real-World Applications

### Use Case 1
### Use Case 2

## Summary

**Key Takeaways**:
- Point 1
- Point 2
- Point 3

**Common Patterns**:
- Pattern A
- Pattern B

**See Also**: [[Related-Topic-1]], [[Related-Topic-2]]
**Next**: [[Following-Topic]]
```

---

### For Reference Files

```markdown
---
type: reference
course: "[[Course Name]]"
---

# [Reference Name]

Quick lookup for [specific purpose]

## Category 1

| Item | Description | Example |
|------|-------------|---------|
| A | ... | ... |
| B | ... | ... |

## Category 2

### Subcategory 2.1
- Item 1
- Item 2

### Subcategory 2.2
...

## See Also
[[Main-Topic-File]]
```

---

## Success Metrics

### Quantitative Metrics

- **File count**: 5-15 files for most courses (excluding examples/exercises)
- **Average file size**: 8-15KB
- **Min file size**: > 3KB (unless it's meta/reference)
- **Max file size**: < 25KB
- **Hierarchy depth**: 2-3 levels (file → section → subsection)

### Qualitative Metrics

Ask yourself:

- **Navigation**: Can I find what I need in < 30 seconds?
- **Context**: Is it clear where to start?
- **Flow**: Can I read through without jumping around?
- **Coherence**: Do related concepts stay together?
- **Maintainability**: Can I add new content easily?
- **Review**: Can I quickly refresh on a topic?

### User Testing

Ask someone unfamiliar with your notes:
1. "Where would you start learning this subject?"
2. "Find information about [specific concept]"
3. "What comes after [topic X]?"

If they struggle, your structure needs work.

---

## Troubleshooting Common Issues

### Issue: "I don't know where to split a large file"

**Solution**: Look for natural boundaries
- Changes in abstraction level (basics → advanced)
- Different types of content (theory → practice)
- Topic shifts (where a new H2 section starts a completely different discussion)
- Places where you pause during study

### Issue: "My files are too small but don't seem related"

**Solution**:
1. Check if they're subtopics of a larger theme
2. Look for shared prerequisites
3. Consider if they're studied together
4. If truly independent, keep separate but organize in folders

### Issue: "My subject doesn't fit sequential or categorical"

**Solution**: Use hybrid structure
- Core path (sequential)
- Optional topics (categorical)
- Reference materials (alphabetical)

### Issue: "Content keeps growing, hitting size limits"

**Solution**:
1. Move examples to separate Examples/ folder
2. Move exercises to Practice/ folder
3. Extract advanced topics to separate file
4. Create appendices for reference material

### Issue: "Too much overlap between files"

**Solution**:
- Choose one canonical location for each concept
- Use links/references from other locations
- If genuinely needed in multiple places, keep one detailed, others brief with links

---

## Advanced Techniques

### Technique 1: Progressive Disclosure

Start simple, add detail in later files:

```
01-Introduction.md (high-level overview)
02-Basics.md (practical, simplified)
03-Theory.md (full formal treatment)
04-Advanced.md (edge cases, optimization)
```

### Technique 2: Spiral Learning

Revisit topics at increasing depth:

```
01-Survey.md (touch on everything briefly)
02-Core-Topic-1-Detailed.md
03-Core-Topic-2-Detailed.md
04-Integration.md (brings topics together)
```

### Technique 3: Hub and Spoke

Central file with specialized branches:

```
Core-Concept.md (central hub)
  → links to →
    Application-1.md
    Application-2.md
    Advanced-Theory.md
    Historical-Context.md
```

### Technique 4: Layered Folders

Separate beginner, intermediate, advanced:

```
Beginner/
  01-Start-Here.md
  02-Basics.md
Intermediate/
  03-Building-On-Basics.md
  04-Common-Patterns.md
Advanced/
  05-Advanced-Topics.md
  06-Research-Frontiers.md
```

---

## Conclusion

Good organization is:
- **Personal**: Fits your learning style
- **Purposeful**: Serves your goals
- **Flexible**: Adapts as content grows
- **Consistent**: Follows clear principles
- **Maintainable**: Easy to update

**There is no one perfect structure.** The best organization is one that:
1. Makes sense to you
2. Scales with your content
3. Helps you learn effectively
4. Reduces friction in finding information

Start with these principles, adapt them to your subject, and iterate based on what works.

---

## Quick Start Checklist

Starting a new course? Use this:

- [ ] Choose organization type (sequential/categorical/hybrid)
- [ ] List 5-10 major topics
- [ ] Map dependencies (what requires what)
- [ ] Decide on naming convention
- [ ] Create 00-README.md template
- [ ] Create first content file
- [ ] Set file size targets (aim 8-15KB)
- [ ] Establish review schedule

Reorganizing existing notes?

- [ ] Audit current files (sizes, organization)
- [ ] Identify problems (too large, too small, unclear)
- [ ] Map current content to new structure
- [ ] Create structure plan
- [ ] Migrate content (keep old files until verified)
- [ ] Create README
- [ ] Verify completeness
- [ ] Archive old files

---

*Last updated: 2026-02-01*
*This is a living document. Update it as you discover new patterns and techniques.*
