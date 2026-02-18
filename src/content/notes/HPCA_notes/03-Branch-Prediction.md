---
type: source
course: "[[HPCA (High Performance Computer Architecture)]]"
lessons: Lesson 4, Lesson 5
prerequisites: "[[02-Pipelining-and-Hazards]]"
---

# Branch Prediction and Predication

> **Prerequisites**: [[02-Pipelining-and-Hazards]]
> **Learning Goals**: Understand how branch mispredictions cost performance, and the spectrum of predictors from simple to sophisticated. Understand when predication is preferable to prediction.

---

## Why Branch Prediction Matters

- Branches account for **~20% of all instructions**
- **~60% of branches are taken**
- Mispredicting a branch wastes all instructions fetched down the wrong path

$$\text{CPI} = 1 + \frac{\text{Mispredictions}}{\text{Instructions}} \times \frac{\text{Penalty}}{\text{Misprediction}}$$

- Mispredictions/Instruction depends on **predictor accuracy**
- Penalty/Misprediction depends on **pipeline depth**

> The deeper the pipeline, the more important accurate branch prediction becomes.

---

## Every Processor Predicts

Options when encountering a branch:

| Approach | Branch Cost | Non-branch Cost |
|----------|-------------|-----------------|
| Refuse-to-predict (stall) | 3 cycles | 2 cycles |
| Predict not-taken | 1 cycle (if NT) or 3 (if taken) | 1 cycle |

> Every processor uses **some form of prediction** — refusing to predict is itself a prediction.

### Predict Not-Taken Accuracy
- 20% of instructions are branches; 60% of those are taken
- Predict Not-Taken is correct: 80% × 100% + 20% × 40% = **88% of all instructions**

---

## Branch Target Buffer (BTB)

The BTB stores the **target PC** for branches, indexed by the branch instruction's PC.

### BTB Steps
1. At Fetch, use current PC to look up BTB
2. Read predicted next PC from BTB
3. Compare predicted PC with actual PC computed in ALU stage
4. If match → correct prediction; if mismatch → misprediction, update BTB

**BTB Design Constraints**:
- Must have **1-cycle latency** → must be small
- Use **LSBs of PC** for indexing (only likely-soon instructions needed)

---

## Direction Prediction: Branch History Table (BHT)

The BHT predicts **whether a branch is taken or not-taken**, separately from the target address.

### 1-Bit Predictor
- Index BHT with LSB of PC
- Entry: `0` = not-taken, `1` = taken
- If BHT=0: just increment PC; if BHT=1: look up target in BTB

**Problem**: Each behavior change causes **2 mispredictions** (e.g., loop exit: predicts taken, misses; changes to not-taken, misses again on next loop entry).

### 2-Bit Predictor (2BP / 2BC)

Uses 2 bits per entry:
- **MSB** = prediction (taken or not-taken)
- **LSB** = conviction (strong or weak)

| State | Prediction | Conviction |
|-------|-----------|------------|
| `00` | Not-taken | Strong |
| `01` | Not-taken | Weak |
| `10` | Taken | Weak |
| `11` | Taken | Strong |

**State transitions**:

| Current State | Branch Outcome | Next State |
|---------------|---------------|------------|
| 00 | Not taken | 00 |
| 00 | Taken | 01 |
| 01 | Not taken | 00 |
| 01 | Taken | 10 |
| 10 | Not taken | 01 |
| 10 | Taken | 11 |
| 11 | Not taken | 10 |
| 11 | Taken | 11 |

- Single anomaly: **1 misprediction**
- Behavior change: **2 mispredictions**
- Usually initialized to `00` (easiest)

> **Note**: Every predictor has a sequence that causes every prediction to be wrong. More bits alone don't dramatically improve accuracy.

---

## History-Based Predictors

Use the **history of past branch outcomes** to predict patterns like TNTNTN... or TTNTTNTTN...

### 1-Bit History with 2-Bit Counters
- BHT entry: 1 history bit + two 2-bit counters (one for when last outcome was NT, one for T)
- Select which counter based on history bit

### N-Bit History Predictor
- Can predict **all patterns of length ≤ N+1**
- Cost: N + 2×2^N bits per entry (most counters wasted)

### Pattern History Table (PHT)
- Share counters across many BHT entries to reduce cost

---

## PShare vs. GShare

| Predictor | History | Counters | Best For |
|-----------|---------|----------|----------|
| **PShare** | Private (per-branch) | Shared | Small loops, predictable short patterns |
| **GShare** | Global (all branches) | Shared | Correlated branches |

> **In practice**: Use **both** together in a processor.

---

## Tournament Predictors

Combines two predictors and a **meta-predictor** that chooses which predictor to trust:

| GShare | PShare | Meta-Predictor Action |
|--------|--------|----------------------|
| Correct | Correct | No change |
| Correct | Incorrect | Count down (favor GShare) |
| Incorrect | Correct | Count up (favor PShare) |
| Incorrect | Incorrect | No change |

---

## Hierarchical Predictors

- **Tournament**: combines two *good* predictors; both always updated
- **Hierarchical**: combines one *good* + one *okay* predictor; the good predictor is only updated when the okay predictor is wrong → saves power

---

## Return Address Stack (RAS)

Different branch types need different prediction:

| Branch Type | Best Predictor |
|-------------|---------------|
| Conditional | BTB |
| Unconditional | BTB |
| Function Return (RET) | RAS |

**RAS**: A small hardware stack storing return addresses for function calls.
- On CALL: push return address onto RAS
- On RET: pop from RAS
- When full: **wrap-around** replacement

**Identifying RET early**: Use a predictor or **predecoding** from the prefetch stage.

---

## Predication

For branches that are **hard to predict** (e.g., small if-then-else), **predication** can be better than prediction.

**Predication**: Execute **both** paths; select the correct result; discard the wrong-path work.

### When to Use Each

| Scenario | Better Approach |
|----------|----------------|
| Loops | Prediction |
| Function calls/returns | Prediction |
| Large if-then-else | Prediction |
| Small if-then-else (hard to predict) | Predication |

### Conditional Move (MOVC)
- Compiler converts if-then-else to conditional instructions (e.g., `MOVC`)
- Requires:
  1. Compiler support
  2. More registers
  3. More instructions executed

**Full Predication HW Support**: Add **predicate bits** to every instruction telling the processor which predicate register qualifies the instruction.

---

## Summary

**Key Takeaways**:
- Branches are frequent (~20%) — misprediction penalty scales with pipeline depth
- BTB: predicts branch targets; BHT: predicts branch direction
- 2-bit predictors reduce mispredictions from behavior changes vs 1-bit
- History-based predictors (PShare, GShare) recognize patterns
- Tournament/hierarchical predictors combine strategies adaptively
- RAS handles function return prediction
- Predication eliminates hard-to-predict branches at the cost of executing both paths

**See Also**: [[02-Pipelining-and-Hazards]], [[04-ILP-and-Register-Renaming]]
**Next**: [[04-ILP-and-Register-Renaming]]
