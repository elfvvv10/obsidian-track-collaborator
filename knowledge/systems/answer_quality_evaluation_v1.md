# Answer Quality Evaluation & Iteration v1

## Summary
Establish a structured, repeatable process for evaluating and improving the quality of assistant responses in music collaboration workflows. v1 focuses on diagnosing failure modes such as drift, lack of specificity, weak implementation guidance, and poor context adherence, and iteratively refining prompt behavior and framework usage to produce consistently actionable, on-topic responses.

This system is evaluation-first, not automation-first. It does not change retrieval or task behavior directly; instead it creates a disciplined loop for improving response quality over time.

## Goals
- Reduce off-topic or drifting responses
- Increase implementation-level guidance ("how", not just "what")
- Improve adherence to track context and current user intent
- Maintain conversational continuity across turns
- Increase practical usefulness for finishing tracks

## Non-Goals (v1)
- No automatic scoring or model-based evaluation
- No changes to retrieval pipeline
- No changes to task system behavior
- No UI scoring dashboards

---

## 1. Evaluation Dimensions

Each assistant response should be evaluated across the following dimensions:

### 1.1 Relevance to User Intent
- Does the response directly address the user’s question?
- Does it stay focused on the current issue?

Failure examples:
- introduces unrelated production advice
- answers a different question than asked

---

### 1.2 Context Adherence
- Does the response use Track Context correctly?
- Does it stay aligned with:
  - genre
  - vibe
  - known issues
  - current section / goal

Failure examples:
- generic advice ignoring progressive house context
- contradicts known issues (e.g. suggests complexity when overwhelm is a problem)

---

### 1.3 Specificity
- Are suggestions concrete and precise?
- Are details actionable rather than vague?

Failure examples:
- “add more movement”
- “make transitions smoother”

Good examples:
- “automate low-pass filter from 120Hz to 18kHz over 16 bars before the drop”

---

### 1.4 Implementation Guidance (Critical)
- Does the response explain HOW to execute the idea?
- Are steps, tools, or parameters included?

Failure examples:
- “build tension into the drop”

Good examples:
- “mute kick for last 2 beats, add reverse reverb tail, then reintroduce kick on bar 1”

---

### 1.5 Structure & Clarity
- Is the response easy to follow?
- Is it organized into logical sections?

Failure examples:
- long unstructured paragraphs
- mixed ideas without grouping

---

### 1.6 Conversational Continuity
- Does the response build on prior turns?
- Does it stay anchored to the evolving discussion?

Failure examples:
- resets to generic advice
- ignores previous assistant/user exchange

---

### 1.7 Producer Value
- Does this actually help finish the track?
- Does it reduce overwhelm or increase clarity?

Failure examples:
- interesting but impractical ideas
- adds complexity without direction

---

## 2. Scoring System (Lightweight)

Each response can be scored:

- 2 = strong
- 1 = acceptable
- 0 = poor

Across all dimensions:

- Relevance
- Context Adherence
- Specificity
- Implementation Guidance
- Structure
- Continuity
- Producer Value

Optional:
- Track total score (0–14)
- Note primary failure mode

---

## 3. Test Prompt Set

Create a small, reusable set of prompts to test consistently.

### 3.1 Core Prompts (Examples)

- “My drop feels weak, what should I do?”
- “How do I transition from breakdown to drop?”
- “This track gets boring after one minute, how do I evolve it?”
- “Help me finish this track, I’m stuck in the arrangement”
- “How do I add movement without getting overwhelmed?”

---

### 3.2 Follow-Up Prompts

- “Can you be more specific?”
- “How exactly do I do that in practice?”
- “What should I do first?”
- “Give me a step-by-step plan”

---

### 3.3 Use With Track Context
Always test with:
- track_context.md loaded
- same track scenario for consistency

---

## 4. Evaluation Workflow

### Step 1: Run Prompt
- Use one test prompt
- Capture assistant response

### Step 2: Score Response
- Evaluate against all dimensions
- Identify weakest dimension

### Step 3: Identify Failure Pattern
Common patterns:
- too generic
- not enough “how”
- ignores context
- too verbose
- loses focus

### Step 4: Record Findings
Store:
- prompt
- response
- scores
- notes

(Optional: save in Obsidian)

---

## 5. Iteration Strategy

### 5.1 Focus on One Failure Type at a Time
Do not fix everything at once.

Example:
- iteration 1 → improve implementation guidance
- iteration 2 → improve relevance
- iteration 3 → improve continuity

---

### 5.2 Adjust Prompt System
Use findings to refine:
- PromptService system instructions
- framework injection wording
- ordering of internal blocks

---

### 5.3 Re-test Same Prompts
- compare before vs after
- confirm improvement

---

## 6. Known Priority Issues (From Current Testing)

- Responses drifting off-topic
- Too much “what”, not enough “how”
- Lack of structured execution steps
- Occasional loss of context across turns

These should be the first iteration targets.

---

## 7. Success Criteria

The system is improving when:

- Responses stay tightly aligned to the question
- Suggestions include clear execution steps
- Advice reflects track context consistently
- Follow-up questions feel natural and connected
- The assistant helps move the track forward, not just analyze it

---

## 8. Assumptions

- The current limitation is not retrieval, but prompt behavior and instruction clarity
- Framework + track context + chat state are sufficient inputs for high-quality responses
- Iterative refinement will produce better results than large one-time prompt changes

---

## 9. Next Steps (After v1)

- Integrate evaluation insights into prompt templates
- Strengthen critique framework with implementation patterns
- Revisit task system to align with improved response quality
- Consider lightweight automated scoring later (optional)