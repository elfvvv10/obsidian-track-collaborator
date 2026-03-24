# Chat + To-Do System v1 — Specification

## Purpose

Transform the app from a single-turn question/answer tool into a session-based conversational collaborator that:

- supports multi-turn interaction
- maintains continuity across messages
- tracks actionable production tasks
- helps the user execute changes step-by-step
- focuses on finishing tracks, not just analyzing them

This system builds on:
- Track Critique Framework (how to think)
- Track Context System (what to think about)

---

## Core Concept

The system introduces two new layers:

### 1. Chat Layer
A persistent, multi-turn conversation within a session.

### 2. To-Do Layer
A structured, track-specific task list that evolves alongside the conversation.

---

## Core Design Principles

- Prioritize execution over analysis
- Keep interactions focused and actionable
- Maintain continuity within a session
- Avoid overwhelming the user
- Treat tasks as small, concrete actions
- Keep v1 simple and reliable

---

# Chat System (v1)

## Behavior

- The user can:
  - ask follow-up questions
  - refine previous suggestions
  - request clarification or implementation help

- The assistant should:
  - remember recent conversation context
  - respond in a conversational, collaborative tone
  - build on previous messages instead of restarting

---

## Session State

Each session should maintain:

- message history (recent turns only)
- selected workflow
- track_context_path (if provided)
- current to-do list (see below)

---

## Message Scope

To keep prompts efficient:

- include only the last N messages (e.g. 6–10 turns)
- do not include full history indefinitely
- prioritize:
  - most recent user requests
  - most recent assistant suggestions

---

## Prompt Integration

Prompt structure becomes:

1. System instructions  
2. Workflow instructions  
3. Critique framework (if applicable)  
4. Track context (if applicable)  
5. Recent conversation messages  
6. User query  

---

## Assistant Behavior Rules

When chat is active, the assistant should:

- reference previous steps naturally
- avoid repeating full critiques unless asked
- shift toward:
  - clarification
  - execution guidance
  - iteration

---

# To-Do System (v1)

## Purpose

Provide a simple, structured way to:

- capture actionable suggestions
- focus on one task at a time
- track progress during production
- reduce overwhelm

---

## Task Model

Each task should include:

- id
- text
- status (open | completed)
- source (assistant | user)
- created_at
- notes

---

## Task Types

Good tasks:
- Remove one support layer before drop
- Add snare fill into bar 65
- Automate filter opening in last 8 bars
- Shorten intro by 8 bars

Avoid vague tasks:
- Improve drop
- Make it better
- Fix arrangement

---

## Task Lifecycle

Tasks can be:

- created (by user or assistant suggestion)
- marked complete
- optionally edited or deleted

---

## UI Behavior (v1)

Display a To-Do panel (likely sidebar):

### Sections:
- Open Tasks
- Completed Tasks

### Actions:
- Add task manually
- Mark complete
- Delete task

---

## Assistant Interaction with Tasks

### v1 (simple)

- Assistant can suggest tasks
- Tasks are added manually or via simple extraction
- Assistant can reference tasks:
  - "Focus on task 2"
  - "Start with the first task"

### Not required in v1:
- automatic task extraction from every response
- complex planning agents
- multi-step decomposition

---

## Prompt Integration (Tasks)

When tasks exist, include them in the prompt as internal state:

CURRENT TASKS
- [ ] Remove support layer before drop
- [ ] Add snare fill into transition
- [x] Shorten intro by 8 bars

Rules:
- tasks are internal guidance, not sources
- tasks should influence prioritization
- tasks should not be cited as evidence

---

## Assistant Behavior with Tasks

When tasks exist, the assistant should:

- prioritize open tasks
- suggest next logical step
- avoid introducing too many new tasks at once
- help execute one task at a time
- acknowledge completed tasks

---

## Example Interaction

User:
"My drop still feels weak."

Assistant:
"Here are 3 focused fixes. I’ve added them as tasks."

Tasks:
- Remove one support layer before drop
- Add pre-drop tension riser
- Strengthen bass re-entry

User:
"Help me with the first one."

Assistant:
"Start by muting the pad and secondary arp in the last 4 bars..."

User:
"I did that, now it feels empty."

Assistant:
"Good — now add a bridging element..."

---

# System Integration

## Relationship to Existing Systems

- Framework → reasoning style  
- Track Context → project state  
- Chat → interaction continuity  
- To-Do → execution layer  

---

## Prompt Order (Final)

1. System instructions  
2. Workflow instructions  
3. Critique framework (if applicable)  
4. Track context (if applicable)  
5. Current tasks  
6. Recent messages  
7. User input  

---

## Failure Handling

If:
- no chat history → behave like current system
- no tasks → omit task section
- no context → omit context section

No errors should occur.

---

# v1 Scope (Important)

## Included

- multi-turn chat within a session
- message history (limited window)
- simple to-do list
- manual task management
- assistant-aware task context

---

## Not Included (Future)

- automatic task extraction
- cross-session persistence
- advanced planning agents
- full project lifecycle tracking
- DAW integration

---

# Future Extensions

- persistent task storage in vault
- task prioritization / tagging
- assistant-generated task breakdowns
- context-aware task suggestions
- integration with track versions or exports

---

## Key Outcome

This system enables:

- real back-and-forth collaboration
- structured execution of ideas
- reduced overwhelm
- higher likelihood of finishing tracks

---

## Guiding Principle

The assistant should behave like:

a producer sitting next to you, helping you decide what to do next, and how to do it