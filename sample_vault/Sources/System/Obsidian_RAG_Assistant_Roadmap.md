---
tags: [system, roadmap, core]
---

# Obsidian RAG Assistant — Roadmap (v1.0)

## 🎯 Core Vision
Transform the assistant into a conversational, session-aware creative collaborator for electronic music production.

---

## 🧱 Current System State
- Local RAG (Ollama + ChromaDB)
- Streamlit UI (Ask / Index / Debug)
- Retrieval filters, reranking, linked notes
- Auto-save to vault
- Hybrid/web fallback
- Chat provider abstraction (Ollama + OpenAI)

---

## 🚧 Phase Roadmap

### Phase 1 — Chat System Evolution (IN PROGRESS)
- Session-based chat history
- Context across turns
- Improved conversational UX

---

### Phase 2 — Track Context System (NEXT)
Core system for persistent track state.

```yaml
track_name:
genre:
bpm:
key:
vibe:
reference_tracks:

arrangement:
  intro:
  buildup:
  drop:
  breakdown:
  outro:

current_state:
  completed_sections:
  issues:
  next_steps:

sound_design:
  kick:
  bass:
  lead:
  drums:
  atmosphere:

notes:
```

---

### Phase 3 — Arrangement Critique
- Structured arrangement input
- Section-level feedback
- Energy + flow analysis

---

### Phase 4 — Producer Mindset Layer
- Assistant behaves like collaborator
- Asks questions, challenges decisions
- Suggests workflow improvements

---

### Phase 5 — Task System (DEFERRED)
- Track-linked to-dos
- Generated from conversations

---

### Phase 6 — Assistant-Aware Workflow
- Understands project stage
- Tracks progress over time

---

### Phase 7 — UX Transformation
- Cleaner UI
- Easier track selection
- Consumer-ready interface

---

## 🔁 Immediate Next Steps
1. Improve chat UX
2. Build Track Context system
3. Add critique mode

---

## 🧠 Key Principle
Everything revolves around Track Context.

---

## ⚠️ Anti-Patterns
- Don’t build features without persistence
- Don’t overbuild task system early
- Don’t rely on stateless chat

---

## 🚀 Codex Plan Anchor
“Add persistent Track Context system with load/save and prompt injection”
