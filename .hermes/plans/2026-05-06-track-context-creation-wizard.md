# Track Context Creation Wizard — Implementation Plan

> **For Hermes:** Use codex skill to delegate this to Codex CLI. Run from the repo root with `pty=true`.

**Goal:** Add a "+ New Track" wizard to the Streamlit sidebar so first-time users can create a YAML Track Context through a friendly form instead of manually writing YAML files.

**Architecture:** Add a UI-only feature to streamlit_app.py. No new service files needed — reuses existing `TrackContextService.create_default_canonical_track_context()` and `TrackContextService.save_canonical_track_context()`. The wizard is a simple Streamlit form inside an expander, triggered by a button in the sidebar's YAML Track Context section.

**Tech Stack:** Python 3.11+, Streamlit, existing `services/track_context_service.py` and `services/models.py` (TrackContext dataclass).

**Non-goals:** Don't touch retrieval pipelines, don't change CLI, don't refactor sidebar layout beyond adding the button/form, don't scaffold Obsidian folders (out of scope), don't add sections support in the wizard (keep it simple — core identity fields only).

**Risk level:** Low. Self-contained UI addition. No core logic changes.

---

### Task 1: Add the "+ New Track" button to the sidebar

**Objective:** Place a visible button in the sidebar's YAML Track Context section, next to the existing "Start / Resume" and "Clear Active Track" buttons.

**Files:**
- Modify: `streamlit_app.py` (around line 214, in `_render_sidebar`, after the `load_col, clear_col` button row)

**Step 1: Add the button**

In `_render_sidebar()`, after the existing `load_col, clear_col = st.columns(2)` block (around line 214), insert a new button row:

```python
        # After the existing load/clear columns block, add:
        new_track_col, _ = st.columns([1, 1])
        new_track_clicked = new_track_col.button(
            "+ New Track",
            use_container_width=True,
            help="Create a new YAML Track Context with a guided form.",
            key="sidebar_new_track_wizard",
        )
        if new_track_clicked:
            st.session_state["show_track_wizard"] = True
```

This button should appear BETWEEN the "Start / Resume" | "Clear Active Track" row and the status captions that follow.

**Step 2: Initialize session state**

In `_init_session_state()` (around line 2223), add the new key:

```python
"show_track_wizard": False,
```

**Step 3: Verify button renders**

Run `streamlit run streamlit_app.py` and confirm the "+ New Track" button appears in the sidebar below the existing buttons. It should set `show_track_wizard` to True when clicked.

---

### Task 2: Build the creation form (wizard)

**Objective:** When `show_track_wizard` is True, render a Streamlit form below the button with fields for creating a new track context.

**Files:**
- Modify: `streamlit_app.py` (in `_render_sidebar`, after the new button)

**Step 1: Add the form rendering block**

After the button code from Task 1, add:

```python
        if st.session_state.get("show_track_wizard", False):
            with st.form("track_context_wizard", clear_on_submit=True, enter_to_submit=False):
                st.markdown("#### New Track Context")
                st.caption("Fill in what you know — you can edit everything later.")

                wizard_track_id = st.text_input(
                    "Track ID *",
                    key="wizard_track_id",
                    help="A short, URL-safe identifier for this track. Used as the filename. Example: moonlit_driver",
                    placeholder="my_new_track",
                )
                wizard_track_name = st.text_input(
                    "Title",
                    key="wizard_track_name",
                    help="The display name of your track.",
                    placeholder="My New Track",
                )
                wizard_genre = st.text_input(
                    "Genre",
                    key="wizard_genre",
                    help="Primary genre.",
                    placeholder="progressive house",
                )
                col1, col2 = st.columns(2)
                wizard_bpm = col1.text_input(
                    "BPM",
                    key="wizard_bpm",
                    placeholder="126",
                )
                wizard_key = col2.text_input(
                    "Key",
                    key="wizard_key",
                    placeholder="A minor",
                )
                wizard_vibe = st.text_input(
                    "Vibe (comma separated)",
                    key="wizard_vibe",
                    help="A few words describing the mood/energy.",
                    placeholder="driving, euphoric, dark",
                )
                wizard_issues = st.text_area(
                    "Known Issues (one per line, optional)",
                    key="wizard_issues",
                    help="Any problems you already know about.",
                    height=70,
                )

                wizard_col1, wizard_col2 = st.columns(2)
                submitted = wizard_col1.form_submit_button("Create Track", use_container_width=True, type="primary")
                cancelled = wizard_col2.form_submit_button("Cancel", use_container_width=True)
```

**Step 2: Handle form submission**

After the form block (still inside the `if show_track_wizard:` block), add:

```python
            if submitted:
                track_id = wizard_track_id.strip()
                if not track_id:
                    st.error("Track ID is required.")
                else:
                    # Build TrackContext from form fields
                    from services.models import TrackContext
                    bpm_value = None
                    if wizard_bpm.strip():
                        try:
                            bpm_value = int(wizard_bpm.strip())
                        except ValueError:
                            st.error("BPM must be a number.")
                            bpm_value = None
                    if bpm_value is not None or not wizard_bpm.strip():
                        new_context = TrackContext(
                            track_id=track_id,
                            track_name=wizard_track_name.strip() or None,
                            genre=wizard_genre.strip() or None,
                            bpm=bpm_value,
                            key=wizard_key.strip() or None,
                            vibe=_split_csv(wizard_vibe),
                            known_issues=_split_lines(wizard_issues),
                        )
                        query_service.track_context_service.save_canonical_track_context(new_context)
                        _load_or_create_track_context(query_service, track_id)
                        st.session_state["show_track_wizard"] = False
                        st.session_state["advanced_track_context_track_id"] = track_id
                        st.success(f"Track '{track_id}' created and loaded.")
                        st.rerun()

            if cancelled:
                st.session_state["show_track_wizard"] = False
                st.rerun()
```

Note: `_split_csv()` and `_split_lines()` are already defined elsewhere in `streamlit_app.py` — reuse them.

**Step 3: Verify form works**

Run the app, click "+ New Track", fill in the form with test data, submit. Verify:
- A new `.yaml` file appears in `<output_path>/track_contexts/<track_id>.yaml`
- The track is loaded as the active track
- The wizard closes and the edit form shows the new track data

---

### Task 3: Handle edge cases and cleanup

**Objective:** Make the wizard robust and well-behaved in edge cases.

**Files:**
- Modify: `streamlit_app.py`

**Step 1: Reset wizard state when clearing active track**

In `_clear_active_track_context()` (around line 1235), add:

```python
    st.session_state["show_track_wizard"] = False
```

**Step 2: Reset wizard state on form reset**

In `_render_ask_tab()` where `reset_ask_form` is handled (around line 397), add `"show_track_wizard"` to the list of reset keys:

```python
"show_track_wizard": False,
```

Actually, this should be in the `st.session_state[key] = ""` loop — but since it's a boolean not a string, add it separately after the loop:

```python
        st.session_state["show_track_wizard"] = False
```

Wait — looking at the code more carefully, the reset block sets keys to `""`. `show_track_wizard` is a boolean. Let me handle it differently — add it to the explicit resets after the loop.

Let me re-read lines 397-439 to be precise.

I'll handle this in the plan by adding a line after the loop.

**Step 3: Close wizard when loading an existing track via "Start / Resume"**

In the `if load_clicked` block (around line 229), add:

```python
            st.session_state["show_track_wizard"] = False
```

**Step 4: Verify edge cases**

- Create a track, then click "Clear Active Track" → wizard should be hidden
- Create a track, then load a different one via "Start / Resume" → wizard should close
- Open wizard, cancel → goes away, no file created
- Submit with empty Track ID → error shown, no file created
- Submit with non-numeric BPM → error shown
- Submit with existing track_id → overwrites (this is acceptable behavior — the service's `save_canonical_track_context` handles it)

---

### Task 4: Review and run tests

**Objective:** Verify nothing is broken and the feature works end-to-end.

**Step 1: Run existing test suite**

```bash
python3 -m unittest discover -s tests
```

All existing tests must pass.

**Step 2: Manual smoke test**

1. Launch `streamlit run streamlit_app.py`
2. In sidebar, click "+ New Track"
3. Fill in: ID=test_track, Title=Test Track, Genre=techno, BPM=140, Key=D minor, Vibe=dark driving, Issues=weak kick
4. Click "Create Track"
5. Verify: track loads, edit form shows values, YAML file created at `sample_vault/Saved Outputs/track_contexts/test_track.yaml`
6. Ask a question with the track loaded: `python main.py ask "How can I improve the kick?" --track-id test_track --use-track-context`
7. Verify the answer references your track

**Step 3: Check the diff**

```bash
git diff streamlit_app.py
```

Review that the additions are clean, no stray changes.

---

### Task 5: Commit

```bash
git add streamlit_app.py
git commit -m "feat: add + New Track wizard for guided YAML Track Context creation"
```
