# Heterogeneous two-arm task allocation (Isaac Lab)

MSc dissertation prototype. A Franka Panda and a UR10e (Robotiq 2F-85) on a table.
A language model (Qwen, served locally) decides which arm handles which object; the
execution layer carries the plan out in Isaac Lab. The plan can include a RELAY: one
arm sets an object down at a shared point, the other picks it up and carries it on.

The research angle: object fragility is shown only VISUALLY and never written into the
symbolic state the planner sees, so we can measure what visual grounding adds to the
allocation decision.

---

## Folder layout (flat code, no packaging needed)

    scene.py            run this; the simulator side (imports Isaac Lab)
    state.py            pure Python: the fragility-free symbolic state
    planner.py          pure Python: the LLM planner (steps + dependencies)
    baselines.py        parked rule-based allocator (not wired in)
    README.md           this file
    docs/
      ARCHITECTURE.md            the layered design + roadmap
      LITERATURE_POSITIONING.md  how the state design relates to prior work

Keep `scene.py`, `state.py`, `planner.py`, and `baselines.py` in the SAME folder so
they import each other by plain name. Do not place a file named `types.py` (or other
standard-library names) beside them: it shadows the standard library and breaks imports.

---

## How to test, from quickest to fullest

### 1. Test the pure-Python parts (no simulator, no model)
From the project folder:

    python state.py        # prints the allocator's view + the cue-leak gate firing
    python planner.py       # shows the prompt, accepts a valid relay, rejects bad plans

These two need nothing installed except `requests` (only used when a real model is
called, not in these examples).

### 2. Run the end-to-end relay in simulation (no model yet)
This is the main thing built so far: a hand-written relay plan executed in Isaac Lab.
Run it the way you run any Isaac Lab script in your environment, e.g.

    python scene.py

(or `./isaaclab.sh -p scene.py`, whichever your setup uses). It will:
  - build the scene and settle both arms,
  - build the symbolic state (to get the object label map),
  - validate a two-step relay plan, then
  - execute it: Franka places object_1 at the relay point, then UR carries it to
    zone_b, then the viewer holds so you can see the result.

Expect possibly one small tuning nudge to the relay point or zone_b coordinates if an
arm strains to reach (the on-paper reach check passes, so any change is minor).

### 3. The full instruction -> plan -> execution loop (needs Qwen)
Not wired yet end to end, and needs your local Qwen server running. When you start it,
point the planner at it with environment variables if the defaults do not match:

    export QWEN_ENDPOINT="http://localhost:8000/v1/chat/completions"
    export QWEN_MODEL="Qwen2-VL-7B-Instruct"

Then `make_plan(state)` in planner.py will return a real model-made plan that
`run_plan` can execute. (Wiring this into scene.py is the next step.)

---

## Status

- Execution layer (two arms reach/grasp/lift/place, kinematic grasp): working.
- Relay via a shared location (plan + validation + execution): built, ready to run.
- Symbolic state with cue-leak gate, and the LLM planner: built, tested offline.
- Next: run the relay in sim; then connect a live Qwen for typed instructions.
