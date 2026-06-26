# Project Architecture and Roadmap

VLM-based heterogeneous multi-robot task allocation in Isaac Lab 2.3.2.

This document maps the overall system, shows where the current code sits, and lays
out the next steps. It is written so it can double as a progress note for the
supervisor.

---

## 1. The one-sentence shape of the project

A centralised allocator decides which of several heterogeneous arms should handle
each task; a fixed, reliable execution layer carries those decisions out; and the
experiment measures *what visual grounding adds to the allocation decision* by
comparing allocators that see different amounts of information.

The architecture is the apparatus. The **contribution to knowledge** is the
controlled floor/ceiling/cost measurement of visual grounding, not the apparatus
itself.

---

## 2. Layered architecture

The system is a stack. Information flows down (a decision becomes motion); a small
amount of state flows up (the scene is read to inform the next decision).

```
+-------------------------------------------------------------+
|  EXPERIMENT / EVALUATION LAYER                              |   <-- contribution
|  - planner conditions: text-LLM (floor),                   |
|    informed-text-LLM (ceiling), end-to-end VLM,            |
|    two-stage describe-then-plan                            |
|  - baselines: random, rule-based, market-based            |
|  - logger: decisions, rationale, fragility outcomes,      |
|    floor/ceiling/cost decomposition                       |
+-------------------------------------------------------------+
                         |  allocation (arm -> task)
                         v
+-------------------------------------------------------------+
|  ALLOCATION LAYER                                          |   <-- contribution
|  - state extractor: builds the SYMBOLIC state the          |
|    allocator sees (deliberately omits fragility)          |
|  - perception: camera frame -> VLM (Qwen2-VL)             |
|  - plan(goal, state) interface: returns an allocation     |
+-------------------------------------------------------------+
                         |  allocation list (which arm, which object, where)
                         v
+-------------------------------------------------------------+
|  EXECUTION LAYER  =  scene.py  (DONE)                      |   <-- apparatus
|  - SceneCfg: table, two arms, objects, camera             |
|  - ArmController: shared differential-IK control loop     |
|      * world->base Jacobian conversion                    |
|      * kinematic attach (uniform across both arms)        |
|  - FrankaController / URController: per-arm grippers       |
|  - pick_and_place(): hover, grasp, lift, transport, place |
|  - fragility_check(): modelled, logged, decoupled         |
+-------------------------------------------------------------+
                         |  joint targets
                         v
+-------------------------------------------------------------+
|  SIMULATION  =  Isaac Lab 2.3.2 / PhysX                    |   <-- platform
+-------------------------------------------------------------+
```

---

## 3. Where `scene.py` sits

`scene.py` IS the execution layer, the bottom application layer above the simulator.
Its job is to take an allocation (which arm handles which object, and where to put
it) and carry it out reliably, identically for every arm, so that when the
experiment varies the allocator, *nothing about execution changes*. That uniformity
is what lets the experiment attribute outcome differences to the allocation alone.

What `scene.py` deliberately does NOT do:
- It does not decide allocation. That is the layer above.
- It does not read or encode object fragility into any state the allocator sees.
- It does not measure anything that is the contribution.

The single seam to the layer above is the `allocation` list in `main()`. Today it is
hardcoded. Later it is produced by the allocator. The format does not change, so the
execution layer never needs to change again.

---

## 4. Current status

DONE (execution layer):
- Two heterogeneous arms (Franka + UR10e/Robotiq) reach, grasp, lift, place.
- Shared differential-IK loop with the world->base Jacobian conversion.
- Kinematic attach, uniform across both arms (reliable, slip-free).
- Sequenced two-arm pick-and-place under a hardcoded allocation.
- Modelled fragility check: logged, parameter-based, decoupled from physics and
  from the symbolic state.
- Robot tuning aligned to the official Isaac Lab UR reference config (ready pose,
  body, IK method, tool offset, arm gains).

NOT STARTED (the contribution):
- State extractor, perception, plan() interface, planner conditions, baselines,
  logger, experiment harness.

---

## 5. Target file/module layout

The current single `scene.py` should grow into a small package as the upper layers
arrive. A suggested structure, keeping execution separate from research:

```
isaac_lab_dissertation/
  scripts/
    run_demo.py            # thin entry point (today's main())
  src/
    execution/
      scene_cfg.py         # SceneCfg (table, arms, objects, camera)
      controllers.py       # ArmController, FrankaController, URController
      pick_place.py        # pick_and_place, motion primitives
      fragility.py         # modelled fragility check + parameters
    allocation/
      state.py             # symbolic state extractor (NO fragility flag)
      perception.py        # camera capture -> Qwen2-VL
      planner.py           # plan(goal, state) + the four planner conditions
      baselines.py         # random, rule-based, market-based
    experiment/
      logger.py            # decisions, rationale, outcomes
      conditions.py        # floor / ceiling / cost decomposition
      run_experiment.py    # the harness that sweeps conditions
  ARCHITECTURE.md          # this file
```

`scene.py` as it stands maps onto `src/execution/*` plus `scripts/run_demo.py`. The
refactor into these modules can wait until the upper layers start, but the seams in
the current file (SceneCfg, controllers, pick_and_place, fragility_check, the
allocation list) already match these boundaries, so the split will be clean.

---

## 6. Next step

Build the **allocation layer's lower half**, which is the first real step into the
contribution, in this order:

1. **State extractor** (`allocation/state.py`): produce the symbolic state the
   allocator consumes from the scene (object positions, arm capabilities such as
   reach/payload/gripper-type). CRITICAL: it must NOT encode fragility. This is the
   cue-leak design gate that needs supervisor sign-off before scenario content is
   built on it.

2. **plan(goal, state) interface** (`allocation/planner.py`): a function that takes
   a goal and the symbolic state and returns an allocation list in exactly the
   format `scene.py` already consumes. Start with the simplest condition:
   the **text-LLM floor** (no visual grounding), via OpenRouter.

3. **Wire it in**: replace the hardcoded `allocation` list in the demo with the
   output of `plan()`, changing nothing in the execution layer.

This proves the full loop end to end (state -> allocator -> execution) with the
simplest allocator, after which the other conditions and the logger are additions
rather than new structure.

---

## 7. Open gates and risks (carry to supervisor)

- **Cue-leak gate (BLOCKING):** the symbolic state must not encode fragility, or the
  floor/ceiling measurement collapses. Needs sign-off before scenario content.
- **Appearance-based fragility cue:** the cue is now a glass-like object appearance,
  not a tippy shape. Confirm this is acceptable for the visual-grounding measure.
- **Time management:** execution (apparatus) is done; the remaining work is the
  contribution and the writing. Implementation should no longer defer writing.
