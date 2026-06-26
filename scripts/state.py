"""Symbolic state extractor for the allocation layer.

================================ HOW TO READ THIS FILE ========================
The whole job of this file is ONE thing: read the live simulation and produce a
clean description of the scene that the allocator is allowed to look at, with
fragility deliberately stripped out. It has four parts:

  1. THE CONTAINERS (ObjectState, ArmState, LocationState, WorldState)
     Small labelled records that define the SHAPE of the information the allocator
     receives. ObjectState has no fragility field, on purpose. LocationState names a
     place a plan can target (a relay point, a destination zone).

  2. THE GATE (_FORBIDDEN_TERMS, _assert_no_cue_leak)
     A tripwire. It checks that no fragility-related word ever appears in what the
     allocator sees, and stops the program loudly if one does. It does nothing in
     normal operation; it just catches mistakes before they silently ruin the
     experiment.

  3. THE BUILDER (build_symbolic_state)
     The recipe that fills the containers from the live scene. It reads object
     positions, labels objects NEUTRALLY by position (object_1, object_2, ...) so
     the name never reveals which is fragile, runs the tripwire on each, reads each
     arm's capabilities, and returns two things: the WorldState the allocator sees,
     plus a PRIVATE map (label -> real scene name) that only the execution layer
     uses. The allocator never sees that map.

  4. THE TRANSLATOR (state_to_code, state_to_text)
     Turns the WorldState into a prompt for a text-LLM allocator. state_to_code is
     the default (a structured, code-like dict block, which prior work found easier
     for models to parse); state_to_text is a readable form kept for a representation
     ablation. Because the WorldState is already fragility-free and neutrally
     labelled, both outputs are too.

One-sentence version: read the simulation, produce a fragility-free, neutrally
labelled description for the allocator, and keep a private side-table so only the
execution layer knows which anonymous object is really which.
==============================================================================

This module builds the SYMBOLIC state that the task allocator is allowed to see. It
is the most design-sensitive file in the project because of one rule:

    THE CUE-LEAK GATE
    -----------------
    Object fragility is a VISUAL property only. It must never appear, directly or by
    proxy, in the symbolic state produced here. If it leaks, the "floor" allocator
    (symbolic only) could match the "ceiling" allocator (told fragility) without any
    visual grounding, and the central measurement of the dissertation collapses.

Leaks are subtle. Three that this module actively guards against:
  1. A fragility FIELD (e.g. is_fragile, max_safe_force)            -> blocklisted.
  2. The object NAME ("fragile", "glass_cup")                       -> neutralised.
  3. A field that CORRELATES with fragility across scenarios, e.g.
     mass, size, or colour. Here the fragile object is a same-size,
     same-mass cube and colour is never serialised, so these carry
     no fragility information.

Objects are therefore exposed to the allocator under neutral labels (object_1,
object_2, ...). A private label -> scene-key map is returned for the EXECUTION layer
to act on; the allocator never sees it.

What IS allowed in the symbolic state (legitimate capability/task dimensions):
  - object positions (needed for reach feasibility)
  - object mass and footprint (capability matching; equal here so they cannot leak)
  - arm capabilities: base position, reach, payload, gripper type, grip force
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple


# Substrings that must never appear in an object field name or label.
_FORBIDDEN_TERMS = {
    "fragil", "breakab", "glass", "delicate", "max_safe", "tolerance", "brittle",
}


# ============================ PART 1: THE CONTAINERS ==========================
# Small labelled records defining the shape of what the allocator receives.

@dataclass(frozen=True)
class ObjectState:
    """Symbolic description of one object under a NEUTRAL label (no fragility)."""
    label: str                                   # neutral: object_1, object_2, ...
    position: Tuple[float, float, float]
    mass: float                                  # kg; equal across objects here
    footprint: float                             # m; equal across objects here


@dataclass(frozen=True)
class ArmState:
    """Symbolic description of one arm's fixed capabilities."""
    name: str
    base_position: Tuple[float, float, float]
    reach: float                                 # m
    payload: float                               # kg
    gripper_type: str
    grip_force: float


@dataclass(frozen=True)
class LocationState:
    """A named place a plan can target, e.g. a relay point or a destination zone.

    Locations are purely symbolic (a name and a position), so they are safe to show
    the allocator. The model refers to them by name; execution resolves the name to
    real coordinates.
    """
    name: str
    position: Tuple[float, float, float]


@dataclass(frozen=True)
class WorldState:
    """The full symbolic state handed to the allocator (neutral labels only)."""
    objects: List[ObjectState]
    arms: List[ArmState]
    locations: List[LocationState]
    goal: str


# ============================ PART 2: THE GATE ===============================
# A tripwire: stops the program if any fragility cue reaches the allocator's view.

def _assert_no_cue_leak(obj_state: ObjectState) -> None:
    """Fail loudly if any field name OR the label looks like a fragility cue."""
    field_names = " ".join(asdict(obj_state).keys()).lower()
    label = obj_state.label.lower()
    for term in _FORBIDDEN_TERMS:
        if term in field_names or term in label:
            raise ValueError(
                f"Cue-leak gate violation: '{term}' appears in object state "
                f"(label={obj_state.label!r}). Fragility must stay visual-only."
            )


def _assert_location_name_clean(name: str) -> None:
    """Fail loudly if a location name looks like a fragility cue (e.g. 'glass_zone')."""
    low = name.lower()
    for term in _FORBIDDEN_TERMS:
        if term in low:
            raise ValueError(
                f"Cue-leak gate violation: '{term}' appears in location name "
                f"{name!r}. Location names must not hint at fragility."
            )


# ============================ PART 3: THE BUILDER ============================
# Fills the containers from the live scene, runs the gate, keeps a private map.

def build_symbolic_state(scene, controllers, object_specs, goal, locations=None):
    """Read the live scene and return (WorldState, label_to_key).

    Args:
        scene: live InteractiveScene; objects are queried for current positions.
        controllers: iterable of arm controllers exposing capability attributes.
        object_specs: dict scene_key -> {"mass": float, "footprint": float}. Carries
            no fragility, only symbolic specs.
        goal: plain-language statement of what must be achieved.
        locations: optional dict name -> (x, y, z) of named places a plan may target
            (relay points, destination zones). Names must not hint at fragility.

    Returns:
        world_state: the symbolic state the allocator sees (neutral object labels).
        label_to_key: private map neutral_label -> scene_key, for the EXECUTION layer
            to translate an allocation back onto real scene objects. Never shown to
            the allocator.

    Objects are labelled in a fragility-independent order (sorted by position), so
    the label is a function only of position, which the allocator already sees, and
    therefore carries no extra information.
    """
    # Read positions first so we can order objects geometrically, not by their keys
    # (key order could otherwise encode fragility).
    keyed = []
    for key in object_specs:
        pos = tuple(float(v) for v in scene[key].data.root_pos_w[0])
        keyed.append((key, pos))
    keyed.sort(key=lambda kp: (round(kp[1][0], 3), round(kp[1][1], 3)))

    objects: List[ObjectState] = []
    label_to_key: Dict[str, str] = {}
    for i, (key, pos) in enumerate(keyed, start=1):
        label = f"object_{i}"
        spec = object_specs[key]
        obj = ObjectState(
            label=label,
            position=pos,
            mass=float(spec["mass"]),
            footprint=float(spec["footprint"]),
        )
        _assert_no_cue_leak(obj)          # enforce the gate at construction time
        objects.append(obj)
        label_to_key[label] = key

    arms: List[ArmState] = []
    for c in controllers:
        base = tuple(float(v) for v in c.robot.data.root_pos_w[0])
        arms.append(ArmState(
            name=c.name,
            base_position=base,
            reach=float(c.REACH),
            payload=float(c.PAYLOAD),
            gripper_type=str(c.GRIPPER_TYPE),
            grip_force=float(c.GRIP_FORCE),
        ))

    location_states: List[LocationState] = []
    for name, pos in (locations or {}).items():
        _assert_location_name_clean(name)   # a location name must not hint at fragility
        location_states.append(LocationState(name=name, position=tuple(float(v) for v in pos)))

    world = WorldState(objects=objects, arms=arms, locations=location_states, goal=goal)
    return world, label_to_key


# ============================ PART 4: THE TRANSLATOR =========================
# Turns the (already clean) state into a prompt for a text-LLM allocator. Two forms
# are provided. state_to_code is the DEFAULT, because prior work (SMART-LLM, Kannan
# et al., 2024) found that encoding robot skills and object properties as structured,
# code-like dictionaries improves an LLM's comprehension of the state versus loose
# prose. state_to_text is the readable form, kept so the two representations can be
# compared as an ablation. Both are fragility-free by construction.

def state_to_code(state: WorldState) -> str:
    """Serialise the symbolic state as a Python-style dict block (the default form).

    This mirrors the structured representation that prior LLM task-allocation work
    found easier for models to parse. Neutral labels only; no fragility, no colour,
    no image. It is the exact text the text-LLM (floor) condition receives.
    """
    lines = [f'goal = "{state.goal}"', "", "robots = {"]
    for a in state.arms:
        b = a.base_position
        lines.append(
            f'    "{a.name}": {{"base": ({b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}), '
            f'"reach": {a.reach:.2f}, "payload": {a.payload:.1f}, '
            f'"gripper": "{a.gripper_type}", "grip_force": {a.grip_force:.1f}}},'
        )
    lines.append("}")
    lines.append("")
    lines.append("objects = {")
    for o in state.objects:
        p = o.position
        lines.append(
            f'    "{o.label}": {{"position": ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}), '
            f'"mass": {o.mass:.2f}, "footprint": {o.footprint:.2f}}},'
        )
    lines.append("}")
    lines.append("")
    lines.append("locations = {")
    for loc in state.locations:
        p = loc.position
        lines.append(f'    "{loc.name}": ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}),')
    lines.append("}")
    return "\n".join(lines)


def state_to_text(state: WorldState) -> str:
    """Serialise the symbolic state as readable prose (alternative to state_to_code).

    Kept for a representation ablation (structured vs prose). Symbolic facts only,
    neutral labels, no fragility, no colour, no image.
    """
    lines = [f"GOAL: {state.goal}", "", "ARMS:"]
    for a in state.arms:
        lines.append(
            f"  - {a.name}: base=({a.base_position[0]:.2f}, {a.base_position[1]:.2f}, "
            f"{a.base_position[2]:.2f}), reach={a.reach:.2f} m, payload={a.payload:.1f} kg, "
            f"gripper={a.gripper_type}, grip_force={a.grip_force:.1f}"
        )
    lines.append("")
    lines.append("OBJECTS:")
    for o in state.objects:
        lines.append(
            f"  - {o.label}: position=({o.position[0]:.2f}, {o.position[1]:.2f}, "
            f"{o.position[2]:.2f}), mass={o.mass:.2f} kg, footprint={o.footprint:.2f} m"
        )
    lines.append("")
    lines.append("LOCATIONS:")
    for loc in state.locations:
        lines.append(
            f"  - {loc.name}: position=({loc.position[0]:.2f}, "
            f"{loc.position[1]:.2f}, {loc.position[2]:.2f})"
        )
    return "\n".join(lines)


# ============================ STANDALONE EXAMPLE =============================
# Run this file directly (`python state.py`) to see the extractor work WITHOUT
# Isaac Lab. It uses tiny fakes that mimic only the bits of the scene and the arm
# controllers that the extractor actually reads (object/arm positions and the arm
# capability attributes), so you can watch the inputs turn into the allocator's
# view, the private map, and the cue-leak tripwire firing.

if __name__ == "__main__":

    # --- Minimal fakes standing in for the live Isaac Lab objects ---
    class _FakeData:
        def __init__(self, pos):
            self.root_pos_w = [pos]            # extractor reads root_pos_w[0]

    class _FakeEntity:
        def __init__(self, pos):
            self.data = _FakeData(pos)

    class _FakeScene:
        def __init__(self, entities):
            self._e = entities
        def __getitem__(self, key):
            return self._e[key]

    class _FakeArm:
        def __init__(self, name, base, reach, payload, gripper_type, grip_force):
            self.name = name
            self.robot = _FakeEntity(base)     # extractor reads robot.data.root_pos_w
            self.REACH = reach
            self.PAYLOAD = payload
            self.GRIPPER_TYPE = gripper_type
            self.GRIP_FORCE = grip_force

    # A scene with two objects on opposite sides of the table.
    scene = _FakeScene({
        "cube":    _FakeEntity([0.0, -0.22, 0.775]),
        "fragile": _FakeEntity([0.0,  0.22, 0.775]),
    })
    # Two heterogeneous arms with real-ish capability specs.
    controllers = [
        _FakeArm("franka", [-0.55, 0.0, 0.75], 0.855, 3.0, "parallel_jaw", 15.0),
        _FakeArm("ur",     [ 0.75, 0.0, 0.75], 1.30, 12.5, "adaptive_2f", 20.0),
    ]
    # Symbolic specs only: note there is NO fragility here. Mass/footprint are equal
    # across objects so they cannot act as a fragility proxy.
    object_specs = {
        "cube":    {"mass": 0.05, "footprint": 0.05},
        "fragile": {"mass": 0.05, "footprint": 0.05},
    }
    goal = "Move both objects to their drop zones."
    # Named places a plan may target: a relay point in the middle (both arms reach it)
    # and a destination zone on each side.
    locations = {
        "relay":  (0.0, 0.0, 0.78),
        "zone_a": (-0.35, -0.22, 0.78),
        "zone_b": (0.35, 0.22, 0.78),
    }

    # --- Build the symbolic state and show what each consumer gets ---
    world, label_to_key = build_symbolic_state(
        scene, controllers, object_specs, goal, locations
    )

    print("=" * 70)
    print("1) WHAT THE ALLOCATOR SEES (default: structured code-like form):")
    print("=" * 70)
    print(state_to_code(world))

    print()
    print("-" * 70)
    print("   (alternative readable form, kept for a representation ablation):")
    print("-" * 70)
    print(state_to_text(world))

    print()
    print("=" * 70)
    print("2) PRIVATE MAP (execution-only; the allocator never sees this):")
    print("=" * 70)
    print("   ", label_to_key)
    print("   -> e.g. if the allocator assigns 'object_2', execution acts on",
          repr(label_to_key["object_2"]))

    # --- Demonstrate the cue-leak tripwire firing ---
    print()
    print("=" * 70)
    print("3) CUE-LEAK GATE DEMO (deliberately try to leak fragility):")
    print("=" * 70)
    try:
        bad = ObjectState(label="glass_cup", position=(0, 0, 0), mass=0.05, footprint=0.05)
        _assert_no_cue_leak(bad)
        print("   ERROR: the gate did NOT fire (this should not happen)")
    except ValueError as e:
        print("   Gate correctly fired:")
        print("   ", e)