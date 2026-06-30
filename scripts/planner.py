"""Ask a language model for a PLAN: an ordered set of steps that move objects.

A plan is a list of STEPS. Each step is one arm picking one object and placing it at
a named location. A step can depend on earlier steps ("do not start me until those
are done"), which is what lets a plan express a relay: arm A places an object at a
shared point, then arm B (depending on A) carries it onward.

This file reads the fragility-free scene from state.py, asks the Qwen model for
a plan, checks the plan makes sense, and hands it back. Execution runs the steps in
dependency order.

WHAT'S IN THIS FILE (in the order things happen):
  Step, Plan               - the result: ordered steps, with dependencies
  write_prompt(state)      - write the message we send the model
  ask_model(messages)      - send it to the Qwen endpoint and get the reply back
  read_plan_from_reply(..) - turn the reply into a checked Plan (or fail loudly)
  check_plan(plan, state)  - the safety checks (real names, no loops, reachable)
  make_plan(state)         - do all of the above; this is what you call

Note on the experiment: this is the text-only way of planning (the floor: no image,
no fragility). The image-based way will be added later and will reuse ask_model.
Baselines live in baselines.py.
"""

import os
import re
import json
import math
from dataclasses import dataclass, field
from typing import List, Optional

from state import state_to_code          # turns the scene into text for the model

try:
    import requests
except ImportError:
    requests = None


class PlannerError(Exception):
    """Raised when we could not produce a valid plan (model unreachable, or its
    reply made no sense). We raise instead of guessing, so a bad attempt is visible."""


# ----- Where the Qwen model lives. Set by environment variables. The endpoint can
# be a local server (e.g. vLLM on localhost) OR a hosted OpenAI-compatible API
# (e.g. Alibaba DashScope). The default below is a local server; override all three. -----
QWEN_ENDPOINT = os.environ.get("QWEN_ENDPOINT", "http://localhost:8000/v1/chat/completions")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen2-VL-7B-Instruct")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "EMPTY")     # local servers usually ignore this
QWEN_TIMEOUT_S = float(os.environ.get("QWEN_TIMEOUT_S", "60"))

# How close an arm's base must be (in the table plane) to reach a point.
# Used only to sanity-check the model's plan, not to make it.


# ----- The result: an ordered plan of steps -----

@dataclass
class Step:
    """One arm picks one object (wherever it currently is) and places it at a location.

    A step may list other step ids it depends on; it must not start until those are
    done. This is what orders a relay correctly.
    """
    id: str                       # e.g. "s1", so other steps can depend on it
    arm: str                      # which arm performs this step
    object_name: str              # which object, by its neutral label
    place_at: str                 # the location name to put the object down at
    depends_on: List[str] = field(default_factory=list)
    reason: str = ""              # the model's short justification


@dataclass
class Plan:
    """The whole plan: the ordered steps, plus where it came from."""
    steps: List[Step]
    made_by: str                          # how this plan was made, e.g. "text_llm"
    model_reply: Optional[str] = None     # the model's raw reply, kept for analysis


def write_prompt(state, image_b64: Optional[str] = None) -> List[dict]:
    """Write the message we send the model: describe the scene, ask for a plan.

    The scene (from state_to_code) lists arms, objects, and named locations, with no
    fragility. We explain that a plan is steps that may depend on each other, and that
    an object too far for one arm can be relayed via a shared location.

    If image_b64 is given (a base64-encoded RGB frame), it is attached to the user
    message as an image content block, and a sentence is added telling the model to
    use what it SEES about the objects. This is the ONLY difference between the
    text-only floor condition (image_b64=None) and the VLM condition: same model, same
    text, with or without the image. So modality, not model, is the variable.
    """
    scene = state_to_code(state)
    objects = [o.label for o in state.objects]
    arms = [a.name for a in state.arms]
    places = [loc.name for loc in state.locations]

    instructions = (
        "You plan pick-and-place steps for a team of robot arms. "
        "Each step is one arm picking one object (wherever it currently is) and "
        "placing it at one named location. An arm can only reach places within its "
        "reach of its base. If no single arm can move an object all the way to its "
        "destination, use a shared location as a relay: one arm places the object "
        "there, and a second step (depending on the first) has another arm carry it "
        "onward. Reply with JSON only, no extra text."
    )
    question_text = (
        f"{scene}\n\n"
        f"Objects: {objects}. Arms: {arms}. Locations: {places}.\n"
    )
    if image_b64 is not None:
        # Explain how the picture relates to the labels, so the model can ground the
        # neutral labels (object_1, ...) to what it sees by POSITION. The objects are
        # labelled by position in the state, so position is the bridge between the
        # text labels and the image.
        question_text += (
            "An overhead photo of the table is attached. The objects in the state are "
            "labelled by their position; match each labelled object to what you see at "
            "that position in the photo, and take its visible appearance into account "
            "when you decide which arm should handle it.\n"
        )
    question_text += (
        "Produce a plan as a list of steps. Give each step an id like \"s1\". "
        "Use depends_on to list step ids that must finish first (empty if none).\n"
        "Reply in this exact JSON shape:\n"
        '{"plan": [{"id": "s1", "arm": "<arm>", "object": "<object>", '
        '"place_at": "<location>", "depends_on": [], "reason": "<short reason>"}]}'
    )

    if image_b64 is None:
        # Text-only floor: the content is a plain string.
        user_content = question_text
    else:
        # VLM condition: the content is a list of blocks (image + text), the shape the
        # OpenAI-compatible vision API expects.
        user_content = [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": question_text},
        ]

    return [{"role": "system", "content": instructions},
            {"role": "user", "content": user_content}]


def ask_model(messages, max_tokens: int = 700) -> str:
    """Send the messages to the Qwen endpoint and return its text reply.

    The endpoint is OpenAI-compatible, so this works against a local vLLM server
    or a hosted API like DashScope; only the environment variables differ.
    """
    if requests is None:
        raise PlannerError("the 'requests' package is needed to talk to the model")
    body = {"model": QWEN_MODEL, "messages": messages,
            "max_tokens": max_tokens, "temperature": 0.0}
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {QWEN_API_KEY}"}
    reply = requests.post(QWEN_ENDPOINT, json=body, headers=headers, timeout=QWEN_TIMEOUT_S)
    reply.raise_for_status()
    return reply.json()["choices"][0]["message"]["content"]


def read_plan_from_reply(reply: str, state) -> Plan:
    """Turn the model's JSON reply into a Plan, then check it (raises if it is bad)."""
    # Clean any ``` code fences, then parse the JSON.
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", reply.strip()).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise PlannerError(f"the reply was not valid JSON: {e}")

    steps = []
    for item in data.get("plan", []):
        steps.append(Step(
            id=str(item.get("id", "")),
            arm=item.get("arm"),
            object_name=item.get("object"),
            place_at=item.get("place_at"),
            depends_on=list(item.get("depends_on", [])),
            reason=str(item.get("reason", "")),
        ))
    plan = Plan(steps=steps, made_by="text_llm", model_reply=reply)
    check_plan(plan, state)              # raises PlannerError if anything is wrong
    return plan


def _planar_distance(a, b) -> float:
    """Distance between two points ignoring height (tabletop reach is 2D)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def check_plan(plan: Plan, state) -> None:
    """Check the plan is well-formed and physically possible. Raises PlannerError.

    This is where a confused model gets caught. The checks, in order:
    """
    arms = {a.name: a for a in state.arms}
    objects = {o.label: o for o in state.objects}
    places = {loc.name: loc.position for loc in state.locations}
    step_ids = [s.id for s in plan.steps]

    # Step 1: every step must name a real arm, object, location, and have a unique id.
    seen_ids = set()
    for s in plan.steps:
        if s.id in seen_ids:
            raise PlannerError(f"two steps share the id {s.id!r}")
        seen_ids.add(s.id)
        if s.arm not in arms:
            raise PlannerError(f"step {s.id} uses an arm that does not exist: {s.arm!r}")
        if s.object_name not in objects:
            raise PlannerError(f"step {s.id} uses an object that does not exist: {s.object_name!r}")
        if s.place_at not in places:
            raise PlannerError(f"step {s.id} uses a location that does not exist: {s.place_at!r}")

    # Step 2: every dependency must point to a real step (and not to itself).
    for s in plan.steps:
        for dep in s.depends_on:
            if dep == s.id:
                raise PlannerError(f"step {s.id} depends on itself")
            if dep not in seen_ids:
                raise PlannerError(f"step {s.id} depends on a step that does not exist: {dep!r}")

    # Step 3: the dependencies must not form a loop. We order the steps so that every
    # step comes after the steps it depends on (a topological sort); if that is
    # impossible, there is a cycle.
    order = _order_by_dependencies(plan.steps)   # raises PlannerError on a cycle

    # Step 4: walk the plan in that order, tracking where each object is, and check
    # the assigned arm can reach both where it picks the object up and where it puts
    # it down. Each object starts at its scene position.
    object_pos = {label: obj.position for label, obj in objects.items()}
    by_id = {s.id: s for s in plan.steps}
    for sid in order:
        s = by_id[sid]
        arm = arms[s.arm]
        pick_pos = object_pos[s.object_name]      # wherever the object is right now
        drop_pos = places[s.place_at]
        if _planar_distance(arm.base_position, pick_pos) > arm.reach:
            raise PlannerError(
                f"step {s.id}: {s.arm} cannot reach object {s.object_name} "
                f"at {tuple(round(v,2) for v in pick_pos)}")
        if _planar_distance(arm.base_position, drop_pos) > arm.reach:
            raise PlannerError(
                f"step {s.id}: {s.arm} cannot reach location {s.place_at} "
                f"at {tuple(round(v,2) for v in drop_pos)}")
        object_pos[s.object_name] = drop_pos      # object is now at the drop location


def _order_by_dependencies(steps) -> List[str]:
    """Return step ids ordered so each comes after its dependencies. Raise on a cycle.

    Standard topological sort (Kahn's method): repeatedly take a step whose
    dependencies are all already placed. If none can be taken but steps remain, the
    dependencies form a loop.
    """
    remaining = {s.id: set(s.depends_on) for s in steps}
    ordered = []
    while remaining:
        ready = [sid for sid, deps in remaining.items() if deps <= set(ordered)]
        if not ready:
            raise PlannerError(f"the plan's dependencies form a loop among {list(remaining)}")
        for sid in ready:
            ordered.append(sid)
            del remaining[sid]
    return ordered


def make_plan(state, image_b64: Optional[str] = None) -> Plan:
    """Ask the model to plan, and return the checked Plan.

    This is the function the rest of the system calls. It fails loudly (PlannerError)
    if the model cannot be reached or its reply does not make sense, rather than
    guessing, so a bad attempt is always visible and can be retried.

    If image_b64 (a base64 RGB frame) is given, the model plans from text + image (the
    VLM condition); if not, it plans from text only (the floor). Same model either way.
    """
    messages = write_prompt(state, image_b64=image_b64)
    try:
        reply = ask_model(messages)
    except Exception as e:
        raise PlannerError(f"the model call failed: {e}") from e
    plan = read_plan_from_reply(reply, state)   # raises PlannerError if the plan is bad
    plan.made_by = "vlm" if image_b64 is not None else "text_llm"
    return plan


def ordered_steps(plan: Plan) -> List[Step]:
    """The plan's steps in an order that respects dependencies (for execution)."""
    by_id = {s.id: s for s in plan.steps}
    return [by_id[sid] for sid in _order_by_dependencies(plan.steps)]


# ----- Try it without Isaac Lab and without a running model -----
# `python planner.py` builds a small scene, shows the prompt, then feeds in pretend
# replies so you can watch the checking accept a valid relay and reject bad plans.
if __name__ == "__main__":
    from state import WorldState, ObjectState, ArmState, LocationState

    # object_1 sits on the franka side; zone_b is on the ur side, too far for franka.
    demo = WorldState(
        objects=[ObjectState("object_1", (-0.30, 0.0, 0.78), 0.05, 0.05)],
        arms=[ArmState("franka", (-0.55, 0.0, 0.75), 0.70, 3.0, "parallel_jaw", 15.0),
              ArmState("ur", (0.75, 0.0, 0.75), 0.70, 12.5, "adaptive_2f", 20.0)],
        locations=[LocationState("relay", (0.10, 0.0, 0.78)),
                   LocationState("zone_b", (0.55, 0.0, 0.78))],
        goal="Move object_1 to zone_b.",
    )

    print("=" * 68)
    print("THE MESSAGE WE WOULD SEND THE MODEL:")
    print("=" * 68)
    for m in write_prompt(demo):
        print(f"[{m['role']}]\n{m['content']}\n")

    print("=" * 68)
    print("A GOOD RELAY PLAN is accepted, and ordered for execution:")
    print("=" * 68)
    good = ('{"plan": ['
            '{"id": "s1", "arm": "franka", "object": "object_1", "place_at": "relay", '
            '"depends_on": [], "reason": "franka can reach object and relay"},'
            '{"id": "s2", "arm": "ur", "object": "object_1", "place_at": "zone_b", '
            '"depends_on": ["s1"], "reason": "ur carries it from relay to zone_b"}]}')
    plan = read_plan_from_reply(good, demo)
    for s in ordered_steps(plan):
        dep = f" after {s.depends_on}" if s.depends_on else ""
        print(f"  {s.id}: {s.arm} moves {s.object_name} -> {s.place_at}{dep}   ({s.reason})")

    print()
    print("=" * 68)
    print("BAD PLANS are rejected:")
    print("=" * 68)
    # (a) franka asked to reach zone_b directly, which is out of its reach
    one_shot = ('{"plan": [{"id": "s1", "arm": "franka", "object": "object_1", '
                '"place_at": "zone_b", "depends_on": []}]}')
    try:
        read_plan_from_reply(one_shot, demo)
    except PlannerError as e:
        print("  (a) out-of-reach plan rejected:", e)
    # (b) a dependency loop
    loop = ('{"plan": ['
            '{"id": "s1", "arm": "franka", "object": "object_1", "place_at": "relay", "depends_on": ["s2"]},'
            '{"id": "s2", "arm": "ur", "object": "object_1", "place_at": "zone_b", "depends_on": ["s1"]}]}')
    try:
        read_plan_from_reply(loop, demo)
    except PlannerError as e:
        print("  (b) looping plan rejected:", e)