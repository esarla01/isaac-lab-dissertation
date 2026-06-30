"""Two-arm tabletop manipulation in Isaac Lab (the execution layer).

A Franka Panda and a UR10e (with a Robotiq gripper) face each other across a table.
This file is the EXECUTION layer: given a plan it carries it out in simulation. The
headline behaviour is a RELAY: the Franka picks the cube up at a corner on its side,
sets it down at a shared centre point, and the UR then picks it up from there and
carries it to the opposite corner on its side.

Environment note: this runs on an Isaac Lab 2.3.0 launchable (isaaclab_assets 0.2.3),
which ships the UR10e with a Robotiq 2F-140 gripper (UR10e_ROBOTIQ_GRIPPER_CFG), not
the 2F-85 the project originally targeted. The import below prefers the 2F-85 name and
falls back to the 2F-140. Because grasping here is KINEMATIC, the exact gripper does
not affect whether a pick succeeds.

================================ HOW TO READ THIS FILE ========================
The file goes top to bottom from configuration, to one arm, to the whole demo:

  1. CONFIG (constants)        table, object, arm, and motion parameters; the named
                              LOCATIONS a plan can target (relay point + corners).
  2. SCENE (make_ur_cfg,       the Isaac Lab scene description: table, both arms,
     SceneCfg)                 the cube and the look-alike fragile cube.
  3. ARM CONTROL               ArmController (shared IK + grasp logic) and the two
     (ArmController, Franka-    per-arm subclasses that fill in gripper details and
     Controller, URController)  capability specs.
  4. RUNTIME HELPERS           build_world, settle, fragility_check, pick_and_place,
                              go_home, hold_viewer.
  5. run_plan + main           run_plan executes a plan step by step in dependency
                              order; main builds the state, makes a relay plan,
                              checks it, and runs it.

Two design choices explain most of the non-obvious code:

* IK runs through Isaac Lab's DifferentialIKController. The subtlety that caused real
  trouble: PhysX returns the Jacobian in the WORLD frame while the IK command is in
  the robot BASE frame. They must agree, so the Jacobian is rotated into the base
  frame before use (see ArmController._jacobian_b). This is a no-op for the Franka
  (base unrotated) but essential for the UR (base yawed 180 degrees to face it).

* Grasping is KINEMATIC, not contact-based. When an arm "grasps", the object is
  locked to the hand and follows it rigidly until release (see attach / detach /
  _update_carried). This is deliberate: the research contribution is task allocation
  and visual grounding, not grasp dynamics, so the execution layer is held constant
  and made reliable rather than tuned against contact physics. The experiment's
  fragility check is a modelled force-threshold decision computed separately, so
  kinematic grasping does not affect it.
==============================================================================

Run:
    python scene.py            # windowed
    python scene.py --headless # no GUI
"""

import argparse

from isaaclab.app import AppLauncher

# The Omniverse app must be launched before any isaaclab asset/sim modules import.
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    subtract_frame_transforms,
    combine_frame_transforms,
    matrix_from_quat,
    quat_inv,
)

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

# This Isaac Lab build (isaaclab_assets 0.2.3) ships the UR10e with a Robotiq 2F-140
# gripper as UR10e_ROBOTIQ_GRIPPER_CFG, not the 2F-85 the script was written against.
# The actuator groups this file tweaks (shoulder/elbow/wrist, gripper_drive,
# gripper_finger) and the finger_joint driver all match, so it is a drop-in alias.
# Prefer the 2F-85 name where it exists; fall back to the 2F-140 otherwise.
try:
    from isaaclab_assets import UR10e_ROBOTIQ_2F_85_CFG
except ImportError:
    try:
        from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_2F_85_CFG
    except ImportError:
        from isaaclab_assets.robots.universal_robots import (
            UR10e_ROBOTIQ_GRIPPER_CFG as UR10e_ROBOTIQ_2F_85_CFG,
        )


# ============================ PART 1: CONFIG ================================
# Table, object, arm, and motion parameters, plus the named LOCATIONS a plan can
# target. Changing the scene mostly means changing values here.

# --- Table geometry. The work surface sits at TABLE_H; legs derive from it, so
#     changing the top size or height automatically repositions the legs. ---
TABLE_H = 0.75                       # work-surface height (m)
TABLE_TOP = (2.8, 1.6, 0.05)         # top slab: length x, width y, thickness z
TABLE_LEG = 0.10                     # square leg cross-section side (m)
_LEG_MARGIN = 0.15                   # inset from table edge to leg centre
_LEG_INSET_X = TABLE_TOP[0] / 2 - _LEG_MARGIN
_LEG_INSET_Y = TABLE_TOP[1] / 2 - _LEG_MARGIN
_LEG_H = TABLE_H - TABLE_TOP[2]      # leg height fills the gap under the slab
_LEG_Z = _LEG_H / 2                  # leg centre height

# --- Robot bases. The Franka faces +x by default; the UR is yawed 180 degrees so
#     both grippers point inward toward the shared workspace. ---
FRANKA_BASE = (-0.55, 0.0, TABLE_H)
UR_BASE = (0.75, 0.0, TABLE_H)
UR_FACING_QUAT = (0.0, 0.0, 0.0, 1.0)  # (w,x,y,z): 180 deg about z

# --- Objects. The cube is the robust object; the fragile object is a same-size,
#     same-mass GLASS-LIKE cube whose appearance (not its shape or mass) is the only
#     fragility cue. Both sit flat and stable when placed. ---
CUBE_XY = (-0.95, -0.40)             # start corner, near the Franka
CUBE_SIZE = 0.05
CUBE_CENTER_Z = TABLE_H + CUBE_SIZE / 2

FRAG_XY = (0.0, 0.22)
FRAG_SIZE = 0.05                     # same footprint as the cube, so it sits stably
FRAG_CENTER_Z = TABLE_H + FRAG_SIZE / 2

# --- Fragility model (modelled decision, NOT physics). Each arm applies a grip
#     force; each object tolerates a maximum before it breaks. The check is a
#     logged ground-truth outcome, kept fully separate from the symbolic state the
#     allocator sees, so it does not leak the fragility cue into allocation. ---
OBJECT_MAX_FORCE = {
    "cube": 100.0,                   # robust: tolerates any grip here
    "fragile": 25.0,                 # fragile: a firm grip would break it
}

# --- Named locations a plan can target, in world coordinates. The relay point sits
#     in the central band BOTH arms can reach, so one arm can set an object down
#     there and the other can pick it up (a relay via a shared location). For the
#     corner-to-corner demo: the cube starts at a corner near the Franka, the Franka
#     relays it to the centre, and the UR carries it to the opposite corner near
#     itself. These names match what state.py / planner.py use. ---
LOCATIONS = {
    "relay":    (0.0, 0.0, TABLE_H + CUBE_SIZE / 2),     # shared: both arms reach it
    "corner_b": (1.15, 0.55, TABLE_H + CUBE_SIZE / 2),   # opposite corner, near the UR
}

# --- Motion parameters shared by both arms. ---
HOVER_Z = TABLE_H + 0.30             # hand height at the hover / pre-grasp pose
GRASP_QUAT_W = (0.0, 1.0, 0.0, 0.0)  # top-down grasp: hand z aligned with world -z
REACH_TOL = 0.015                    # arrival tolerance for a reach (m)
REACH_MAX_STEPS = 900                # cap on reach iterations (UR converges slower)
GRIP_STEPS = 120                     # steps held while the gripper opens/closes
LIFT_MIN_RISE = 0.05                 # min object rise to count the grasp as successful
SETTLE_STEPS = 90                    # steps to let the arms settle before the demo
VIEWER_HOLD_STEPS = 300              # steps to keep the window open at the end


# ============================ PART 2: THE SCENE ==============================
# The Isaac Lab description of the world: table, both arms, and the two objects
# (the cube and the look-alike fragile cube).

def make_ur_cfg():
    """Build the UR articulation config, matching Isaac Lab's reference arm gains.

    The arm gains below are the official per-joint-group values from Isaac Lab's
    UR10e configuration (shoulder/elbow/wrist), so this rig uses the validated
    reference tuning rather than hand-picked numbers. The Robotiq drive gain is
    raised above the reference default only because that default stalls partway when
    the fingers meet an object; grasping here is kinematic, so this is harmless.
    """
    cfg = UR10e_ROBOTIQ_2F_85_CFG.replace(prim_path="{ENV_REGEX_NS}/UR")
    # Reference per-group arm gains (stiffness, damping) from the official UR10e cfg.
    arm_gains = {
        "shoulder": (1320.0, 72.6636085),
        "elbow": (600.0, 34.64101615),
        "wrist": (216.0, 29.39387691),
    }
    for grp, (stiff, damp) in arm_gains.items():
        cfg.actuators[grp].stiffness = stiff
        cfg.actuators[grp].damping = damp
    # Robotiq drive: raised from the reference default (which stalls under contact).
    for grp in ("gripper_drive", "gripper_finger"):
        if grp in cfg.actuators:
            cfg.actuators[grp].stiffness = 80.0
            cfg.actuators[grp].damping = 10.0
    return cfg


UR_CFG = make_ur_cfg()


def _leg(path, pos):
    """A single square table leg at the given world position."""
    return AssetBaseCfg(
        prim_path=path,
        spawn=sim_utils.CuboidCfg(
            size=(TABLE_LEG, TABLE_LEG, _LEG_H),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.22, 0.15)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Ground, lighting, table, both arms, and both objects.

    Note: every asset is declared as its own class attribute. Helper variables in
    the class body would be misread as assets, so the legs are built via the _leg
    helper at module scope and assigned here.
    """

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)),
    )

    table_top = AssetBaseCfg(
        prim_path="/World/Table/Top",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_TOP,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.46, 0.30)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_H - TABLE_TOP[2] / 2)),
    )

    # Leg names encode the corner sign: pp=(+x,+y), pn=(+x,-y), np=(-x,+y), nn=(-x,-y).
    leg_pp = _leg("/World/Table/LegPP", (_LEG_INSET_X, _LEG_INSET_Y, _LEG_Z))
    leg_pn = _leg("/World/Table/LegPN", (_LEG_INSET_X, -_LEG_INSET_Y, _LEG_Z))
    leg_np = _leg("/World/Table/LegNP", (-_LEG_INSET_X, _LEG_INSET_Y, _LEG_Z))
    leg_nn = _leg("/World/Table/LegNN", (-_LEG_INSET_X, -_LEG_INSET_Y, _LEG_Z))

    franka = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Franka",
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(pos=FRANKA_BASE),
    )
    ur = UR_CFG.replace(
        init_state=UR10e_ROBOTIQ_2F_85_CFG.init_state.replace(
            pos=UR_BASE, rot=UR_FACING_QUAT
        ),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.5, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(CUBE_XY[0], CUBE_XY[1], CUBE_CENTER_Z)),
    )

    # Fragile object: a glass-like translucent cube. The breakable APPEARANCE is the
    # visual fragility cue (not the shape), so it stays visually distinct from the
    # solid robust cube while sitting flat and stable when placed. The fragility is
    # never encoded symbolically; only its look signals it.
    fragile = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Fragile",
        spawn=sim_utils.CuboidCfg(
            size=(FRAG_SIZE, FRAG_SIZE, FRAG_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),  # equal to the cube: mass must NOT correlate with fragility
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=2.0, dynamic_friction=2.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.45, 0.85, 0.85),  # icy cyan: reads as glass, fully visible
                metallic=0.0, roughness=0.1,       # smooth/shiny, glass-like finish
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(FRAG_XY[0], FRAG_XY[1], FRAG_CENTER_Z)),
    )


# ============================ PART 3: ARM CONTROL ============================
# ArmController holds the shared IK + grasp logic; the two subclasses below fill in
# each arm's gripper details and capability specs.

class ArmController:
    """Shared differential-IK control and kinematic-grasp logic for one arm.

    Subclasses provide only what differs between arms: the gripper joints and how
    to drive them, the arm joint names, the end-effector body, and the tool offset.
    Everything else (reaching a pose, holding the gripper, carrying an attached
    object) lives here so both arms behave identically through one code path.
    """

    # Subclasses must define these.
    GRIP_OPEN = None       # gripper command for "open"
    GRIP_CLOSE = None      # gripper command for "closed"
    TOOL_OFFSET = None     # distance from the hand/tool frame to the fingertips (m)
    HOME_ARM_POSE = None   # tucked stance between steps; None = default joint pose

    def __init__(self, name, scene, sim, arm_joint_expr, ee_body_name):
        self.name = name
        self.scene = scene
        self.sim = sim
        self.robot = scene[name]
        self.device = sim.device
        self.sim_dt = sim.get_physics_dt()

        # Resolve arm joint indices and the end-effector body from the scene.
        cfg = SceneEntityCfg(name, joint_names=arm_joint_expr, body_names=[ee_body_name])
        cfg.resolve(scene)
        self.arm_ids = cfg.joint_ids
        self.ee_body = cfg.body_ids[0]
        # For a fixed-base robot the root body is excluded from the Jacobian, so the
        # end-effector's Jacobian row is one less than its body index.
        self.ee_jac = self.ee_body - 1 if self.robot.is_fixed_base else self.ee_body

        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        )
        self.ik = DifferentialIKController(ik_cfg, num_envs=1, device=self.device)

        # Kinematic-grasp state: the object currently locked to the hand, plus its
        # fixed pose offset relative to the hand recorded at grasp time.
        self._carried = None
        self._carry_off_pos = None
        self._carry_off_quat = None

        print(f"[DIAG] {name}: arm_ids={self.arm_ids} ee_body={self.ee_body} "
              f"jac={self.ee_jac} fixed_base={self.robot.is_fixed_base}")

    # --- Gripper: each arm drives its own gripper joints differently. ---
    def set_gripper(self, val):
        raise NotImplementedError

    # --- Kinematics helpers ---
    def _ee_pose_b(self):
        """Current end-effector pose expressed in the robot base frame."""
        ee_w = self.robot.data.body_state_w[:, self.ee_body, 0:7]
        root_w = self.robot.data.root_state_w[:, 0:7]
        return subtract_frame_transforms(
            root_w[:, 0:3], root_w[:, 3:7], ee_w[:, 0:3], ee_w[:, 3:7]
        )

    def _jacobian_b(self):
        """End-effector Jacobian rotated from the world frame into the base frame.

        PhysX returns the Jacobian in world coordinates, but the IK command is given
        in the base frame, so the two must be expressed in the same frame. This is a
        no-op for an unrotated base (Franka) and essential for a yawed base (UR).
        """
        jac_w = self.robot.root_physx_view.get_jacobians()[:, self.ee_jac, :, self.arm_ids]
        root_rot = matrix_from_quat(quat_inv(self.robot.data.root_quat_w))
        jac_b = jac_w.clone()
        jac_b[:, 0:3, :] = torch.bmm(root_rot, jac_b[:, 0:3, :])  # linear block
        jac_b[:, 3:6, :] = torch.bmm(root_rot, jac_b[:, 3:6, :])  # angular block
        return jac_b

    # --- Single control step ---
    def reach_step(self, goal_pos_w, goal_quat_w, grip, render):
        """Advance one sim step toward a world-frame goal pose; return position error.

        Each step: convert the goal to the base frame, solve IK for joint targets,
        command the arm and gripper, step physics, then re-lock any carried object.
        """
        root_w = self.robot.data.root_state_w[:, 0:7]
        gp = torch.tensor([goal_pos_w], device=self.device, dtype=torch.float32)
        gq = torch.tensor([goal_quat_w], device=self.device, dtype=torch.float32)
        pos_b_cmd, quat_b_cmd = subtract_frame_transforms(root_w[:, 0:3], root_w[:, 3:7], gp, gq)
        self.ik.set_command(torch.cat([pos_b_cmd, quat_b_cmd], dim=-1))

        pos_b, quat_b = self._ee_pose_b()
        jac_b = self._jacobian_b()
        joint_des = self.ik.compute(pos_b, quat_b, jac_b, self.robot.data.joint_pos[:, self.arm_ids])
        self.robot.set_joint_position_target(joint_des, joint_ids=self.arm_ids)
        self.set_gripper(grip)

        self.scene.write_data_to_sim()
        self.sim.step(render=render)
        self.scene.update(self.sim_dt)
        self._update_carried()

        ee_w = self.robot.data.body_state_w[0, self.ee_body, 0:3]
        return torch.norm(ee_w - gp[0]).item()

    # --- Motion primitives ---
    def move_to(self, xy, z, grip, label):
        """Reach a hover/descend pose above (xy) at height z, holding the given grip."""
        err, reached = 99.0, False
        for s in range(REACH_MAX_STEPS):
            err = self.reach_step((xy[0], xy[1], z), GRASP_QUAT_W, grip, render=(s % 2 == 0))
            if err < REACH_TOL:
                reached = True
                break
        print(f"[{'OK' if reached else '..'}] {self.name} {label}: err={err:.4f}")

    def grip_hold(self, xy, z, grip, label):
        """Hold position for GRIP_STEPS while the gripper finishes opening/closing."""
        for _ in range(GRIP_STEPS):
            self.reach_step((xy[0], xy[1], z), GRASP_QUAT_W, grip, render=True)
        state = "open" if grip > self.GRIP_CLOSE else "closed"
        print(f"[OK] {self.name} {label} (gripper -> {state})")

    def go_home(self, steps=150, render=True):
        """Return to the tucked home stance, clearing the shared centre for the next arm.

        Called between plan steps so the arm that just acted is fully clear of the
        shared centre before the next arm approaches (prevents the arms clashing).
        """
        if self.HOME_ARM_POSE is None:
            home = self.robot.data.default_joint_pos[:, self.arm_ids]
        else:
            home = torch.tensor([self.HOME_ARM_POSE], device=self.device, dtype=torch.float32)
        for _ in range(steps):
            self.robot.set_joint_position_target(home, joint_ids=self.arm_ids)
            self.set_gripper(self.GRIP_OPEN)
            self.scene.write_data_to_sim()
            self.sim.step(render=render)
            self.scene.update(self.sim_dt)

    def grasp_z(self, obj_center_z):
        """Hand height at which the fingertips reach an object centred at obj_center_z."""
        return obj_center_z + self.TOOL_OFFSET

    # --- Kinematic grasp ---
    def attach(self, obj):
        """Lock obj to the hand, centred under the tool, and disable its gravity.

        The grasp is kinematic (the grip is not physical), so we lock the object
        directly under the tool centre regardless of where the hand actually landed.
        This makes the grasp look clean and the placement precise even when the arm's
        hand arrives a centimetre or two off the object (the UR, at stretched poses,
        tracks less tightly than the Franka). Gravity is disabled while carried.
        """
        self._carried = obj
        hand = self.robot.data.body_state_w[:, self.ee_body, 0:7]
        # Position the object directly below the tool at fingertip depth (hand local
        # z points down for a top-down grasp), instead of wherever it happened to sit.
        self._carry_off_pos = torch.tensor(
            [[0.0, 0.0, self.TOOL_OFFSET]], device=self.device, dtype=torch.float32
        )
        # off_quat = inverse(hand_quat) makes the object upright in the world.
        self._carry_off_quat = quat_inv(hand[:, 3:7])
        self._set_gravity(obj, disabled=True)

    def detach(self):
        """Release the carried object and restore its gravity so physics resumes."""
        if self._carried is not None:
            self._set_gravity(self._carried, disabled=False)
        self._carried = None

    def _update_carried(self):
        """Re-place the carried object at its recorded offset from the hand."""
        if self._carried is None:
            return
        hand = self.robot.data.body_state_w[:, self.ee_body, 0:7]
        new_pos, new_quat = combine_frame_transforms(
            hand[:, 0:3], hand[:, 3:7], self._carry_off_pos, self._carry_off_quat
        )
        self._carried.write_root_pose_to_sim(torch.cat([new_pos, new_quat], dim=-1))
        self._carried.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.device))

    @staticmethod
    def _set_gravity(obj, disabled):
        """Toggle gravity on a rigid object (1 = disabled, 0 = enabled)."""
        flags = obj.root_physx_view.get_disable_gravities()
        flags[:] = 1 if disabled else 0
        obj.root_physx_view.set_disable_gravities(flags, torch.arange(flags.shape[0]))


class FrankaController(ArmController):
    """Franka Panda with a parallel two-finger gripper."""
    GRIP_OPEN = 0.04          # finger separation when open (m)
    GRIP_CLOSE = 0.0          # finger separation when closed (m)
    TOOL_OFFSET = 0.107       # panda_hand origin -> fingertip (m)
    GRIP_FORCE = 15.0         # modelled grip force applied to a held object
    # --- Capability specs (legitimate symbolic dimensions for allocation) ---
    REACH = 0.855             # max reach radius (m), Franka Panda spec
    PAYLOAD = 3.0             # rated payload (kg)
    GRIPPER_TYPE = "parallel_jaw"

    def __init__(self, scene, sim):
        super().__init__("franka", scene, sim, ["panda_joint.*"], "panda_hand")
        # Both prismatic fingers are commanded together.
        self.finger_ids = [
            next(i for i, n in enumerate(self.robot.joint_names) if n == "panda_finger_joint1"),
            next(i for i, n in enumerate(self.robot.joint_names) if n == "panda_finger_joint2"),
        ]
        print(f"[DIAG] franka finger_ids={self.finger_ids}")

    def set_gripper(self, val):
        self.robot.set_joint_position_target(
            torch.full((1, 2), val, device=self.device), joint_ids=self.finger_ids
        )


class URController(ArmController):
    """UR10e with a single Robotiq driver joint (the linkage mimic joints follow it).

    Note: on this launchable the gripper is a Robotiq 2F-140 (see the import note at
    the top); the 2F-85 numbers below still work because the grasp is kinematic. If
    the UR's lift ever comes up short, raise TOOL_OFFSET.
    """
    GRIP_OPEN = 0.0           # driver joint angle when open (rad)
    GRIP_CLOSE = 0.85         # driver joint angle when closed (rad)
    TOOL_OFFSET = 0.18        # wrist_3_link -> Robotiq fingertip (m)
    GRIP_FORCE = 20.0         # modelled grip force (firmer industrial gripper)
    # --- Capability specs (legitimate symbolic dimensions for allocation) ---
    REACH = 1.30              # max reach radius (m), UR10e spec
    PAYLOAD = 12.5            # rated payload (kg), UR10e spec
    GRIPPER_TYPE = "adaptive_2f"
    # Bent-elbow "ready" stance over the table, so IK starts near a clean solution
    # instead of the folded default pose (which makes the arm swing wide).
    # Order: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3 (rad).
    READY_POSE = (0.0, -1.57, 1.57, -1.57, -1.57, 0.0)
    HOME_ARM_POSE = READY_POSE      # tuck back to the ready stance between steps

    def __init__(self, scene, sim):
        super().__init__(
            "ur", scene, sim,
            ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            "wrist_3_link",
        )
        # Only the driver joint is commanded; the Robotiq linkage mimics it.
        self.finger_ids = [
            next(i for i, n in enumerate(self.robot.joint_names) if n == "finger_joint")
        ]
        print(f"[DIAG] ur finger_ids={self.finger_ids}")

    def set_ready_pose(self):
        """Place the arm in READY_POSE as both live state and held target.

        Setting the actual joint state (not just the target) means the IK starts
        from a clean configuration on the first reach.
        """
        q = self.robot.data.joint_pos.clone()
        ready = torch.tensor(self.READY_POSE, device=self.device, dtype=q.dtype)
        for col, jid in enumerate(self.arm_ids):
            q[0, jid] = ready[col]
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.robot.set_joint_position_target(q[:, self.arm_ids], joint_ids=self.arm_ids)

    def set_gripper(self, val):
        self.robot.set_joint_position_target(
            torch.full((1, 1), val, device=self.device), joint_ids=self.finger_ids
        )


# ============================ PART 4: RUNTIME HELPERS ========================
# Building the world, settling it, the fragility check, and the motion primitives.

def build_world():
    """Create the sim context, scene, camera, and both arm controllers."""
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1 / 120, device=args_cli.device)
    )
    sim.set_camera_view(eye=(2.5, -2.1, 2.0), target=(0.0, 0.0, TABLE_H))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=8.0))
    sim.reset()

    franka = FrankaController(scene, sim)
    ur = URController(scene, sim)
    return sim, scene, franka, ur


def settle(sim, scene, franka, ur):
    """Let both arms reach a stable rest pose before the demo begins.

    The UR holds its ready stance; the Franka holds its default pose.
    """
    ur.set_ready_pose()
    sim_dt = sim.get_physics_dt()
    for c in range(SETTLE_STEPS):
        franka.robot.set_joint_position_target(franka.robot.data.default_joint_pos)
        ur.robot.set_joint_position_target(
            ur.robot.data.joint_pos[:, ur.arm_ids], joint_ids=ur.arm_ids
        )
        scene.write_data_to_sim()
        sim.step(render=(c % 3 == 0))
        scene.update(sim_dt)


def fragility_check(arm, obj_key):
    """Log whether the assigned arm's grip is safe for the object (modelled outcome).

    This is ground-truth logging of the allocation's consequence, computed from
    parameters only. It is intentionally independent of the grasp physics and of
    the symbolic state the allocator sees.
    """
    grip = arm.GRIP_FORCE
    max_safe = OBJECT_MAX_FORCE[obj_key]
    safe = grip <= max_safe
    verdict = "SAFE" if safe else "BROKEN (grip exceeds object tolerance)"
    print(f"[FRAGILITY] {arm.name} on '{obj_key}': "
          f"grip={grip:.1f} max_safe={max_safe:.1f} -> {verdict}")
    return safe


def pick_and_place(arm, obj_key, pick_xy, obj_center_z, drop_xy, lift_z, label):
    """Full pick-and-place: hover, descend, grasp, lift, transport, place, retreat.

    The object is locked to the hand on grasp (attach) and released over the drop
    location (detach restores gravity so it settles). Returns whether the pick
    achieved a real lift.
    """
    scene = arm.scene
    grasp_height = arm.grasp_z(obj_center_z)
    place_height = arm.grasp_z(obj_center_z)   # flat table, so same height to set down
    z_before = float(scene[obj_key].data.root_pos_w[0, 2])

    # --- Pick ---
    arm.move_to(pick_xy, HOVER_Z, arm.GRIP_OPEN, "pre_grasp")
    arm.move_to(pick_xy, grasp_height, arm.GRIP_OPEN, "descend")
    arm.grip_hold(pick_xy, grasp_height, arm.GRIP_CLOSE, "grasp")
    arm.attach(scene[obj_key])              # kinematic attach: lock object to the hand
    arm.move_to(pick_xy, lift_z, arm.GRIP_CLOSE, "lift")

    z_after = float(scene[obj_key].data.root_pos_w[0, 2])
    hand_z = float(arm.robot.data.body_state_w[0, arm.ee_body, 2])
    lifted = (z_after - z_before) > LIFT_MIN_RISE

    # --- Place ---
    arm.move_to(drop_xy, lift_z, arm.GRIP_CLOSE, "transport")
    arm.move_to(drop_xy, place_height, arm.GRIP_CLOSE, "place_descend")
    arm.detach()                            # release: restore gravity, object settles
    arm.grip_hold(drop_xy, place_height, arm.GRIP_OPEN, "release")
    arm.move_to(drop_xy, lift_z, arm.GRIP_OPEN, "retreat")

    print(f"\n========== {label} RESULT ==========")
    print(f"grasp hand target z: {grasp_height:.3f}   hand z at lift: {hand_z:.3f}")
    print(f"object z before: {z_before:.3f}   object z after lift: {z_after:.3f}")
    print(f">>> {'LIFTED' if lifted else 'NOT lifted (tune TOOL_OFFSET or GRIP_CLOSE)'}")
    print("===================================\n")
    return lifted


def hold_viewer(sim, scene, arms):
    """Keep the window open at the end, keeping carried objects locked to their hands."""
    sim_dt = sim.get_physics_dt()
    for _ in range(VIEWER_HOLD_STEPS):
        for arm in arms:
            arm._update_carried()
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        for arm in arms:
            arm._update_carried()


# ============================ PART 5: RUN A PLAN =============================
# Execute a plan step by step in dependency order, then the demo entry point.

def run_plan(plan, scene, sim, arms_by_name, label_to_key):
    """Carry out a plan in simulation, step by step, in dependency order.

    For each step: look up which arm and which real object it means, read where that
    object currently is (so a relay's second leg picks up wherever the first leg left
    it), and run a pick-and-place to the step's destination location. Steps run one at
    a time, so the arms never collide and a relay simply runs leg one then leg two.
    """
    from planner import ordered_steps          # order steps so dependencies come first

    def settle_scene(steps=60):
        """Let the scene come to rest so object positions are read accurately."""
        for _ in range(steps):
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(sim.get_physics_dt())

    for step in ordered_steps(plan):
        arm = arms_by_name[step.arm]
        obj_key = label_to_key[step.object_name]

        # Let the object come fully to rest (it may still be settling from the
        # previous step's release), then read its TRUE resting position. Reading a
        # still-settling object gives a stale target and the arm grabs off-centre.
        settle_scene()
        here = scene[obj_key].data.root_pos_w[0]
        pick_xy = (float(here[0]), float(here[1]))
        obj_center_z = float(here[2])

        # Where should it go? The named location's x, y on the table.
        dest = LOCATIONS[step.place_at]
        drop_xy = (dest[0], dest[1])

        print(f"\n[STEP {step.id}] {step.arm} moves {step.object_name} "
              f"({obj_key}) -> {step.place_at}   reason: {step.reason}")
        pick_and_place(arm, obj_key, pick_xy, obj_center_z, drop_xy, HOVER_Z,
                       f"{step.id} {step.arm}->{step.place_at}")

        # Tuck this arm back to its own side before the next step runs.
        arm.go_home()


def main():
    sim, scene, franka, ur = build_world()
    settle(sim, scene, franka, ur)
    arms_by_name = {"franka": franka, "ur": ur}

    # Build the symbolic state so we get the private label->object map. The planner
    # works in neutral labels (object_1, ...); execution uses this map to act on the
    # real scene objects.
    import state as state_mod
    object_specs = {
        "cube":    {"mass": 0.05, "footprint": 0.05},
        "fragile": {"mass": 0.05, "footprint": 0.05},
    }
    world, label_to_key = state_mod.build_symbolic_state(
        scene, (franka, ur), object_specs,
        goal="Move object_1 from its corner to the opposite corner (corner_b).",
        locations=LOCATIONS,
    )
    print("object_1 is really:", label_to_key.get("object_1"))

    # A hand-written RELAY plan (the shape the planner will later produce):
    #   s1: franka picks object_1 at its corner and sets it at the shared relay point
    #   s2: ur (after s1) picks it up from the relay point and carries it to corner_b
    from planner import Plan, Step, check_plan
    relay_plan = Plan(
        steps=[
            Step(id="s1", arm="franka", object_name="object_1", place_at="relay",
                 depends_on=[], reason="franka can reach its corner and the relay point"),
            Step(id="s2", arm="ur", object_name="object_1", place_at="corner_b",
                 depends_on=["s1"], reason="ur carries it from the relay point to the far corner"),
        ],
        made_by="hand_written_demo",
    )
    check_plan(relay_plan, world)        # validate before running (reach, deps, no loops)

    run_plan(relay_plan, scene, sim, arms_by_name, label_to_key)

    hold_viewer(sim, scene, (franka, ur))
    simulation_app.close()


if __name__ == "__main__":
    main()