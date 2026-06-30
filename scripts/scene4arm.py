"""Four-arm tabletop setup in Isaac Lab (one arm per table edge).

This is the four-arm extension of scene.py. Four arms stand one per edge of the
table, all facing inward:

    UR10e (west)  at (-1.15, 0),  facing +x
    UR10e (east)  at ( 1.15, 0),  facing -x
    Franka (south) at (0, -0.60), facing +y
    Franka (north) at (0,  0.60), facing -y

Why this assignment: the table is 2.8 m long in x. The Franka's reach (0.855 m)
cannot cross that long dimension, so the Frankas take the SHORT (y) edges, closer to
the centre, and the longer-reach UR10e's (1.30 m) take the LONG (x) edges. With this,
all four arms can reach a central point, which is what makes shared work possible.

THIS FILE'S SCOPE (first four-arm step): get all four arms to SPAWN and SETTLE
cleanly so you can see them stand on the table. Task logic (plans, relays across four
arms) is the next step and is intentionally not wired in here yet. The plan-running
machinery (run_plan, pick_and_place) is carried over unchanged for that next step.

Run:
    python scene_4arm.py            # windowed
    python scene_4arm.py --headless # no GUI

Environment note: runs on the Isaac Lab 2.3.0 launchable; the UR uses the Robotiq
2F-140 (UR10e_ROBOTIQ_GRIPPER_CFG) via the fallback import, same as scene.py.
"""

import argparse

from isaaclab.app import AppLauncher

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

# UR10e Robotiq config: prefer the 2F-85 name, fall back to the 2F-140 (see scene.py).
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

# --- Table geometry (same as the two-arm scene). ---
TABLE_H = 0.75
TABLE_TOP = (2.8, 1.6, 0.05)
TABLE_LEG = 0.10
_LEG_MARGIN = 0.15
_LEG_INSET_X = TABLE_TOP[0] / 2 - _LEG_MARGIN
_LEG_INSET_Y = TABLE_TOP[1] / 2 - _LEG_MARGIN
_LEG_H = TABLE_H - TABLE_TOP[2]
_LEG_Z = _LEG_H / 2

# --- Four arm bases, one per edge, all facing inward toward the centre.
#     Orientation is a yaw about z; quaternions are (w, x, y, z).
#       yaw   0  -> faces +x -> (1, 0, 0, 0)
#       yaw 180  -> faces -x -> (0, 0, 0, 1)
#       yaw +90  -> faces +y -> (0.70711, 0, 0, 0.70711)
#       yaw -90  -> faces -y -> (0.70711, 0, 0, -0.70711)
#     UR (long reach) on the long x-edges; Franka on the short y-edges. ---
_R2 = 0.70710678
UR_W_BASE,  UR_W_QUAT  = (-1.15, 0.0, TABLE_H), (1.0, 0.0, 0.0, 0.0)      # west, faces +x
UR_E_BASE,  UR_E_QUAT  = ( 1.15, 0.0, TABLE_H), (0.0, 0.0, 0.0, 1.0)      # east, faces -x
FR_S_BASE,  FR_S_QUAT  = (0.0, -0.60, TABLE_H), (_R2, 0.0, 0.0,  _R2)     # south, faces +y
FR_N_BASE,  FR_N_QUAT  = (0.0,  0.60, TABLE_H), (_R2, 0.0, 0.0, -_R2)     # north, faces -y

# --- Objects (kept where the two-arm scene had them; they do not interfere with
#     spawning the arms). ---
CUBE_XY = (-0.95, -0.40)
CUBE_SIZE = 0.05
CUBE_CENTER_Z = TABLE_H + CUBE_SIZE / 2
FRAG_XY = (0.0, 0.22)
FRAG_SIZE = 0.05
FRAG_CENTER_Z = TABLE_H + FRAG_SIZE / 2

OBJECT_MAX_FORCE = {"cube": 100.0, "fragile": 25.0}

# --- A central relay point all four arms can reach (for later task work). ---
LOCATIONS = {
    "relay": (0.0, 0.0, TABLE_H + CUBE_SIZE / 2),
}

# --- Motion parameters. ---
HOVER_Z = TABLE_H + 0.30
GRASP_QUAT_W = (0.0, 1.0, 0.0, 0.0)
REACH_TOL = 0.015
REACH_MAX_STEPS = 900
GRIP_STEPS = 120
LIFT_MIN_RISE = 0.05
SETTLE_STEPS = 90
VIEWER_HOLD_STEPS = 300


# ============================ PART 2: THE SCENE ==============================

def make_ur_cfg():
    """UR articulation config with Isaac Lab's reference arm gains (see scene.py)."""
    cfg = UR10e_ROBOTIQ_2F_85_CFG.replace(prim_path="{ENV_REGEX_NS}/UR")
    arm_gains = {
        "shoulder": (1320.0, 72.6636085),
        "elbow": (600.0, 34.64101615),
        "wrist": (216.0, 29.39387691),
    }
    for grp, (stiff, damp) in arm_gains.items():
        cfg.actuators[grp].stiffness = stiff
        cfg.actuators[grp].damping = damp
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


def _franka_cfg(prim, base, quat):
    """A Franka config at a given base position and facing (quaternion)."""
    return FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path=prim,
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(pos=base, rot=quat),
    )


def _ur_cfg(prim, base, quat):
    """A UR config at a given base position and facing (quaternion)."""
    return UR_CFG.replace(
        prim_path=prim,
        init_state=UR10e_ROBOTIQ_2F_85_CFG.init_state.replace(pos=base, rot=quat),
    )


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Ground, lighting, table, FOUR arms (one per edge), and the two objects."""

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

    leg_pp = _leg("/World/Table/LegPP", (_LEG_INSET_X, _LEG_INSET_Y, _LEG_Z))
    leg_pn = _leg("/World/Table/LegPN", (_LEG_INSET_X, -_LEG_INSET_Y, _LEG_Z))
    leg_np = _leg("/World/Table/LegNP", (-_LEG_INSET_X, _LEG_INSET_Y, _LEG_Z))
    leg_nn = _leg("/World/Table/LegNN", (-_LEG_INSET_X, -_LEG_INSET_Y, _LEG_Z))

    # Four arms, one per edge. The attribute name is the scene key the controller uses.
    ur_w = _ur_cfg("{ENV_REGEX_NS}/URW", UR_W_BASE, UR_W_QUAT)
    ur_e = _ur_cfg("{ENV_REGEX_NS}/URE", UR_E_BASE, UR_E_QUAT)
    franka_s = _franka_cfg("{ENV_REGEX_NS}/FrankaS", FR_S_BASE, FR_S_QUAT)
    franka_n = _franka_cfg("{ENV_REGEX_NS}/FrankaN", FR_N_BASE, FR_N_QUAT)

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

    fragile = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Fragile",
        spawn=sim_utils.CuboidCfg(
            size=(FRAG_SIZE, FRAG_SIZE, FRAG_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=2.0, dynamic_friction=2.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.45, 0.85, 0.85), metallic=0.0, roughness=0.1,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(FRAG_XY[0], FRAG_XY[1], FRAG_CENTER_Z)),
    )


# ============================ PART 3: ARM CONTROL ============================
# Same shared controller as the two-arm scene, but the subclasses now take a NAME so
# more than one of each arm type can exist.

class ArmController:
    """Shared differential-IK control and kinematic-grasp logic for one arm."""

    GRIP_OPEN = None
    GRIP_CLOSE = None
    TOOL_OFFSET = None
    HOME_ARM_POSE = None

    def __init__(self, name, scene, sim, arm_joint_expr, ee_body_name):
        self.name = name
        self.scene = scene
        self.sim = sim
        self.robot = scene[name]
        self.device = sim.device
        self.sim_dt = sim.get_physics_dt()

        cfg = SceneEntityCfg(name, joint_names=arm_joint_expr, body_names=[ee_body_name])
        cfg.resolve(scene)
        self.arm_ids = cfg.joint_ids
        self.ee_body = cfg.body_ids[0]
        self.ee_jac = self.ee_body - 1 if self.robot.is_fixed_base else self.ee_body

        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        )
        self.ik = DifferentialIKController(ik_cfg, num_envs=1, device=self.device)

        self._carried = None
        self._carry_off_pos = None
        self._carry_off_quat = None

        print(f"[DIAG] {name}: arm_ids={self.arm_ids} ee_body={self.ee_body} "
              f"jac={self.ee_jac} fixed_base={self.robot.is_fixed_base}")

    # --- Hooks used during settling (overridden by the UR) ---
    def prepare(self):
        """Called once before settling. Default: nothing (Franka needs no prep)."""
        pass

    def apply_rest_target(self):
        """Command the arm to hold its rest pose for one settle step."""
        self.robot.set_joint_position_target(self.robot.data.default_joint_pos)

    # --- Gripper ---
    def set_gripper(self, val):
        raise NotImplementedError

    # --- Kinematics helpers ---
    def _ee_pose_b(self):
        ee_w = self.robot.data.body_state_w[:, self.ee_body, 0:7]
        root_w = self.robot.data.root_state_w[:, 0:7]
        return subtract_frame_transforms(
            root_w[:, 0:3], root_w[:, 3:7], ee_w[:, 0:3], ee_w[:, 3:7]
        )

    def _jacobian_b(self):
        """EE Jacobian rotated from world frame into the base frame (handles any yaw)."""
        jac_w = self.robot.root_physx_view.get_jacobians()[:, self.ee_jac, :, self.arm_ids]
        root_rot = matrix_from_quat(quat_inv(self.robot.data.root_quat_w))
        jac_b = jac_w.clone()
        jac_b[:, 0:3, :] = torch.bmm(root_rot, jac_b[:, 0:3, :])
        jac_b[:, 3:6, :] = torch.bmm(root_rot, jac_b[:, 3:6, :])
        return jac_b

    # --- Single control step ---
    def reach_step(self, goal_pos_w, goal_quat_w, grip, render):
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
        err, reached = 99.0, False
        for s in range(REACH_MAX_STEPS):
            err = self.reach_step((xy[0], xy[1], z), GRASP_QUAT_W, grip, render=(s % 2 == 0))
            if err < REACH_TOL:
                reached = True
                break
        print(f"[{'OK' if reached else '..'}] {self.name} {label}: err={err:.4f}")

    def grip_hold(self, xy, z, grip, label):
        for _ in range(GRIP_STEPS):
            self.reach_step((xy[0], xy[1], z), GRASP_QUAT_W, grip, render=True)
        state = "open" if grip > self.GRIP_CLOSE else "closed"
        print(f"[OK] {self.name} {label} (gripper -> {state})")

    def go_home(self, steps=150, render=True):
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
        return obj_center_z + self.TOOL_OFFSET

    # --- Kinematic grasp ---
    def attach(self, obj):
        self._carried = obj
        hand = self.robot.data.body_state_w[:, self.ee_body, 0:7]
        self._carry_off_pos = torch.tensor(
            [[0.0, 0.0, self.TOOL_OFFSET]], device=self.device, dtype=torch.float32
        )
        self._carry_off_quat = quat_inv(hand[:, 3:7])
        self._set_gravity(obj, disabled=True)

    def detach(self):
        if self._carried is not None:
            self._set_gravity(self._carried, disabled=False)
        self._carried = None

    def _update_carried(self):
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
        flags = obj.root_physx_view.get_disable_gravities()
        flags[:] = 1 if disabled else 0
        obj.root_physx_view.set_disable_gravities(flags, torch.arange(flags.shape[0]))


class FrankaController(ArmController):
    """Franka Panda with a parallel two-finger gripper. Takes a NAME so two can exist."""
    GRIP_OPEN = 0.04
    GRIP_CLOSE = 0.0
    TOOL_OFFSET = 0.107
    GRIP_FORCE = 15.0
    REACH = 0.855
    PAYLOAD = 3.0
    GRIPPER_TYPE = "parallel_jaw"

    def __init__(self, name, scene, sim):
        super().__init__(name, scene, sim, ["panda_joint.*"], "panda_hand")
        self.finger_ids = [
            next(i for i, n in enumerate(self.robot.joint_names) if n == "panda_finger_joint1"),
            next(i for i, n in enumerate(self.robot.joint_names) if n == "panda_finger_joint2"),
        ]
        print(f"[DIAG] {name} finger_ids={self.finger_ids}")

    def set_gripper(self, val):
        self.robot.set_joint_position_target(
            torch.full((1, 2), val, device=self.device), joint_ids=self.finger_ids
        )


class URController(ArmController):
    """UR10e with a single Robotiq driver joint. Takes a NAME so two can exist."""
    GRIP_OPEN = 0.0
    GRIP_CLOSE = 0.85
    TOOL_OFFSET = 0.18
    GRIP_FORCE = 20.0
    REACH = 1.30
    PAYLOAD = 12.5
    GRIPPER_TYPE = "adaptive_2f"
    READY_POSE = (0.0, -1.57, 1.57, -1.57, -1.57, 0.0)
    HOME_ARM_POSE = READY_POSE

    def __init__(self, name, scene, sim):
        super().__init__(
            name, scene, sim,
            ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            "wrist_3_link",
        )
        self.finger_ids = [
            next(i for i, n in enumerate(self.robot.joint_names) if n == "finger_joint")
        ]
        print(f"[DIAG] {name} finger_ids={self.finger_ids}")

    def prepare(self):
        """Seat the arm in its ready stance before settling."""
        self.set_ready_pose()

    def apply_rest_target(self):
        """Hold the ready-stance arm joints during settling."""
        self.robot.set_joint_position_target(
            self.robot.data.joint_pos[:, self.arm_ids], joint_ids=self.arm_ids
        )

    def set_ready_pose(self):
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

def build_world():
    """Create the sim, scene, camera, and the four arm controllers (as a dict)."""
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1 / 120, device=args_cli.device)
    )
    # Pulled back a little so all four arms are in view.
    sim.set_camera_view(eye=(3.2, -2.8, 2.5), target=(0.0, 0.0, TABLE_H))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=8.0))
    sim.reset()

    arms = {
        "ur_w": URController("ur_w", scene, sim),
        "ur_e": URController("ur_e", scene, sim),
        "franka_s": FrankaController("franka_s", scene, sim),
        "franka_n": FrankaController("franka_n", scene, sim),
    }
    return sim, scene, arms


def settle(sim, scene, arms):
    """Let all arms reach a stable rest pose before anything else.

    Each arm's prepare() runs once (the URs seat their ready stance), then every arm
    holds its rest target for SETTLE_STEPS. Works for any number of arms.
    """
    for a in arms.values():
        a.prepare()
    sim_dt = sim.get_physics_dt()
    for c in range(SETTLE_STEPS):
        for a in arms.values():
            a.apply_rest_target()
        scene.write_data_to_sim()
        sim.step(render=(c % 3 == 0))
        scene.update(sim_dt)


def fragility_check(arm, obj_key):
    grip = arm.GRIP_FORCE
    max_safe = OBJECT_MAX_FORCE[obj_key]
    safe = grip <= max_safe
    verdict = "SAFE" if safe else "BROKEN (grip exceeds object tolerance)"
    print(f"[FRAGILITY] {arm.name} on '{obj_key}': "
          f"grip={grip:.1f} max_safe={max_safe:.1f} -> {verdict}")
    return safe


def pick_and_place(arm, obj_key, pick_xy, obj_center_z, drop_xy, lift_z, label):
    """Full pick-and-place (carried over from scene.py for the next, task step)."""
    scene = arm.scene
    grasp_height = arm.grasp_z(obj_center_z)
    place_height = arm.grasp_z(obj_center_z)
    z_before = float(scene[obj_key].data.root_pos_w[0, 2])

    arm.move_to(pick_xy, HOVER_Z, arm.GRIP_OPEN, "pre_grasp")
    arm.move_to(pick_xy, grasp_height, arm.GRIP_OPEN, "descend")
    arm.grip_hold(pick_xy, grasp_height, arm.GRIP_CLOSE, "grasp")
    arm.attach(scene[obj_key])
    arm.move_to(pick_xy, lift_z, arm.GRIP_CLOSE, "lift")

    z_after = float(scene[obj_key].data.root_pos_w[0, 2])
    lifted = (z_after - z_before) > LIFT_MIN_RISE

    arm.move_to(drop_xy, lift_z, arm.GRIP_CLOSE, "transport")
    arm.move_to(drop_xy, place_height, arm.GRIP_CLOSE, "place_descend")
    arm.detach()
    arm.grip_hold(drop_xy, place_height, arm.GRIP_OPEN, "release")
    arm.move_to(drop_xy, lift_z, arm.GRIP_OPEN, "retreat")
    print(f"[{label}] lifted={lifted}")
    return lifted


def hold_viewer(sim, scene, arms):
    """Keep the window open, keeping any carried objects locked to their hands."""
    sim_dt = sim.get_physics_dt()
    for _ in range(VIEWER_HOLD_STEPS):
        for arm in arms:
            arm._update_carried()
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        for arm in arms:
            arm._update_carried()


# ============================ PART 5: MAIN ==================================
# First four-arm step: spawn the arms, settle them, and hold the viewer so you can
# see all four standing one per edge. Task logic comes next.

def main():
    sim, scene, arms = build_world()
    settle(sim, scene, arms)

    print("\n=== Four arms placed, one per edge ===")
    for name, a in arms.items():
        base = a.robot.data.root_pos_w[0]
        print(f"  {name:9s} {type(a).__name__:16s} base=({float(base[0]):+.2f}, "
              f"{float(base[1]):+.2f})  reach={a.REACH}")
    print("Watch the viewer: all four should stand on the table, facing inward.\n")

    hold_viewer(sim, scene, list(arms.values()))
    simulation_app.close()


if __name__ == "__main__":
    main()