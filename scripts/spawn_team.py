"""Phase 1, milestone 1: spawn the four-robot team and save a render.

Run from your project (or the IsaacLab root):
    ./isaaclab.sh -p scripts/spawn_team.py --headless --enable_cameras

Goal: confirm all four robots load and render. Nothing moves yet. The arm and
the two wheeled bases are stable on spawn. The quadruped and the drone are not
passively stable and need controllers (added in a later milestone), so the
render is captured early, before they fall or sag.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spawn the four-robot transport team.")
parser.add_argument("--out", type=str, default="team_render.png", help="output image path")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the simulator first. Everything that touches Isaac Lab is imported below this line.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports that depend on Isaac Lab MUST come after the app has launched.
# ---------------------------------------------------------------------------
import copy

import numpy as np
import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

# Pre-defined robot configurations.
# If any of these fail to import, grep source/isaaclab_assets/isaaclab_assets/robots/
# in your IsaacLab clone for the exact constant name and fix the import.
from isaaclab_assets.robots.ridgeback_franka import RIDGEBACK_FRANKA_PANDA_CFG
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG
from isaaclab_assets import CRAZYFLIE_CFG


def placed(base_cfg, prim_path, pos):
    """Return a deep copy of a robot config at a given prim path and position.

    The deep copy avoids the common trap where two robots share one config
    object and only one of them ends up in the scene.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg.prim_path = prim_path
    cfg.init_state.pos = pos
    return cfg


@configclass
class TeamSceneCfg(InteractiveSceneCfg):
    """Four heterogeneous robots on a ground plane, plus one overhead camera."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)),
    )

    # R1: mobile manipulator (Ridgeback base + Franka arm). Stable on spawn.
    r1 = placed(RIDGEBACK_FRANKA_PANDA_CFG, "{ENV_REGEX_NS}/R1", (0.0, 0.0, 0.0))
    # R2: transporter placeholder (second Ridgeback + Franka, arm unused for now).
    r2 = placed(RIDGEBACK_FRANKA_PANDA_CFG, "{ENV_REGEX_NS}/R2", (2.5, 0.0, 0.0))
    # R3: quadruped (ANYmal). Spawned a little above ground so the feet clear it.
    r3 = placed(ANYMAL_C_CFG, "{ENV_REGEX_NS}/R3", (0.0, 2.5, 0.6))
    # R4: drone (Crazyflie). Starts in the air; needs a hover controller later.
    r4 = placed(CRAZYFLIE_CFG, "{ENV_REGEX_NS}/R4", (2.5, 2.5, 0.5))

    # Overhead-ish camera. Pose is aimed after reset, below.
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
    )


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = TeamSceneCfg(num_envs=1, env_spacing=8.0)
    scene = InteractiveScene(scene_cfg)

    # Build physics views before reading or writing any asset data.
    sim.reset()

    # Aim the camera at the middle of the team so all four robots are in frame.
    eyes = torch.tensor([[6.0, -6.0, 6.0]], device=sim.device)
    targets = torch.tensor([[1.25, 1.25, 0.0]], device=sim.device)
    scene["camera"].set_world_poses_from_view(eyes, targets)

    sim_dt = sim.get_physics_dt()
    capture_step = 30  # about half a second in, before the drone/quadruped fall
    count = 0
    saved = False

    while simulation_app.is_running() and count < 90:
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        count += 1

        if count == capture_step and not saved:
            rgb = scene["camera"].data.output["rgb"]  # (1, H, W, 4) uint8 for Camera
            img = rgb[0, :, :, :3].cpu().numpy().astype(np.uint8)
            Image.fromarray(img).save(args_cli.out)
            print(f"[OK] saved render to {args_cli.out}")
            saved = True

    if not saved:
        print("[WARN] simulation ended before the capture step; no image saved.")

    simulation_app.close()


if __name__ == "__main__":
    main()
