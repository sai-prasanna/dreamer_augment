import ray
import subprocess
#ray.resource_spec._autodetect_num_gpus = lambda: len(list(filter(lambda c: c.wmi_property('AdapterCompatibility').value == "NVIDIA", __import__('wmi').WMI().Win32_VideoController())))

from ray import tune
from ray.tune.registry import register_env
from ray.rllib.examples.env.dm_control_suite import cartpole_swingup
import os

from dreamer.dreamer import DREAMERTrainer


os.environ['MUJOCO_GL'] = 'egl'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

ray.init()
register_env("cartpole", lambda x: cartpole_swingup())
tune.run(
    DREAMERTrainer,
    name="dmc-dreamer",
    stop={"timesteps_total": 100000},
    local_dir=os.path.join(os.getcwd(), "dmc"),
    checkpoint_at_end=True,
    #restore="/Users/kfarid/PycharmProjects/hyperdreamer/dmc/dmc-dreamer/DREAMER_cheetah_run-v20_beca2_00000_0_2021-08-13_18-37-27/checkpoint_26/checkpoint-26",
    config={
        "batch_size": 64,
        "framework": "torch",
        "num_gpus": 1,
        "num_workers": 0,
        "env": "cartpole",
        "dreamer_model": tune.grid_search([
            {"augment": {"params": {"consistent": True}, "augmented_target": False}},
            {},
            {"augment": {"params": {"consistent": True}, "augmented_target": True}},
            {"augment": {"params": {"consistent": False}, "augmented_target": False}},
            {"augment": {"params": {"consistent": False}, "augmented_target": True}},
        ])
    }
)
