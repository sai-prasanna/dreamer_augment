import ray
import subprocess
#ray.resource_spec._autodetect_num_gpus = lambda: len(list(filter(lambda c: c.wmi_property('AdapterCompatibility').value == "NVIDIA", __import__('wmi').WMI().Win32_VideoController())))

from ray import tune
from ray.tune.registry import register_env
from ray.rllib.examples.env.dm_control_suite import cheetah_run
import os

from dreamer.dreamer import DREAMERTrainer


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

ray.init()
register_env("cheetah_run", lambda x: cheetah_run())
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
        #"matching_coeff": 0.0001,
        #"mem_smoothing": 0.1,
        "env": "cheetah_run",
        "dreamer_model": {
            "consistent": True#tune.grid_search([True, False])
        },
    },
)
