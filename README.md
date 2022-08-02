# Dreamer Augment

## Dreamer 2

We build our work upon the [Dreamerv2](https://github.com/jsikyoon/dreamer-torch) pytorch implementation.

### Environment

In a conda or virtual env, install the requirements.

`pip install -r requirements.txt`

### Training

The `dreamer-torch/config.yaml` has various configurations that can be selected using `--config` argument and every individual keys can be overridden using `--key`.

Note: 

1. Setting batch size to 156 is important for sample efficiency.
2. You can disable weights and biases logging by passing `--wandb False`.

Few examples are below.
#### Dreamerv2 Baseline

```sh
cd dreamer-torch
python dreamer.py --configs defaults dmc --logdir fs_baseline --seed 42 --task dmc_finger_spin --wandb_name fs_baseline
```

#### Dreamerv2 CURL

```sh
cd dreamer-torch
python dreamer.py --configs defaults dmc --logdir ch_curl --size 84,84 --seed 1337 --task dmc_cheetah_run --augment True --augment_random_crop True --augment_pad 0 --augment_crop_size 64,64 --augment_strong False --augment_consistent True --wandb_name ch_curl --curl True --batch_size 150 --seed 1337 --batch_size 156
```

### Evaluation

Use `dreamer-torch/dreamer_eval.py` for running evaluation.

## Dreamer 1 Rllib

Our prior experiments used rllib's dreamer1 implementation, which we modified to include some elements from dreamer2. For running them, check the `dreamer/train.py`. The configurations can be run together with grid search.