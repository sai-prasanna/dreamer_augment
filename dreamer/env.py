from ray.rllib.env.wrappers.dm_control_wrapper import DMCEnv

"""
8 Environments from Deepmind Control Suite
"""

def finger_spin(
    from_pixels=True, height=64, width=64, frame_skip=2, channels_first=True, **kwargs
):
    return DMCEnv(
        "finger",
        "spin",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first,
        **kwargs
    )

def acrobot_swingup(
    from_pixels=True, height=64, width=64, frame_skip=2, channels_first=True, **kwargs
):
    return DMCEnv(
        "acrobot",
        "swingup",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first,
        **kwargs
    )


def walker_walk(
    from_pixels=True, height=64, width=64, frame_skip=2, channels_first=True, **kwargs
):
    return DMCEnv(
        "walker",
        "walk",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first,
        **kwargs
    )


def hopper_hop(
    from_pixels=True, height=64, width=64, frame_skip=2, channels_first=True, **kwargs
):
    return DMCEnv(
        "hopper",
        "hop",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first,
        **kwargs
    )


def hopper_stand(
    from_pixels=True, height=64, width=64, frame_skip=2, channels_first=True, **kwargs
):
    return DMCEnv(
        "hopper",
        "stand",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first,
        **kwargs
    )


def cheetah_run(
    from_pixels=True, height=64, width=64, frame_skip=2, channels_first=True, **kwargs
):
    return DMCEnv(
        "cheetah",
        "run",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first,
        **kwargs
    )


def walker_run(
    from_pixels=True, height=64, width=64, frame_skip=2, channels_first=True, **kwargs
):
    return DMCEnv(
        "walker",
        "run",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first,
        **kwargs
    )


def pendulum_swingup(
    from_pixels=True, height=64, width=64, frame_skip=2, channels_first=True, **kwargs
):
    return DMCEnv(
        "pendulum",
        "swingup",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first,
        **kwargs
    )


def cartpole_swingup(
    from_pixels=True, height=64, width=64, frame_skip=2, channels_first=True,  **kwargs
):
    return DMCEnv(
        "cartpole",
        "swingup",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first,
        **kwargs
    )


def humanoid_walk(
    from_pixels=True, height=64, width=64, frame_skip=2, channels_first=True,  **kwargs
):
    return DMCEnv(
        "humanoid",
        "walk",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first,
        **kwargs
    )
