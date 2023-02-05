import jax
import jax.numpy as jnp

from x_xy import base, rcmg, rcmg_callbacks
from x_xy.utils import add_floating_base


def three_segment_system() -> base.System:
    """Nodes: 6--(5)--7"""
    joint1 = base.Link.create(
        base.Transform.zero(), base.Joint(base.JointType.RevoluteY)
    )
    joint2 = base.Link.create(
        base.Transform.zero(),
        base.Joint(base.JointType.RevoluteZ),
    )
    sys = base.System(jnp.array([-1, -1]), joint1.batch(joint2))
    return add_floating_base(sys)


def three_segment_generator(T, Ts):
    sys = three_segment_system()

    @jax.jit
    def generator(key):
        return rcmg.rcmg(
            key,
            sys,
            T,
            Ts,
            params=rcmg.RCMG_Parameters(),
            flags=rcmg.RCMG_Flags(),
            callbacks=(
                rcmg_callbacks.RCMG_Callback_randomize_middle_segment_length(),
                rcmg_callbacks.RCMG_Callback_random_sensor2segment_position(),
                rcmg_callbacks.RCMG_Callback_6D_IMU_at_nodes(
                    [6, 7], [0, 2], sys.gravity, Ts
                ),
                rcmg_callbacks.RCMG_Callback_qrel_to_parent([5, 7], [6, 5], [1, 2]),
                rcmg_callbacks.RCMG_Callback_noise_and_bias([0, 2]),
            ),
        )

    return generator
