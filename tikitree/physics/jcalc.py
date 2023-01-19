import jax
import jax.numpy as jnp

from tikitree import maths
from tikitree.base import Motion, Transform


def jcalc(joint_type: jnp.ndarray, q):
    def revolute(q):
        t = Transform.create(rot=maths.quat_rot_axis(jnp.array([0, 0, 1]), q))
        s = Motion.create(ang=[0.0, 0, 1])
        return t, s

    def prismatic(q):
        t = Transform.create(pos=[0.0, 0, q])
        s = Motion.create(vel=[0.0, 0, 1])
        return t, s

    def frozen(q):
        del q
        t = Transform.zero()
        s = Motion.zero()
        return t, s

    return jax.lax.switch(joint_type, [revolute, prismatic, frozen], q)
