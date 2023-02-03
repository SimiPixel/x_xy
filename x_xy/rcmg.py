from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as random

from x_xy import maths
from x_xy.base import System
from x_xy.kinematics import forward_kinematics, update_link_transform
from x_xy.random import random_angle_over_time, random_position_over_time


@dataclass
class RCMG_Parameters:
    t_min: float = 0.15  # min time between two generated angles
    t_max: float = 0.75  # max time ..
    dang_min: float = jnp.deg2rad(0)  # minimum angular velocity in deg/s
    dang_max: float = jnp.deg2rad(120)  # maximum angular velocity in deg/s
    dang_min_global: float = jnp.deg2rad(0)
    dang_max_global: float = jnp.deg2rad(60)
    dpos_min: float = 0.001  # speed of translation
    dpos_max: float = 0.1
    pos_min: float = -2.5
    pos_max: float = +2.5
    randomized_interpolation: bool = False
    range_of_motion: bool = True
    range_of_motion_method: str = "uniform"


class RCMG_Callback:
    def A_at_start(self, key, sys, extras, Ts):
        """Use this for
        - a random chain geometry
        - random joint axes
        by modifying `sys.Xtree`
        """
        return sys, extras

    def B_before_kinematics(self, key, sys, q, extras, Ts):
        """Use this for
        - Swapping alpha and beta -- (To perform perfectly non-observable motion)
        """
        return sys, q, extras

    def C_after_kinematics(self, key, sys, x, extras, Ts):
        return x, extras

    def D_at_return_value(self, key, sys, q, x, extras, Ts):
        """Use this for
        - creating measurement / sensor data
        - sensor2segment orientation
        - sensor2segment position
        - random absolute orientation
        - creating the return value of the function
        """
        return extras


class RCMB_Callback_6D_IMU_at_nodes(RCMG_Callback):
    def __init__(self, nodes: list[int], gravity: jax.Array, Ts: float):
        @jax.vmap
        def imu_measurement_function(rot, pos):
            extras = {}
            for i in nodes:
                extras[f"node_{i}"] = {}
                extras[f"node_{i}"]["gyr"] = quat2gyr(rot[i], Ts)
                extras[f"node_{i}"]["acc"] = pos2acc(rot[i], pos[i], gravity, Ts)
                extras[f"node_{i}"]["quat_to_earth"] = rot[i]
            return extras

        self.measure = imu_measurement_function

    def D_at_return_value(self, key, sys, q, x, extras, Ts):
        extras.update(self.measure(x.rot, x.pos))
        return extras


def draw_angle_and_pos(params: RCMG_Parameters, T, Ts):
    # TODO
    ANG_0 = 0.0
    POS_0 = 0.0
    max_it = 100

    # if one choses key_t identical for two joints then
    # the relative motion is synchronised
    def _random_angle(key_t, key_ang):
        return random_angle_over_time(
            key_t,
            key_ang,
            ANG_0,
            params.dang_min,
            params.dang_max,
            params.t_min,
            params.t_max,
            T,
            Ts,
            params.randomized_interpolation,
            params.range_of_motion,
            params.range_of_motion_method,
        )

    def _random_pos(key, _):
        return random_position_over_time(
            key,
            POS_0,
            params.pos_min,
            params.pos_max,
            params.dpos_min,
            params.dpos_max,
            params.t_min,
            params.t_max,
            T,
            Ts,
            max_it,
        )

    def _draw_angle_and_pos(joint_type, key_t, key_value):
        is_revolute = jnp.any(jnp.isclose(joint_type, jnp.array([0, 1, 2])))
        return jax.lax.cond(
            is_revolute,
            _random_angle,
            _random_pos,
            key_t,
            key_value,
        )

    return _draw_angle_and_pos


def rcmg(sys: System, T, Ts, params: RCMG_Parameters, callbacks: list[RCMG_Callback]):
    def generator(key):
        nonlocal sys
        extras = {}

        for cb in callbacks:
            key, consume = random.split(key)
            sys, extras = cb.A_at_start(consume, sys, extras, Ts)

        # generalized coordinates q
        key, *consume = random.split(key, sys.N * 2 + 1)
        consume = jnp.array(consume).reshape((2, sys.N, 2))
        q = jax.vmap(draw_angle_and_pos(params, T, Ts))(
            sys.links.joint.joint_type, consume[0], consume[1]
        )
        q = q.T  # shape of q before: (sys.N, T / Ts)

        for cb in callbacks:
            key, consume = random.split(key)
            sys, q, extras = cb.B_before_kinematics(consume, sys, q, extras, Ts)

        @jax.vmap
        def vmap_forward_kinematics(q):
            nonlocal sys
            sys = update_link_transform(sys, q)
            x = forward_kinematics(sys)
            return x

        x = vmap_forward_kinematics(q)

        for cb in callbacks:
            key, consume = random.split(key)
            x, extras = cb.C_after_kinematics(consume, sys, x, extras, Ts)

        for cb in callbacks:
            key, consume = random.split(key)
            extras = cb.D_at_return_value(consume, sys, q, x, extras, Ts)

        return extras

    return generator


def pos2acc(q, pos, gravity, Ts):
    N = len(q)
    acc = jnp.zeros((N, 3))
    acc = acc.at[1:-1].set((pos[:-2] + pos[2:] - 2 * pos[1:-1]) / Ts**2)
    acc = acc + gravity
    # TODO
    # used to be qinv
    return maths.rotate(acc, q)


def quat2gyr(q, Ts):

    q = jnp.vstack((q, jnp.array([[1.0, 0, 0, 0]])))
    # 1st-order approx to derivative
    dq = maths.quat_mul(maths.quat_inv(q[:-1]), q[1:])
    dt = Ts

    axis, angle = maths.quat_to_rot_axis(dq)
    angle = angle[:, None]

    gyr = axis * angle / dt
    return jnp.where(jnp.abs(angle) > 1e-10, gyr, jnp.zeros(3))
