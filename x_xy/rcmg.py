import functools as ft
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as random

from x_xy import maths
from x_xy.base import System
from x_xy.kinematics import forward_kinematics, update_link_transform
from x_xy.random import random_angle_over_time, random_position_over_time

# TODO
# dang_min_global and dang_max_global are currently unused
# the reason is that in `draw_angle_and_pos` we don't differentiate
# between a global or local dof; how would we even do that?


@dataclass(eq=True, frozen=True)
class RCMG_Parameters:
    t_min: float = 0.15  # min time between two generated angles
    t_max: float = 0.75  # max time ..
    dang_min: float = float(jnp.deg2rad(0))  # minimum angular velocity in deg/s
    dang_max: float = float(jnp.deg2rad(120))  # maximum angular velocity in deg/s
    dang_min_global: float = float(jnp.deg2rad(0))
    dang_max_global: float = float(jnp.deg2rad(60))
    dpos_min: float = 0.001  # speed of translation
    dpos_max: float = 0.1
    pos_min: float = -2.5
    pos_max: float = +2.5


@dataclass(eq=True, frozen=True)
class RCMG_Flags:
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
        """
        The x.rot is
            Top / FROM   -> EPS
                                q
            Bottom / TO  -> NODE

        The x.pos is
            Top / FROM   -> EPS
                                pos
            Bottom / TO  -> NODE   EPS <- Given in Frame

        Use this for
        - creating measurement / sensor data
        - sensor2segment orientation
        - sensor2segment position
        - random absolute orientation
        - creating the return value of the function
        """
        return extras

    def hidden(self, key, vmapped_system, q):
        return vmapped_system


def draw_angle_and_pos(
    params: RCMG_Parameters, flags: RCMG_Flags, T, Ts, floating_base: bool = False
):
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
            params.dang_min_global if floating_base else params.dang_min,
            params.dang_max_global if floating_base else params.dang_max,
            params.t_min,
            params.t_max,
            T,
            Ts,
            flags.randomized_interpolation,
            flags.range_of_motion,
            flags.range_of_motion_method,
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


@ft.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def rcmg(
    key,
    sys: System,
    T,
    Ts,
    batchsize: int = 1,
    params: RCMG_Parameters = RCMG_Parameters(),
    flags: RCMG_Flags = RCMG_Flags(),
    callbacks: tuple[RCMG_Callback] = (),
):
    def generator(key):
        nonlocal sys
        extras = {}

        for cb in callbacks:
            key, consume = random.split(key)
            sys, extras = cb.A_at_start(consume, sys, extras, Ts)

        # generalized coordinates q
        key, *consume = random.split(key, sys.N * 2 + 1)
        consume = jnp.array(consume).reshape((2, sys.N, 2))
        q_floating_base = jax.vmap(
            draw_angle_and_pos(params, flags, T, Ts, floating_base=True)
        )(sys.links.joint.joint_type[:6], consume[0][:6], consume[1][:6])
        q = jax.vmap(draw_angle_and_pos(params, flags, T, Ts))(
            sys.links.joint.joint_type[6:], consume[0][6:], consume[1][6:]
        )
        q = jnp.concatenate((q_floating_base, q))
        q = q.T  # shape of q before: (sys.N, T / Ts)

        for cb in callbacks:
            key, consume = random.split(key)
            sys, q, extras = cb.B_before_kinematics(consume, sys, q, extras, Ts)

        @ft.partial(jax.vmap, in_axes=(None, 0))
        def vmap_update_link_transform(sys, q):
            sys = update_link_transform(sys, q)
            return sys

        vmapped_system = vmap_update_link_transform(sys, q)

        for cb in callbacks:
            key, consume = random.split(key)
            vmapped_system = cb.hidden(consume, vmapped_system, q)

        x = jax.vmap(forward_kinematics)(vmapped_system)

        for cb in callbacks:
            key, consume = random.split(key)
            x, extras = cb.C_after_kinematics(consume, sys, x, extras, Ts)

        for cb in callbacks:
            key, consume = random.split(key)
            extras = cb.D_at_return_value(consume, sys, q, x, extras, Ts)

        return extras

    pmap_size, vmap_size = distribute_batchsize(batchsize)

    results = jax.pmap(jax.vmap(generator))(
        random.split(key, batchsize).reshape(pmap_size, vmap_size, 2)
    )

    # merge the pmap and vmap batch dimension
    results = merge_batchsize(results, pmap_size, vmap_size)
    results = jax.tree_map(jnp.squeeze, results)

    # do the unsqueeze in this scenario
    if batchsize == 1:
        results = jax.tree_map(lambda arr: arr[None], results)

    return results


def distribute_batchsize(batchsize: int) -> Tuple[int, int]:
    """Distributes batchsize accross pmap and vmap."""
    vmap_size_min = 8
    if batchsize <= vmap_size_min:
        return 1, batchsize
    else:
        n_devices = jax.local_device_count()
        print(n_devices)
        assert (
            batchsize % n_devices
        ) == 0, f"Your GPU count of {n_devices} does not split batchsize {batchsize}"
        vmap_size = int(batchsize / n_devices)
        return int(batchsize / vmap_size), vmap_size


def merge_batchsize(tree, pmap_size, vmap_size):
    return jax.tree_map(
        lambda arr: arr.reshape((pmap_size * vmap_size,) + arr.shape[2:]), tree
    )


def expand_batchsize(tree, pmap_size, vmap_size):
    return jax.tree_map(
        lambda arr: arr.reshape(
            (
                pmap_size,
                vmap_size,
            )
            + arr.shape[1:]
        ),
        tree,
    )


# APPROVED
def pos2acc(q, pos, gravity, Ts):
    N = len(q)
    acc = jnp.zeros((N, 3))
    acc = acc.at[1:-1].set((pos[:-2] + pos[2:] - 2 * pos[1:-1]) / Ts**2)

    if isinstance(gravity, float) or gravity.shape == ():
        gravity = jnp.array([0.0, 0, gravity])

    # gravity is a scalar value
    acc = acc + gravity
    # TODO
    # used to be qinv
    # the reason is that before q represented
    # FROM LOCAL TO EPS
    # but now we q represents
    # FROM EPS TO LOCAL
    return maths.rotate(acc, q)


# APPROVED
def quat2gyr(q, Ts):

    # this was not needed before
    # the reason is that before q represented
    # FROM LOCAL TO EPS
    # but now we q represents
    # FROM EPS TO LOCAL
    q = maths.quat_inv(q)

    q = jnp.vstack((q, jnp.array([[1.0, 0, 0, 0]])))
    # 1st-order approx to derivative
    dq = maths.quat_mul(maths.quat_inv(q[:-1]), q[1:])
    dt = Ts

    axis, angle = maths.quat_to_rot_axis(dq)
    angle = angle[:, None]

    gyr = axis * angle / dt
    return jnp.where(jnp.abs(angle) > 1e-10, gyr, jnp.zeros(3))
