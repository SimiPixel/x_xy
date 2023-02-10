"""
This module tests that the functionality migration from `rnno` to this package
has been kept.

What this tests:
    - I build a three segment kinematic chain 6-(5)-7
"""

import jax
import jax.numpy as jnp
import jax.random as random

from x_xy import maths
from x_xy import random as xxy_random
from x_xy import rcmg as rcmg_new


def rcmg(
    batchsize,
    randomized_interpolation=False,
    randomized_anchors=False,
    range_of_motion=True,
    range_of_motion_method="uniform",
    Ts=0.01,  # seconds
    T=60,  # seconds
    t_min=0.15,  # min time between two generated angles
    t_max=0.75,  # max time ...
    dang_min=jnp.deg2rad(0),  # minimum angular velocity in deg/s
    dang_max=jnp.deg2rad(180),  # maximum angular velocity in deg/s
    dang_min_global=jnp.deg2rad(0),
    dang_max_global=jnp.deg2rad(60),
    dpos_min=0.001,  # speed of translation
    dpos_max=0.1,
    pos_min=-2.5,
    pos_max=+2.5,
    param_ident=None,
    jit=True,
    legacy=False,
):
    pmap_size, vmap_size = rcmg_new.distribute_batchsize(batchsize)

    def generator(key):
        keys = random.split(key, batchsize).reshape(pmap_size, vmap_size, 2)
        return rcmg_new.merge_batchsize(
            generateMovementSparse(keys), pmap_size, vmap_size
        )

    max_it = 100
    g = 9.81

    @jax.pmap
    @jax.vmap
    @(jax.jit if jit else lambda f: f)
    def generateMovementSparse(key):

        if param_ident is not None:
            key, consume = random.split(key)
            r_12, r_23, d = param_ident(consume)
        else:
            const = jnp.array([0.5, 0.5, 0.5])
            r_23 = r_12 = d = const

        # absolute chain rotation
        key, *consume = random.split(key, 7)
        consume = jnp.array(consume).reshape((3, 2, 2))

        @jax.vmap
        def constraint_generateAnglePoints(key_ang, key_t):
            return xxy_random.random_angle_over_time(
                key_t,
                key_ang,
                0.0,
                dang_min_global,
                dang_max_global,
                t_min,
                t_max,
                T,
                Ts,
                randomized_interpolation,
                range_of_motion,
                range_of_motion_method,
            )

        ANG = constraint_generateAnglePoints(consume[:, 0], consume[:, 1])
        ORI = jnp.transpose(ANG)

        # relative joint rotation
        key, *consume = random.split(key, 13)
        consume = jnp.array(consume).reshape((6, 2, 2))

        @jax.vmap
        def constraint_generateAnglePoints(key_ang, key_t):
            return xxy_random.random_angle_over_time(
                key_t,
                key_ang,
                0.0,
                dang_min,
                dang_max,
                t_min,
                t_max,
                T,
                Ts,
                randomized_interpolation,
                range_of_motion,
                range_of_motion_method,
            )

        ANG_12 = constraint_generateAnglePoints(consume[:3, 0], consume[:3, 1])
        ANG_32 = constraint_generateAnglePoints(consume[3:, 0], consume[3:, 1])
        JA_12 = jnp.transpose(ANG_12)
        JA_32 = jnp.transpose(ANG_32)

        q = maths.quat_euler(ORI)

        key, consume = random.split(key)
        q_12, q_23 = random_hinge(consume, JA_12, JA_32)

        if randomized_anchors:
            key, consume = random.split(key)
            q_1, q_2, q_3 = randomize_anchor_qs(consume, q, q_12, q_23)
        else:
            q_1, q_2, q_3 = q1_is_anchor(q, q_12, q_23)

        q_1, q_2, q_3 = map(maths.quat_positive_w, (q_1, q_2, q_3))

        @jax.vmap
        def constraint_generatePosPoints(key):
            return xxy_random.random_position_over_time(
                key,
                0.0,
                pos_min,
                pos_max,
                dpos_min,
                dpos_max,
                t_min,
                t_max,
                T,
                Ts,
                max_it,
            )

        key, *consume = random.split(key, 4)
        pos = constraint_generatePosPoints(jnp.array(consume))
        pos = jnp.transpose(pos)

        r_12_earth = maths.safe_rotate(r_12, q_1)
        d_earth = maths.safe_rotate(d, q_2)
        r_23_earth = maths.safe_rotate(r_23, q_3)

        if randomized_anchors:
            key, consume = random.split(key)
            pos_1, pos_3 = randomize_anchor_pos(
                consume, pos, d_earth, r_12_earth, r_23_earth
            )
        else:
            pos_1, pos_3 = pos1_is_anchor(pos, d_earth, r_12_earth, r_23_earth)

        # because in this package those functions expect
        # q to follow the inverse convention
        q_1, q_2, q_3 = map(maths.quat_inv, (q_1, q_2, q_3))

        acc_1 = rcmg_new.pos2acc(q_1, pos_1, g, Ts)
        acc_3 = rcmg_new.pos2acc(q_3, pos_3, g, Ts)

        gyr_1 = rcmg_new.quat2gyr(q_1, Ts)
        gyr_3 = rcmg_new.quat2gyr(q_3, Ts)

        data = {
            "X": {0: {"acc": acc_1, "gyr": gyr_1}, 2: {"acc": acc_3, "gyr": gyr_3}},
            "y": {1: relquat_keepFrame(q_1, q_2), 2: relquat_keepFrame(q_2, q_3)},
        }

        return add_noise_and_bias(key, data)

    if legacy:

        def legacy_generator(key):
            data = generator(key)
            data_expanded = rcmg_new.expand_batchsize(data, pmap_size, vmap_size)
            X_exp = data_expanded["X"]
            y_exp = data_expanded["y"]
            X = jnp.concatenate(
                (X_exp[0]["acc"], X_exp[0]["gyr"], X_exp[2]["acc"], X_exp[2]["gyr"]),
                axis=-1,
            )
            y = jnp.concatenate(
                (maths.quat_positive_w(y_exp[1]), maths.quat_positive_w(y_exp[2])),
                axis=-1,
            )
            toy_acc = jnp.zeros_like(X_exp[0]["acc"])
            chain_dims = {value: toy_acc for value in ["d", "r_12", "r_23"]}

            return (
                X,
                {"quaternions": y, "chain_dimensions": chain_dims},
                maths.quat_unit_quats_like(y_exp[1]),
            )

        return legacy_generator

    return generator


def add_noise_and_bias(key, data):
    print("uses noise and bias..")
    noisy_data = {"X": {0: {}, 2: {}}, "y": data["y"]}
    noise_level = {"gyr": jnp.deg2rad(1.0), "acc": 0.5}
    bias_level = noise_level

    for i in [0, 2]:
        for sensor in ["acc", "gyr"]:
            measure = data["X"][i][sensor]
            key, c1, c2 = random.split(key, 3)
            noise = random.normal(c1, shape=measure.shape) * noise_level[sensor]
            bl = bias_level[sensor]
            bias = random.uniform(c2, minval=-bl, maxval=+bl)
            noisy_data["X"][i][sensor] = measure + noise + bias

    return noisy_data


def relquat_keepFrame(q1, q2):
    q1 = maths.safe_normalize(q1)
    q2 = maths.safe_normalize(q2)
    return maths.quat_mul(q1, maths.quat_inv(q2))


def pos1_is_anchor(pos, d, r_12, r_23):
    return pos, pos + r_12 + d + r_23


def posj1_is_anchor(pos, d, r_12, r_23):
    return pos - r_12, pos + d + r_23


def posj2_is_anchor(pos, d, r_12, r_23):
    return pos - r_12 - d, pos + r_23


def pos2_is_anchor(pos, d, r_12, r_23):
    return pos - r_12 - d - r_23, pos


def randomize_anchor_pos(key, pos, d, r_12, r_23):

    anchor = jax.random.randint(key, (), 1, 5)
    return jax.lax.cond(
        anchor == 1,
        pos1_is_anchor,
        lambda pos, d, r_12, r_23: jax.lax.cond(
            anchor == 2,
            posj1_is_anchor,
            lambda pos, d, r_12, r_23: jax.lax.cond(
                anchor == 3, posj2_is_anchor, pos2_is_anchor, pos, d, r_12, r_23
            ),
            pos,
            d,
            r_12,
            r_23,
        ),
        pos,
        d,
        r_12,
        r_23,
    )


def q1_is_anchor(q, q_12, q_23):
    q_2 = maths.quat_mul(q, q_12)
    q_3 = maths.quat_mul(q_2, q_23)
    return q, q_2, q_3


def q2_is_anchor(q, q_12, q_23):
    # no need to invert; the angle of the quaternion is random
    q_1 = maths.quat_mul(q, q_12)
    q_3 = maths.quat_mul(q, q_23)
    return q_1, q, q_3


def q3_is_anchor(q, q_12, q_23):
    q_2 = maths.quat_mul(q, q_23)
    q_1 = maths.quat_mul(q_2, q_12)
    return q_1, q_2, q


def randomize_anchor_qs(key, q, q_12, q_23):
    anchor = jax.random.randint(key, (), 1, 4)
    return jax.lax.cond(
        anchor == 1,
        q1_is_anchor,
        lambda q, q_12, q_23: jax.lax.cond(
            anchor == 2, q2_is_anchor, q3_is_anchor, q, q_12, q_23
        ),
        q,
        q_12,
        q_23,
    )


def random_hinge(key, ANG_12, ANG_32):
    @jax.vmap
    def draw_random_joint_axis(key):
        J = jax.random.uniform(key, (3,), minval=-1.0, maxval=1.0)
        Jnorm = jax.numpy.linalg.norm(J)
        return J / Jnorm

    Js = draw_random_joint_axis(jax.random.split(key))
    J_12, J_32 = Js[0], Js[1]

    @jax.vmap
    def q_12(ANG):
        return maths.quat_rot_axis(J_12, ANG[0])

    @jax.vmap
    def q_32(ANG):
        return maths.quat_rot_axis(J_32, ANG[0])

    return q_12(ANG_12), q_32(ANG_32)


def param_ident_dustin(key):
    consume = random.split(key, 9)

    r_12 = jnp.array(
        [
            random.uniform(consume[0], minval=0.05, maxval=0.25),
            random.uniform(consume[1], minval=-0.05, maxval=0.05),
            random.uniform(consume[2], minval=-0.05, maxval=0.05),
        ]
    )
    r_23 = jnp.array(
        [
            random.uniform(consume[3], minval=0.05, maxval=0.25),
            random.uniform(consume[4], minval=-0.05, maxval=0.05),
            random.uniform(consume[5], minval=-0.05, maxval=0.05),
        ]
    )
    d = jnp.array(
        [
            random.uniform(consume[6], minval=0.1, maxval=0.35),
            random.uniform(consume[7], minval=-0.02, maxval=0.02),
            random.uniform(consume[8], minval=-0.02, maxval=0.02),
        ]
    )
    return r_12, r_23, d
