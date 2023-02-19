"""
This is the original Random Chain Motion Generator (RCMG) but modified
to simulate a kinematic chain with four segments.
"""

import jax
import jax.numpy as jnp
import jax.random as random

from x_xy import maths
from x_xy import random as xxy_random
from x_xy import rcmg as rcmg_new


def rcmg_4Seg(
    batchsize,
    randomized_interpolation=False,
    range_of_motion=True,
    range_of_motion_method="uniform",
    Ts=0.01,  # seconds
    T=60,  # seconds
    t_min=0.15,  # min time between two generated angles
    t_max=0.75,  # max time ...
    dang_min=jnp.deg2rad(0),  # minimum angular velocity in deg/s
    dang_max=jnp.deg2rad(120),  # maximum angular velocity in deg/s
    dang_min_global=jnp.deg2rad(0),
    dang_max_global=jnp.deg2rad(60),
    dpos_min=0.001,  # speed of translation
    dpos_max=0.1,
    pos_min=-2.5,
    pos_max=+2.5,
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
    def generateMovementSparse(key):

        # absolute chain rotation
        key, *consume = random.split(key, 7)
        consume = jnp.array(consume).reshape((3, 2, 2))

        @jax.vmap
        def constraint_generateAnglePoints(key_ang, key_t):
            return xxy_random.random_angle_over_time(
                key_ang,  # TODO
                key_t,
                0.0,
                dang_min_global,
                dang_max_global,
                t_min,
                t_max,
                T,
                Ts,
                randomized_interpolation,
                False,  # global orientation has no ROM restriction
                range_of_motion_method,
            )

        ANG = constraint_generateAnglePoints(consume[:, 0], consume[:, 1]).T
        q = maths.quat_euler(ANG)

        # relative joint rotation
        key, *consume = random.split(key, 19)
        consume = jnp.array(consume).reshape((9, 2, 2))

        @jax.vmap
        def constraint_generateAnglePoints(key_ang, key_t):
            return xxy_random.random_angle_over_time(
                key_ang,  # TODO
                key_t,
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

        ANG_12 = constraint_generateAnglePoints(consume[:3, 0], consume[:3, 1]).T
        ANG_23 = constraint_generateAnglePoints(consume[3:6, 0], consume[3:6, 1]).T
        ANG_34 = constraint_generateAnglePoints(consume[6:, 0], consume[6:, 1]).T

        key, consume = random.split(key)
        q_12 = random_hinge(consume, ANG_12)
        key, consume = random.split(key)
        q_23 = random_hinge(consume, ANG_23)
        key, consume = random.split(key)
        q_34 = random_hinge(consume, ANG_34)

        key, consume = random.split(key)
        q_1, q_2, q_3, q_4 = randomize_anchor_qs(consume, q, q_12, q_23, q_34)

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
        pos = constraint_generatePosPoints(jnp.array(consume)).T

        key, consume = random.split(key)
        r_1, d_1, d_2, r_2 = param_ident_dustin(consume)

        r_1 = maths.rotate(r_1, q_1)
        d_1 = maths.rotate(d_1, q_2)
        d_2 = maths.rotate(d_2, q_3)
        r_2 = maths.rotate(r_2, q_4)

        key, consume = random.split(key)
        pos_1, pos_2 = randomize_anchor_pos(consume, pos, r_1, d_1, d_2, r_2)

        # because in this package those functions expect
        # q to follow the inverse convention
        q_1, q_2, q_3, q_4 = map(maths.quat_inv, (q_1, q_2, q_3, q_4))

        acc_1 = rcmg_new.pos2acc(q_1, pos_1, g, Ts)
        acc_2 = rcmg_new.pos2acc(q_4, pos_2, g, Ts)

        gyr_1 = rcmg_new.quat2gyr(q_1, Ts)
        gyr_2 = rcmg_new.quat2gyr(q_4, Ts)

        data = {
            "X": {0: {"acc": acc_1, "gyr": gyr_1}, 3: {"acc": acc_2, "gyr": gyr_2}},
            "y": {
                1: relquat_keepFrame(q_1, q_2),
                2: relquat_keepFrame(q_2, q_3),
                3: relquat_keepFrame(q_3, q_4),
            },
        }

        return _add_noise_and_bias_to_gyr_and_acc(key, data)

    return generator


def _add_noise_and_bias_to_gyr_and_acc(key, data):
    noisy_data = {"X": {}, "y": data["y"]}
    noise_level = {"gyr": jnp.deg2rad(1.0), "acc": 0.5}
    bias_level = noise_level

    for i in data["X"].keys():
        noisy_data["X"].update({i: {}})
        for sensor in ["acc", "gyr"]:
            measure = data["X"][i][sensor]
            key, c1, c2 = random.split(key, 3)
            noise = random.normal(c1, shape=measure.shape) * noise_level[sensor]
            bl = bias_level[sensor]
            bias = random.uniform(c2, minval=-bl, maxval=+bl)
            noisy_data["X"][i][sensor] = measure + noise + bias

    return noisy_data


def relquat_keepFrame(q1, q2):
    return maths.quat_mul(q1, maths.quat_inv(q2))


def pos1_is_anchor(pos, r_1, d_1, d_2, r_2):
    return pos, pos + r_1 + d_1 + d_2 + r_2


def posj1_is_anchor(pos, r_1, d_1, d_2, r_2):
    return pos - r_1, pos + d_1 + d_2 + r_2


def posj2_is_anchor(pos, r_1, d_1, d_2, r_2):
    return pos - r_1 - d_1, pos + d_2 + r_2


def posj3_is_anchor(pos, r_1, d_1, d_2, r_2):
    return pos - r_1 - d_1 - d_2, pos + r_2


def pos2_is_anchor(pos, r_1, d_1, d_2, r_2):
    return pos - r_1 - d_1 - d_2 - r_2, pos


def randomize_anchor_pos(key, pos, r_1, d_1, d_2, r_2):
    anchor = jax.random.randint(key, (), 0, 5)
    return jax.lax.switch(
        anchor,
        (
            pos1_is_anchor,
            posj1_is_anchor,
            posj2_is_anchor,
            posj3_is_anchor,
            pos2_is_anchor,
        ),
        pos,
        r_1,
        d_1,
        d_2,
        r_2,
    )


def q1_is_anchor(q, q_12, q_23, q_34):
    q_2 = maths.quat_mul(q, q_12)
    q_3 = maths.quat_mul(q_2, q_23)
    q_4 = maths.quat_mul(q_3, q_34)
    return q, q_2, q_3, q_4


def q2_is_anchor(q, q_12, q_23, q_34):
    # no need to invert; the angle of the quaternion is random
    q_1 = maths.quat_mul(q, q_12)
    q_3 = maths.quat_mul(q, q_23)
    q_4 = maths.quat_mul(q_3, q_34)
    return q_1, q, q_3, q_4


def q3_is_anchor(q, q_12, q_23, q_34):
    q_2 = maths.quat_mul(q, q_23)
    q_1 = maths.quat_mul(q_2, q_12)
    q_4 = maths.quat_mul(q, q_34)
    return q_1, q_2, q, q_4


def q4_is_anchor(q, q_12, q_23, q_34):
    q_3 = maths.quat_mul(q, q_34)
    q_2 = maths.quat_mul(q_3, q_23)
    q_1 = maths.quat_mul(q_2, q_12)
    return q_1, q_2, q_3, q


def randomize_anchor_qs(key, q, q_12, q_23, q_34):
    anchor = jax.random.randint(key, (), 0, 4)
    return jax.lax.switch(
        anchor,
        (q1_is_anchor, q2_is_anchor, q3_is_anchor, q4_is_anchor),
        q,
        q_12,
        q_23,
        q_34,
    )


def random_hinge(key, ANG):
    def draw_random_joint_axis(key):
        J = jax.random.uniform(key, (3,), minval=-1.0, maxval=1.0)
        Jnorm = jax.numpy.linalg.norm(J)
        return J / Jnorm

    joint_axis = draw_random_joint_axis(key)

    @jax.vmap
    def q(ANG):
        return maths.quat_rot_axis(joint_axis, ANG[0])

    return q(ANG)


def param_ident_dustin(key):
    consume = random.split(key, 12).reshape(4, 3, 2)

    def create_r(key):
        return jnp.array(
            [
                random.uniform(key[0], minval=0.05, maxval=0.25),
                random.uniform(key[1], minval=-0.05, maxval=0.05),
                random.uniform(key[2], minval=-0.05, maxval=0.05),
            ]
        )

    def create_d(key):
        return jnp.array(
            [
                random.uniform(key[6], minval=0.1, maxval=0.35),
                random.uniform(key[7], minval=-0.02, maxval=0.02),
                random.uniform(key[8], minval=-0.02, maxval=0.02),
            ]
        )

    return (
        create_r(consume[0]),
        create_d(consume[1]),
        create_d(consume[2]),
        create_r(consume[3]),
    )
