import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import checkify

from x_xy import base, maths, rcmg


class RCMG_Callback_6D_IMU_at_nodes(rcmg.RCMG_Callback):
    def __init__(
        self, nodes: list[int], convention: list[int], gravity: jax.Array, Ts: float
    ):
        def imu_measurement_function(rot, pos):
            extras = {"X": {}}
            for i, conv_i in zip(nodes, convention):
                extras["X"][conv_i] = {}
                node = extras["X"][conv_i]
                rot_i, pos_i = rot[:, i], pos[:, i]
                node["gyr"] = rcmg.quat2gyr(rot_i, Ts)
                node["acc"] = rcmg.pos2acc(rot_i, pos_i, gravity, Ts)
            return extras

        self.measure = imu_measurement_function

    def D_at_return_value(self, key, sys, q, x, extras, Ts):
        extras.update(self.measure(x.rot, x.pos))
        return extras


class RCMG_Callback_qrel_to_parent(rcmg.RCMG_Callback):
    def __init__(self, nodes: list[int], parents: list[int], convention: list[int]):
        def build_y(rot):
            extras = {"y": {}}
            for i, par_i, conv_i in zip(nodes, parents, convention):
                # This really is identical to q1^{-1} x q2
                # because we should do q -> q^{-1} for both
                # before; but it cancels to this expression
                qrel = maths.quat_mul(rot[:, par_i], maths.quat_inv(rot[:, i]))
                extras["y"][conv_i] = qrel
            return extras

        self.build_y = build_y

    def D_at_return_value(self, key, sys, q, x, extras, Ts):
        extras.update(self.build_y(x.rot))
        return extras


class RCMG_Callback_noise_and_bias(rcmg.RCMG_Callback):
    def __init__(
        self,
        nodes: list[int],
        noise_stds={"gyr": jnp.deg2rad(1.0), "acc": 0.5},
        bias_minmax={"gyr": (-jnp.deg2rad(1.0), jnp.deg2rad(1.0)), "acc": (-0.5, 0.5)},
    ):
        def noisify(extras, key):
            for i in nodes:
                X_i = extras["X"][i]
                for sensor in ["acc", "gyr"]:
                    measurement = X_i[sensor]
                    key, c1, c2 = random.split(key, 3)
                    noise = (
                        random.normal(c1, shape=measurement.shape) * noise_stds[sensor]
                    )
                    bias = random.uniform(
                        c2, minval=bias_minmax[sensor][0], maxval=bias_minmax[sensor][1]
                    )
                    X_i[sensor] = measurement + noise + bias

            return extras

        self.noisify = noisify

    def D_at_return_value(self, key, sys, q, x, extras, Ts):
        extras.update(self.noisify(extras, key))
        return extras


class RCMG_Callback_randomize_middle_segment_length(rcmg.RCMG_Callback):
    def __init__(self):
        @checkify.checkify
        def check_parents(sys):
            checkify.check(
                jnp.allclose(sys.parent[-2:], jnp.array([5, 5])),
                "The parents should be (5, 5) but they are {}",
                sys.parent[-2:],
            )

        self.check = check_parents

    def A_at_start(self, key, sys, extras, Ts):
        keys = random.split(key, 3)
        trafo_x = random.uniform(keys[0], minval=0.05, maxval=0.2, shape=(2,))
        trafo_y = random.uniform(keys[1], minval=-0.02, maxval=0.02, shape=(2,))
        trafo_z = random.uniform(keys[2], minval=-0.02, maxval=0.02, shape=(2,))

        self.check(sys)

        def update_position(sys: base.System, at, new_pos):
            return sys.replace(
                links=sys.links.replace(
                    Xtree=sys.links.Xtree.index_set(
                        at, sys.links.Xtree.take(at).replace(pos=new_pos)
                    )
                )
            )

        # TODO
        # That minus sign is arbitrary..
        sys = update_position(sys, 6, jnp.array([-trafo_x[0], trafo_y[0], trafo_z[0]]))
        sys = update_position(sys, 7, jnp.array([trafo_x[1], trafo_y[1], trafo_z[1]]))
        return super().A_at_start(key, sys, extras, Ts)


class RCMG_Callback_random_sensor2segment_position(rcmg.RCMG_Callback):
    def C_after_kinematics(self, key, sys, x: base.Transform, extras, Ts):
        keys = random.split(key, 3)
        trafo_x = random.uniform(keys[0], minval=0.05, maxval=0.2, shape=(2,))
        trafo_y = random.uniform(keys[1], minval=-0.05, maxval=0.05, shape=(2,))
        trafo_z = random.uniform(keys[2], minval=-0.05, maxval=0.05, shape=(2,))

        def update_x(x, at, new_pos):
            segment_to_sensor_trafo1 = base.Transform.create(pos=new_pos)
            new_trafo = segment_to_sensor_trafo1.do(x.take(at))
            return x.index_set(at, new_trafo)

        x = update_x(x, 6, jnp.array([-trafo_x[0], trafo_y[0], trafo_z[0]]))
        x = update_x(x, 7, jnp.array([trafo_x[0], trafo_y[0], trafo_z[0]]))

        return super().C_after_kinematics(key, sys, x, extras, Ts)


class RCMG_Callback_random_joint_axes(rcmg.RCMG_Callback):
    def A_at_start(self, key, sys, extras, Ts):
        def update_rot(sys: base.System, at, new_rot):
            return sys.replace(
                links=sys.links.replace(
                    Xtree=sys.links.Xtree.index_set(
                        at, sys.links.Xtree.take(at).replace(rot=new_rot)
                    )
                )
            )

        two_random_quats = maths.quat_random(key, (2,))
        sys = update_rot(sys, 6, two_random_quats[0])
        sys = update_rot(sys, 7, two_random_quats[1])
        return super().A_at_start(key, sys, extras, Ts)