import jax.numpy as jnp

from tikitree import maths, spatial
from tikitree.base import Box, Inertia, Joint, JointType, Link, System, Transform
from tikitree.dynamics import _compute_mass_matrix, inverse_dynamics
from tikitree.kinematics import forward_kinematics, update_link_transform

# these values are computed in MATLAB
ref = {
    "I_6x6": jnp.array(
        [
            [10.5081, 2.5201, 1.6069, 0, 0, 3.2000],
            [2.5201, 2.2391, 2.3389, 0, 0, -1.2000],
            [1.6069, 2.3389, 11.9908, -3.2000, 1.2000, 0],
            [0, 0, -3.2000, 1.0000, 0, 0],
            [0, 0, 1.2000, 0, 1.0000, 0],
            [3.2000, -1.2000, 0, 0, 0, 1.0000],
        ]
    ),
    "I_3x3": jnp.array(
        [
            [0.2681391, 6.36006338, 1.6068761],
            [6.36006338, 0.79910134, 2.33889834],
            [1.6068761, 2.33889834, 0.31081915],
        ]
    ),
    "CoM": jnp.array([1.2, 3.2, 0]),
    "mass": jnp.array(1.0),
    "I_6x6_after_forward_with_X": jnp.array(
        [
            [5.1081, 4.5508, -4.1134, 0, 1.8512, 1.1887],
            [4.5508, 6.0472, 1.0052, -1.8512, 0.0000, -0.2000],
            [-4.1134, 1.0052, -0.0173, -1.1887, 0.2000, 0.0000],
            [0, -1.8512, -1.1887, 1.0000, 0, 0],
            [1.8512, 0.0000, 0.2000, 0, 1.0000, -0.0000],
            [1.1887, -0.2000, -0.0000, 0, -0.0000, 1.0000],
        ]
    ),
    "Transform.pos": jnp.array([1, 1, 0.0]),
    "Transform.rot": maths.quat_from_3x3(spatial.rx(1.0)),
    "X": jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5403023, 0.841471, 0.0, 0.0, 0.0],
            [0.0, -0.841471, 0.5403023, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
            [0.841471, -0.841471, 0.5403023, 0.0, 0.5403023, 0.841471],
            [0.5403023, -0.5403023, -0.841471, 0.0, -0.841471, 0.5403023],
        ]
    ),
    "build_2_link_system_w_inertia": {
        "H": jnp.array([[73.3333, -10.0], [-10.0, 21.6667]]),
        "C": jnp.array([196.20001, -98.100006]),
    },
    "build_5_link_system_w_inertia": {
        "H": jnp.array(
            [
                [432.4185, 51.72377, 17.823357, 11.685562, 62.78546],
                [51.72377, 81.90752, 0.0, -4.212685, 17.669222],
                [17.823357, 0.0, 10.0, 0.0, 0.0],
                [11.685562, -4.212685, 0.0, 20.0, -1.336268],
                [62.78546, 17.669222, 0.0, -1.336268, 21.666666],
            ]
        ),
        "C": jnp.array([204.1036, -70.65334, 61.40471, -19.645096, -81.372955]),
        "tau": jnp.array([853.72986, -77.547806, 109.22806, -86.21554, 59.752487]),
        "forward_kinematics_X_link_5": jnp.array(
            [
                [0.14709038, -0.11325587, 0.9826178, 0.0, 0.0, 0.0],
                [-0.29055873, -0.9545419, -0.06652541, 0.0, 0.0, 0.0],
                [0.9454841, -0.27572286, -0.17331147, 0.0, 0.0, 0.0],
                [
                    -0.97066563,
                    0.16543113,
                    0.16436872,
                    0.14709038,
                    -0.11325587,
                    0.9826178,
                ],
                [
                    1.2254342,
                    -0.349398,
                    -0.33890152,
                    -0.29055873,
                    -0.9545419,
                    -0.06652541,
                ],
                [0.52759874, 1.1416496, 1.0620029, 0.9454841, -0.27572286, -0.17331147],
            ]
        ),
    },
    "build_5_link_system_wo_inertia": {
        "H": jnp.array(
            [
                [159.67403, 15.597029, 17.823357, 1.8737172, 0.0],
                [15.597029, 42.140446, 0.0, -2.1766448, 0.0],
                [17.823357, 0.0, 10.0, 0.0, 0.0],
                [1.8737172, -2.1766448, 0.0, 10.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        "C": jnp.array([516.4324, -114.888725, 61.40471, 104.64065, 0.0]),
        "tau": jnp.array([690.88763, -174.866, 109.22806, 70.86765, 0.0]),
    },
}

t = Transform(ref["Transform.pos"], ref["Transform.rot"])
inertia = Inertia.create(ref["mass"], ref["CoM"], ref["I_3x3"])


def get_box(mass=10):
    com = jnp.ones((3,))
    return Box(mass, com, *com)


def build_2_link_system_w_inertia():
    geom = get_box()
    l1 = Link.create(
        Transform.create(pos=jnp.array([0.0, 1.0, 0])),
        Joint(JointType.RevoluteX),
        geom,
    )
    l2 = Link.create(
        Transform.create(pos=jnp.array([0.0, 1.0, 0])),
        Joint(JointType.RevoluteZ),
        geom,
    )
    sys = System(jnp.array([-1, 0]), l1.batch(l2))
    q = jnp.array([jnp.pi / 2, 0.0])
    qd = jnp.array([0.0, 0])
    qdd = qd
    sys = update_link_transform(sys, q)
    return sys, q, qd, qdd


def build_5_link_system_w_inertia():
    geom = get_box()
    pos = jnp.array([1.0, 1.0, -1.0])
    qX = maths.quat_from_3x3(spatial.rx(1.77))
    qY = maths.quat_from_3x3(spatial.ry(1.77))
    qZ = maths.quat_from_3x3(spatial.rz(1.77))

    l1 = Link.create(
        Transform.create(pos=pos, rot=qX),
        Joint(JointType.RevoluteX),
        geom,
    )
    l2 = Link.create(
        Transform.create(pos=pos, rot=qY),
        Joint(JointType.RevoluteY),
        geom,
    )
    l3 = Link.create(
        Transform.create(pos=pos, rot=qZ),
        Joint(JointType.PrismaticZ),
        geom,
    )
    l4 = Link.create(
        Transform.create(pos=pos, rot=qX),
        Joint(JointType.PrismaticX),
        geom,
    )
    l5 = Link.create(
        Transform.create(pos=pos, rot=qY),
        Joint(JointType.RevoluteZ),
        geom,
    )
    sys = System(jnp.array([-1, 0, 0, 1, 3]), l1.batch(l2, l3, l4, l5))
    q = jnp.array([1, -2, 3, -4, 5.0])
    qd = -q
    qdd = q
    sys = update_link_transform(sys, q)
    return sys, q, qd, qdd


def build_5_link_system_wo_inertia():
    geom = get_box()
    geom_no_mass = get_box(0.0)
    pos = jnp.array([1.0, 1.0, -1.0])
    qX = maths.quat_from_3x3(spatial.rx(1.77))
    qY = maths.quat_from_3x3(spatial.ry(1.77))
    qZ = maths.quat_from_3x3(spatial.rz(1.77))

    l1 = Link.create(
        Transform.create(pos=pos, rot=qX),
        Joint(JointType.RevoluteX),
        geom_no_mass,
    )
    l2 = Link.create(
        Transform.create(pos=pos, rot=qY),
        Joint(JointType.RevoluteY),
        geom_no_mass,
    )
    l3 = Link.create(
        Transform.create(pos=pos, rot=qZ),
        Joint(JointType.PrismaticZ),
        geom,
    )
    l4 = Link.create(
        Transform.create(pos=pos, rot=qX),
        Joint(JointType.PrismaticX),
        geom,
    )
    l5 = Link.create(
        Transform.create(pos=pos, rot=qY),
        Joint(JointType.RevoluteZ),
        geom_no_mass,
    )
    sys = System(jnp.array([-1, 0, 0, 1, 3]), l1.batch(l2, l3, l4, l5))
    q = jnp.array([1, -2, 3, -4, 5.0])
    qd = -q
    qdd = q
    sys = update_link_transform(sys, q)
    return sys, q, qd, qdd


def test_transform():
    assert jnp.allclose(t.as_matrix(), ref["X"], atol=1e-4)


def test_inertia():
    assert jnp.allclose(inertia.as_matrix(), ref["I_6x6"], atol=1e-4)


def test_inertia_transform():
    # forward transform X^* x I_6x6 x X^-1
    assert jnp.allclose(
        t.do(inertia).as_matrix(), ref["I_6x6_after_forward_with_X"], atol=1e-4
    )


def test_composite_rigid_body_algorithm():
    for sys_id in [
        "build_2_link_system_w_inertia",
        "build_5_link_system_w_inertia",
        "build_5_link_system_wo_inertia",
    ]:
        sys, q, qd, qdd = eval(sys_id)()
        ours = _compute_mass_matrix(sys)
        roys = ref[sys_id]["H"]
        assert jnp.allclose(ours, roys, atol=1e-4)


def test_inverse_dynamics_algorithm():
    for sys_id in [
        "build_2_link_system_w_inertia",
        "build_5_link_system_w_inertia",
        "build_5_link_system_wo_inertia",
    ]:
        sys, q, qd, qdd = eval(sys_id)()
        ours, stop = inverse_dynamics(sys, qd, jnp.zeros_like(q))
        roys = ref[sys_id]["C"]
        assert jnp.allclose(ours, roys, atol=1e-4)
        assert stop == -1

        # this test was not recorded with this model
        if sys_id == "build_2_link_system_w_inertia":
            continue

        ours, stop = inverse_dynamics(sys, qd, qdd)
        roys = ref[sys_id]["tau"]
        assert jnp.allclose(ours, roys, atol=1e-4)
        assert stop == -1


def test_forward_kinematics():
    sys, *_ = build_5_link_system_w_inertia()
    Xs = forward_kinematics(sys)
    ours = Xs.take(-1).as_matrix()
    roys = ref["build_5_link_system_w_inertia"]["forward_kinematics_X_link_5"]
    assert jnp.allclose(ours, roys)
