import jax
import jax.numpy as jnp
from x_xy import maths
from x_xy.base import JointType, Link, Motion, System, Transform


def jcalc(joint_type: JointType, q):
    axes = {
        "x": jnp.array([1.0, 0, 0]),
        "y": jnp.array([0.0, 1, 0]),
        "z": jnp.array([0.0, 0, 1]),
    }

    def revolute(axis):
        def _revolute(q):
            # TODO
            # inconsistent angle handness convention with spatial.py
            t = Transform.create(rot=maths.quat_rot_axis(axes[axis], -q))
            s = Motion.create(ang=axes[axis])
            return t, s

        return _revolute

    def prismatic(axis):
        def _prismatic(q):
            t = Transform.create(pos=axes[axis] * q)
            s = Motion.create(vel=axes[axis])
            return t, s

        return _prismatic

    return jax.lax.switch(
        jnp.squeeze(joint_type),
        [
            revolute("x"),
            revolute("y"),
            revolute("z"),
            prismatic("x"),
            prismatic("y"),
            prismatic("z"),
        ],
        q,
    )


def update_link_transform(sys: System, q: jax.Array) -> System:
    @jax.vmap
    def _update(link: Link, q):
        t_joint, s = jcalc(link.joint.joint_type, q)
        new_transform = t_joint.do(link.Xtree)
        new_joint = link.joint.replace(motion=s)
        return link.replace(transform=new_transform, joint=new_joint)

    new_links = _update(sys.links, q)
    return sys.replace(links=new_links)


def forward_kinematics(sys: System) -> Transform:
    """Computes all world to link transforms"""

    def body_fn(i: int, transforms_world: Transform):
        p = sys.parent[i]
        i_world = sys.links.take(i).transform.do(transforms_world.take(p))
        transforms_world = transforms_world.index_set(i, i_world)
        return transforms_world

    return jax.lax.fori_loop(0, sys.N, body_fn, Transform.zero((sys.N,)))


"""
def inverse_kinematics(m, b: Link, td: Transform, q0: jax.Array) -> jax.Array:
    def cond_fn(val):
        _, dpos = val
        return jnp.linalg.norm(dpos) > 0.1

    def loop_fn(val):
        q, _ = val
        t = forward_kinematics(m, b, q)
        J = bodyJac(m, b, q)
        J = jax.vmap(t.do)(J)
        J_matrix = jnp.hstack((J.ang, J.vel)).T
        dpos = spatial.XtoV(td.do(t.inv()).to_matrix())
        dq = jnp.linalg.pinv(J_matrix) @ dpos
        return (q + jnp.squeeze(dq), jnp.squeeze(dpos))

    return jax.lax.while_loop(cond_fn, loop_fn, (q0, jnp.ones((6,))))


def bodyJac(m, body: Link, q: jax.Array) -> Motion:
    J = Motion.zero((m.graph.N,))
    t = Transform.zero()
    for b in m.root_to_tip():
        if b.i not in m.graph.kappa_(body.i):
            continue
        tj, s = jcalc(b.joint.joint_type, q[b.i])
        t = tj.do(b.t).do(t)
        J = J.index_set(b.i, t.inv().do(s))
    return J
"""
