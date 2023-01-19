import jax
import jax.numpy as jnp

from tikitr.base import Force, Motion, Transform
from tikitr.model import Link, Model

from .jcalc import jcalc


def inverse_dynamics(m: Model, q, qd, qdd, return_intermediates: bool = False):

    N = m.graph.N
    ss = Motion.zero((N,))
    ts = Transform.zero((N,))
    vs = Motion.zero((N,))
    As = Motion.zero((N,))
    fs = Force.zero((N,))
    taus = jnp.empty((N,))

    for b in m.root_to_tip():
        tj, s = jcalc(b.joint.joint_type, q[b.i])
        ss = ss.index_set(b.i, s)
        vJ = s * qd[b.i]
        t = tj.do(b.t)
        ts.index_set(b.i, t)
        p = m.parent(b)
        if m.is_root(p):
            v = vJ
            a = t.do(Motion.create(vel=[0, 0, 9.81])) + s * qdd[b.i]
        else:
            v = t.do(vs.take(p.i)) + vJ
            a = t.do(As.take(p.i)) + s * qdd[b.i] + v.cross(vJ)

        vs = vs.index_set(b.i, v)
        As = As.index_set(b.i, a)

        f = b.geo.inertia.mul(a) + v.cross(b.geo.inertia.mul(v))
        fs = fs.index_set(b.i, f)

    for b in m.tip_to_root():
        tau = ss.take(b.i).dot(fs.take(b.i))
        taus = taus.at[b.i].set(tau)
        p = m.parent(b)
        if not m.is_root(p):
            fs = fs.index_set(p.i, fs.take(p.i) + ts.take(b.i).inv().do(fs.take(b.i)))

    if return_intermediates:
        return taus, ss, ts, vs, As, fs
    else:
        return taus


def _composite_rigid_body_algo(m: Model, q, ss, ts):

    # if we are given the intermediates `ss` and `ts`, then `q` is not needed
    del q

    N = m.graph.N
    H = jnp.zeros((N, N))

    inertias = []
    for b in m.root_to_tip():
        inertias.append(b.geo.inertia)

    Ic = inertias[0].concatenate(*inertias[1:])

    for b in m.tip_to_root():
        p = m.parent(b)
        if not m.is_root(p):
            # Figure 7: line 7 from `Beginner's guide Part 2 by Roy Featherstone`
            Ic = Ic.index_set(p.i, Ic.take(p.i) + ts.take(b.i).inv().do(Ic.take(b.i)))

        # line 9
        s = ss.take(b.i)
        f = Ic.take(b.i).mul(s)
        H = H.at[b.i, b.i].set(f.dot(s))

        j = b
        while not m.is_root(m.parent(j)):
            f = ts.take(j.i).inv().do(f)
            j = m.parent(j)
            s = ss.take(j.i)
            fdots = f.dot(s)
            H = H.at[b.i, j.i].set(fdots)
            H = H.at[j.i, b.i].set(fdots)

    return H


def forward_dynamics(m: Model, q, qd, taus):
    C, ss, ts, *_ = inverse_dynamics(m, q, qd, jnp.zeros_like(q), True)
    H = _composite_rigid_body_algo(m, q, ss, ts)

    @jax.vmap
    def spring_damper_force(link: Link, q, qd):
        lj = link.joint
        tau_spring = lj.gamma * qd + lj.k * (lj.q_setpoint_springforce - q)
        return tau_spring

    spring_damper = spring_damper_force(m.return_stacked_links(), q, qd)
    spring_damper = jnp.squeeze(spring_damper)

    qdd = jnp.linalg.solve(H, taus + spring_damper - C)

    return qdd
