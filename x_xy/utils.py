import jax.numpy as jnp

from x_xy import base


def add_floating_base(sys: base.System) -> base.System:
    links = []
    for joint_type in [
        base.JointType.PrismaticX,
        base.JointType.PrismaticY,
        base.JointType.PrismaticZ,
        base.JointType.RevoluteX,
        base.JointType.RevoluteY,
        base.JointType.RevoluteZ,
    ]:
        links.append(base.Link.create(base.Transform.zero(), base.Joint(joint_type)))

    unbatched_links = [sys.links.take(i) for i in range(sys.N)]
    links = links + unbatched_links

    parent_arr = jnp.array([-1, 0, 1, 2, 3, 4])
    parent_arr = jnp.concatenate((parent_arr, sys.parent_array + 6))
    return base.System(parent_arr, links[0].batch(*links[1:]), sys.gravity)
