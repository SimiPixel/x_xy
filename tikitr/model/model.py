from types import SimpleNamespace
from typing import Iterable

import jax.numpy as jnp
from flax import struct

from tikitr.base import Inertia, Transform, _Base
from tikitr.model.graph import ROOT, Graph, is_root

JointType = SimpleNamespace(
    Revolute=jnp.array(0), Prismatic=jnp.array(1), Frozen=jnp.array(2)
)


@struct.dataclass
class Joint:
    joint_type: jnp.ndarray
    q_setpoint_springforce: jnp.ndarray = jnp.array([0.0])
    k: jnp.ndarray = jnp.array([0.0])
    gamma: jnp.ndarray = jnp.array([0.0])


@struct.dataclass
class Geometry(_Base):
    inertia: Inertia


@struct.dataclass
class Rectangle(Geometry):
    # TODO we do not have to store dimensions
    dim_x: jnp.ndarray
    dim_y: jnp.ndarray
    dim_z: jnp.ndarray


def create_rectangular_geometry(
    dim_x: float, dim_y: float, dim_z: float, mass: float
) -> Geometry:
    com = jnp.array([dim_x, dim_y, dim_z]) / 2
    inertia = (
        1
        / 12
        * mass
        * jnp.array(
            [
                [dim_y**2 + dim_z**2, 0, 0],
                [0, dim_x**2 + dim_z**2, 0],
                [0, 0, dim_x**2 + dim_y**2],
            ]
        )
    )
    mass, dim_x, dim_y, dim_z = map(jnp.atleast_1d, (mass, dim_x, dim_y, dim_z))
    return Rectangle(Inertia(inertia, com, mass), dim_x, dim_y, dim_z)


def create_massless_geometry():
    return create_rectangular_geometry(0.0, 0.0, 0.0, 0.0)


@struct.dataclass
class Link(_Base):
    geo: Geometry
    t: Transform
    joint: Joint
    number: int = struct.field(True)

    @classmethod
    def create_massless(cls, number: int, t: Transform, joint: Joint) -> "Link":
        return cls(create_massless_geometry(), t, joint, number)

    @classmethod
    def create_massless_and_frozen(cls, number: int, t: Transform) -> "Link":
        """Creates a body which has no mass and no joint.
        It simply creates a new frame relative to its parent frame using Transform t."""
        return cls.create_massless(number, t, Joint(JointType.Frozen))

    @property
    def i(self):
        return self.number


@struct.dataclass
class _RootLink:
    """Dummy class."""

    number: int = ROOT


@struct.dataclass
class Model:
    bodies: set[Link]
    bodies_sorted: list[Link]
    graph: Graph = struct.field(False)

    @classmethod
    def create(cls, graph: Graph, bodies: set[Link]):
        bodies_sorted = sorted(bodies, key=lambda body: body.number)
        return cls(bodies, bodies_sorted, graph)

    def _get_body(self, i: int) -> Link:
        return self.bodies_sorted[i]

    def parent(self, body: Link) -> Link:
        parent = self.graph.parent(body.number)
        if is_root(parent):
            return _RootLink()
        else:
            return self._get_body(parent)

    def is_root(self, body: Link) -> bool:
        return is_root(body.number)

    def direct_to_root(self, start: Link, include_root: bool = False):
        pass

    def root_to_tip(self) -> Iterable[Link]:
        for i in range(self.graph.N):
            yield self._get_body(i)

    def tip_to_root(self) -> Iterable[Link]:
        for i in range(self.graph.N - 1, -1, -1):
            yield self._get_body(i)

    def return_stacked_links(self):
        return self.bodies_sorted[0].concatenate(*self.bodies_sorted[1:])
