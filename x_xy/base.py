import functools
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import tree_utils as tu
from flax import struct
from jax.tree_util import tree_map

from x_xy import maths, spatial

Scalar = jax.Array
Vector = jax.Array
Quaternion = jax.Array


class _Base:
    """Base functionality of all spatial datatypes.
    Copied and modified from https://github.com/google/brax/blob/main/brax/v2/base.py
    """

    def __add__(self, o: Any) -> Any:
        return tree_map(lambda x, y: x + y, self, o)

    def __sub__(self, o: Any) -> Any:
        return tree_map(lambda x, y: x - y, self, o)

    def __mul__(self, o: Any) -> Any:
        return tree_map(lambda x: x * o, self)

    def __neg__(self) -> Any:
        return tree_map(lambda x: -x, self)

    def __truediv__(self, o: Any) -> Any:
        return tree_map(lambda x: x / o, self)

    def reshape(self, shape: Sequence[int]) -> Any:
        return tree_map(lambda x: x.reshape(shape), self)

    def select(self, o: Any, cond: jnp.ndarray) -> Any:
        return tree_map(lambda x, y: (x.T * cond + y.T * (1 - cond)).T, self, o)

    def slice(self, beg: int, end: int) -> Any:
        return tree_map(lambda x: x[beg:end], self)

    def take(self, i, axis=0) -> Any:
        return tree_map(lambda x: jnp.take(x, i, axis=axis, mode="fill"), self)

    def hstack(self, *others: Any) -> Any:
        return tree_map(lambda *x: jnp.hstack(x), self, *others)

    def vstack(self, *others: Any) -> Any:
        return tree_map(lambda *x: jnp.vstack(x), self, *others)

    def concatenate(self, *others: Any, axis: int = 0) -> Any:
        return tree_map(lambda *x: jnp.concatenate(x, axis=axis), self, *others)

    def concat(self, *others: Any, along_existing_first_axis: bool = False) -> Any:
        raise Exception("Use .batch instead")
        return tu.tree_concat((self,) + others, along_existing_first_axis, "jax")

    def batch(self, *others, along_existing_first_axis: bool = False) -> Any:
        return tu.tree_batch((self,) + others, along_existing_first_axis, "jax")

    def index_set(self, idx: Union[jnp.ndarray, Sequence[jnp.ndarray]], o: Any) -> Any:
        return tree_map(lambda x, y: x.at[idx].set(y), self, o)

    def index_sum(self, idx: Union[jnp.ndarray, Sequence[jnp.ndarray]], o: Any) -> Any:
        return tree_map(lambda x, y: x.at[idx].add(y), self, o)

    def vmap(self, in_axes=0, out_axes=0):
        """Returns an object that vmaps each follow-on instance method call."""

        outer_self = self

        class VmapField:
            """Returns instance method calls as vmapped."""

            def __init__(self, in_axes, out_axes):
                self.in_axes = [in_axes]
                self.out_axes = [out_axes]

            def vmap(self, in_axes=0, out_axes=0):
                self.in_axes.append(in_axes)
                self.out_axes.append(out_axes)
                return self

            def __getattr__(self, attr):
                fun = getattr(outer_self.__class__, attr)
                # load the stack from the bottom up
                vmap_order = reversed(list(zip(self.in_axes, self.out_axes)))
                for in_axes, out_axes in vmap_order:
                    fun = jax.vmap(fun, in_axes=in_axes, out_axes=out_axes)
                fun = functools.partial(fun, outer_self)
                return fun

        return VmapField(in_axes, out_axes)

    @property
    def T(self):
        return tree_map(lambda x: x.T, self)

    def flatten(self, num_batch_dims: int = 0) -> jax.Array:
        return tu.batch_concat(self, num_batch_dims)

    def squeeze(self):
        return tree_map(lambda x: jnp.squeeze(x), self)

    def squeeze_1d(self):
        return tree_map(lambda x: jnp.atleast_1d(jnp.squeeze(x)), self)


@struct.dataclass
class Transform(_Base):
    pos: Vector
    rot: Quaternion

    def do(self, other):
        return _transform_do(other, self)

    def inv(self) -> "Transform":
        """Inverts the transform.

        Prove:

        Assume that this represents the inverse transform
        r -> E @ (-r)
        E -> E^T

        Then,
        Starting from RoyBook (2.24) one gets
        -> -E^T(-E @ r) x
        -> E^T(E (r x) E^T) (using identity of Table 2.1 4th row)
        -> r x E^T
        which corresponds to (2.27) and represents the inverse.

        q.e.d.
        """
        pos = maths.rotate(-self.pos, self.rot)
        return Transform(pos, maths.quat_inv(self.rot))

    @classmethod
    def create(cls, pos=None, rot=None):
        assert not (pos is None and rot is None), "One must be given."
        if pos is None:
            pos = jnp.zeros((3,))
        if rot is None:
            rot = jnp.array([1.0, 0, 0, 0])
        return Transform(pos, rot)

    @classmethod
    def zero(cls, shape=()) -> "Transform":
        """Returns a zero transform with a batch shape."""
        pos = jnp.zeros(shape + (3,))
        rot = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), shape + (1,))
        return Transform(pos, rot)

    def as_matrix(self) -> jax.Array:
        E = maths.quat_to_3x3(self.rot)
        return spatial.quadrants(aa=E, bb=E) @ spatial.xlt(self.pos)


@struct.dataclass
class Motion(_Base):
    ang: Vector
    vel: Vector

    def dot(self, other: "Force") -> Scalar:
        return self.ang @ other.ang + self.vel @ other.vel

    def cross(self, other):
        return _motion_cross(other, self)

    @classmethod
    def create(cls, ang=None, vel=None):
        assert not (ang is None and vel is None), "One must be given."
        if ang is None:
            ang = jnp.zeros((3,))
        if vel is None:
            vel = jnp.zeros((3,))
        return Motion(ang, vel)

    @classmethod
    def zero(cls, shape=()) -> "Motion":
        ang = jnp.zeros(shape + (3,))
        vel = jnp.zeros(shape + (3,))
        return Motion(ang, vel)

    def as_matrix(self):
        return self.flatten()


@struct.dataclass
class Force(_Base):
    ang: Vector
    vel: Vector

    def dot(self, other: Motion) -> Scalar:
        return other.dot(self)

    @classmethod
    def create(cls, ang=None, vel=None):
        assert not (ang is None and vel is None), "One must be given."
        if ang is None:
            ang = jnp.zeros((3,))
        if vel is None:
            vel = jnp.zeros((3,))
        return Force(ang, vel)

    @classmethod
    def zero(cls, shape=()) -> "Force":
        ang = jnp.zeros(shape + (3,))
        vel = jnp.zeros(shape + (3,))
        return Force(ang, vel)

    def as_matrix(self):
        return self.flatten()


@struct.dataclass
class Inertia(_Base):
    it_3x3: jnp.ndarray
    h: Vector
    mass: Vector

    def mul(self, m: Motion) -> Force:
        cc = spatial.cross(self.h)
        ang = self.it_3x3 @ m.ang + cc @ m.vel
        vel = self.mass * m.vel - cc @ m.ang
        return Force(ang, vel)

    def __add__(self, other: "Inertia") -> "Inertia":
        return Inertia(
            self.it_3x3 + other.it_3x3, self.h + other.h, self.mass + other.mass
        )

    @classmethod
    def zero(cls, shape=()) -> "Inertia":
        it_shape_3x3 = jnp.zeros(shape + (3, 3))
        h = jnp.zeros(shape + (3,))
        mass = jnp.zeros(shape + (1,))
        return Inertia(it_shape_3x3, h, mass)

    def as_matrix(self):
        hcross = spatial.cross(self.h)
        return spatial.quadrants(self.it_3x3, hcross, -hcross, self.mass * jnp.eye(3))

    @classmethod
    def create(cls, mass: Vector, CoM: Vector, it_3x3: jnp.ndarray):
        it_3x3 = spatial.mcI(mass, CoM, it_3x3)[:3, :3]
        h = mass * CoM
        return Inertia(it_3x3, h, mass)

    @classmethod
    def create_from_geometry(cls, geom: "Geometry"):
        it_3x3 = geom.get_it_3x3()
        return Inertia.create(geom.mass, geom.CoM, it_3x3)


@dataclass
class Geometry(_Base):
    mass: jax.Array
    CoM: jax.Array


@dataclass
class Sphere(Geometry):
    radius: jax.Array
    vispy_kwargs: dict = field(default_factory=lambda: {})

    def get_it_3x3(self) -> jax.Array:
        it_3x3 = 2 / 5 * self.mass * self.radius**2 * jnp.eye(3)
        return it_3x3


@dataclass
class Box(Geometry):
    dim_x: jax.Array
    dim_y: jax.Array
    dim_z: jax.Array
    vispy_kwargs: dict = field(default_factory=lambda: {})

    def get_it_3x3(self) -> jax.Array:
        it_3x3 = (
            1
            / 12
            * self.mass
            * jnp.array(
                [
                    [self.dim_y**2 + self.dim_z**2, 0, 0],
                    [0, self.dim_x**2 + self.dim_z**2, 0],
                    [0, 0, self.dim_x**2 + self.dim_y**2],
                ]
            )
        )
        return it_3x3


@dataclass
class Cylinder(Geometry):
    """Length is along x-axis."""

    radius: jax.Array
    length: jax.Array
    vispy_kwargs: dict = field(default_factory=lambda: {})

    def get_it_3x3(self) -> jax.Array:
        radius_dir = 3 * self.radius**2 + self.length**2
        it_3x3 = (
            1
            / 12
            * self.mass
            * jnp.array(
                [
                    [6 * self.radius**2, 0, 0],
                    [0, radius_dir, 0],
                    [0, 0, radius_dir],
                ]
            )
        )
        return it_3x3


@dataclass
class GeometryCollection:
    """Dataclass that stores all geometry information for rendering with VisPy.
    Attr:
      - geoms: The list of geometries associated with every link.
    """

    geoms: list[list[Geometry]] = field(default_factory=lambda: [[]])


class JointType(IntEnum):
    RevoluteX = 0
    RevoluteY = 1
    RevoluteZ = 2
    PrismaticX = 3
    PrismaticY = 4
    PrismaticZ = 5


@struct.dataclass
class Joint(_Base):
    joint_type: JointType
    stiffness: jax.Array = jnp.array(0.0)
    damping: jax.Array = jnp.array(0.0)
    armature: jax.Array = jnp.array(0.0)
    zero_position: jax.Array = jnp.array(0.0)
    motion: Motion = Motion.zero()


@struct.dataclass
class Link(_Base):
    """
    Attributes:
        transform: transform from parent to link frame
        Xtree: transform from parent to joint frame
        inertia: ...
    """

    Xtree: Transform
    joint: Joint
    inertia: Inertia
    transform: Transform = Transform.zero()

    @classmethod
    def create(
        cls,
        Xtree: Transform,
        joint: Joint,
        geoms: Optional[Union[Geometry, list[Geometry]]] = None,
    ):
        if geoms is not None:
            if not isinstance(geoms, list):
                geoms = [geoms]

            inertia = Inertia.zero()
            for geom in geoms:
                inertia = inertia + Inertia.create_from_geometry(geom)
        else:
            inertia = Inertia.zero()

        return Link(Xtree, joint, inertia)


@struct.dataclass
class System(_Base):
    parent_array: jax.Array
    links: Link
    gravity: jax.Array = jnp.array([0, 0, 9.81])

    @property
    def parent(self) -> jax.Array:
        return self.parent_array

    @property
    def N(self):
        return len(self.parent_array)


@struct.dataclass
class State(_Base):
    q: jax.Array
    qd: jax.Array
    x: jax.Array
    xd: jax.Array


@functools.singledispatch
def _transform_do(other, self: Transform):
    raise NotImplementedError


@_transform_do.register(Transform)
def _(t: Transform, self: Transform) -> Transform:
    t1, t2 = self, t
    pos = t2.pos + maths.rotate(t1.pos, maths.quat_inv(t2.rot))
    rot = maths.quat_mul(t1.rot, t2.rot)
    return Transform(pos, rot)


@_transform_do.register(Motion)
def _(m: Motion, self: Transform) -> Motion:
    ang = maths.rotate(m.ang, self.rot)
    vel = maths.rotate(-jnp.cross(self.pos, m.ang) + m.vel, self.rot)
    return Motion(ang, vel)


@_transform_do.register(Force)
def _(f: Force, self: Transform) -> Force:
    ang = maths.rotate(f.ang - jnp.cross(self.pos, f.vel), self.rot)
    vel = maths.rotate(f.vel, self.rot)
    return Force(ang, vel)


@_transform_do.register(Inertia)
def _(it: Inertia, self: Transform) -> Inertia:
    r = self.pos
    E = maths.quat_to_3x3(self.rot)
    h = it.h
    I_ = it.it_3x3
    m = it.mass
    rcross = spatial.cross(r)

    new_com = maths.rotate(h - m * r, self.rot)
    new_it = (
        E @ (I_ + rcross @ spatial.cross(h) + spatial.cross(h - m * r) @ rcross) @ E.T
    )
    return Inertia(new_it, new_com, it.mass)


@functools.singledispatch
def _motion_cross(other, self: Motion):
    raise NotImplementedError


@_motion_cross.register(Motion)
def _(m: Motion, self: Motion) -> Motion:
    ang = jnp.cross(self.ang, m.ang)
    vel = jnp.cross(self.vel, m.ang) + jnp.cross(self.ang, m.vel)
    return Motion(ang, vel)


@_motion_cross.register(Force)
def _(f: Force, self: Motion) -> Force:
    ang = jnp.cross(self.ang, f.ang) + jnp.cross(self.vel, f.vel)
    vel = jnp.cross(self.ang, f.vel)
    return Force(ang, vel)
