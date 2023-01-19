import functools
from typing import Any, Sequence, Union

import jax.numpy as jnp
import tree_utils as tu
from flax import struct
from jax.tree_util import tree_map

from tikitree import maths, spatial

Scalar = jnp.ndarray
Vector = jnp.ndarray
Quaternion = jnp.ndarray


def to_arr(*args):
    return tuple(jnp.array(arg) for arg in args)


class _Base:
    def slice(self, beg: int, end: int) -> Any:
        return tree_map(lambda x: x[beg:end], self)

    def take(self, i, axis: int = 0) -> Any:
        return tree_map(lambda x: jnp.take(x, i, axis=axis), self)

    def concatenate(self, *others: Any, along_existing: bool = False) -> Any:
        return tu.tree_concat((self,) + others, along_existing, "jax")

    def index_set(self, idx: Union[jnp.ndarray, Sequence[jnp.ndarray]], o: Any) -> Any:
        return tree_map(lambda x, y: x.at[idx].set(y), self, o)


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

        Returns:
            r: _description_
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
        return Transform(*to_arr(pos, rot))

    @classmethod
    def zero(cls, shape=()) -> "Transform":
        """Returns a zero transform with a batch shape."""
        pos = jnp.zeros(shape + (3,))
        rot = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), shape + (1,))
        return Transform(pos, rot)

    def to_matrix(self):
        E = maths.quat_to_3x3(self.rot)
        return spatial.quadrants(aa=E, bb=E) @ spatial.xlt(self.pos)


@struct.dataclass
class Motion(_Base):
    ang: Vector
    vel: Vector

    def __add__(self, other: "Motion") -> "Motion":
        return _motion_add(other, self)

    def __sub__(self, other: "Motion") -> "Motion":
        return _motion_add(Motion(other.ang * -1.0, other.vel * -1.0), self)

    def __mul__(self, other: Scalar) -> "Motion":
        return Motion(self.ang * other, self.vel * other)

    def dot(self, other: "Force") -> Scalar:
        return jnp.sum(self.ang * other.ang) + jnp.sum(self.vel * other.vel)

    def cross(self, other):
        return _motion_cross(other, self)

    @classmethod
    def create(cls, ang=None, vel=None):
        assert not (ang is None and vel is None), "One must be given."
        if ang is None:
            ang = jnp.zeros((3,))
        if vel is None:
            vel = jnp.zeros((3,))
        return Motion(*to_arr(ang, vel))

    @classmethod
    def zero(cls, shape=()) -> "Motion":
        ang = jnp.zeros(shape + (3,))
        vel = jnp.zeros(shape + (3,))
        return Motion(ang, vel)


@struct.dataclass
class Force(_Base):
    ang: Vector
    vel: Vector

    def __add__(self, other: "Force") -> "Force":
        return _force_add(other, self)

    def __sub__(self, other: "Force") -> "Force":
        return _force_add(Force(other.ang * -1.0, other.vel * -1.0), self)

    def __mul__(self, other: Scalar) -> "Force":
        return Force(self.ang * other, self.vel * other)

    def dot(self, other: Motion) -> Scalar:
        return other.dot(self)

    @classmethod
    def create(cls, ang=None, vel=None):
        assert not (ang is None and vel is None), "One must be given."
        if ang is None:
            ang = jnp.zeros((3,))
        if vel is None:
            vel = jnp.zeros((3,))
        return Force(*to_arr(ang, vel))

    @classmethod
    def zero(cls, shape=()) -> "Force":
        ang = jnp.zeros(shape + (3,))
        vel = jnp.zeros(shape + (3,))
        return Force(ang, vel)


@struct.dataclass
class Inertia(_Base):
    it_3x3: jnp.ndarray
    CoM: Vector
    mass: Vector

    def mul(self, m: Motion) -> Force:
        cc = spatial.cross(self.CoM)
        # ang = (self.it_3x3 + self.mass * cc @ cc.T) @ m.ang + self.mass * cc @ m.vel
        ang = self.it_3x3 @ m.ang + cc @ m.vel
        vel = self.mass * m.vel - cc @ m.ang
        return Force(ang, vel)

    def __add__(self, other: "Inertia") -> "Inertia":
        return Inertia(
            self.it_3x3 + other.it_3x3, self.CoM + other.CoM, self.mass + other.mass
        )

    @classmethod
    def zero(cls, shape=()) -> "Inertia":
        it_shape_3x3 = jnp.zeros(shape + (3, 3))
        com = jnp.zeros(shape + (3,))
        mass = jnp.zeros(shape + (1,))
        return Inertia(it_shape_3x3, com, mass)


@functools.singledispatch
def _transform_do(other, self: Transform):
    raise NotImplementedError


@_transform_do.register(Transform)
def _(t: Transform, self: Transform) -> Transform:
    t2, t1 = self, t
    pos = t1.pos + maths.rotate(t2.pos, maths.quat_inv(t1.rot))
    rot = maths.quat_mul(t2.rot, t1.rot)
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
    h = it.CoM
    I_ = it.it_3x3
    m = it.mass
    rcross = spatial.cross(r)

    new_com = maths.rotate(h - m * r, self.rot)
    new_it = (
        E @ (I_ + rcross @ spatial.cross(h) + spatial.cross(h - m * r) @ rcross) @ E.T
    )
    return Inertia(new_it, new_com, it.mass)


@functools.singledispatch
def _motion_add(other, self: Motion):
    raise NotADirectoryError


@_motion_add.register(Motion)
def _(m: Motion, self: Motion) -> Motion:
    return Motion(m.ang + self.ang, m.vel + self.vel)


@functools.singledispatch
def _force_add(other, self: Force):
    raise NotADirectoryError


@_force_add.register(Force)
def _(f: Force, self: Force) -> Force:
    return Force(f.ang + self.ang, f.vel + self.vel)


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
