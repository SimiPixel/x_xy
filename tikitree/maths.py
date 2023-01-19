# many quaternion operations, also safe operations
# see brax for inspiration
from functools import partial

import jax
import jax.numpy as jnp


@partial(jnp.vectorize, signature="(k)->(1)")
def safe_norm(x):
    """Grad-safe for x=0.0"""
    assert x.ndim == 1

    is_zero = jnp.all(jnp.isclose(x, 0.0), axis=-1, keepdims=False)
    return jax.lax.cond(
        is_zero,
        lambda x: jnp.array([0.0], dtype=x.dtype),
        lambda x: jnp.linalg.norm(x, keepdims=True),
        x,
    )


@partial(jnp.vectorize, signature="(k)->(k)")
def safe_normalize(x):
    """Execution- and Grad-safe for x=0.0"""
    assert x.ndim == 1

    is_zero = jnp.allclose(x, 0.0)
    return jax.lax.cond(
        is_zero,
        lambda x: jnp.zeros_like(x),
        lambda x: x / jnp.where(is_zero, 1.0, safe_norm(x)),
        x,
    )


def quat_mul(u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Multiplies two quaternions.
    Args:
      u: (4,) quaternion (w,x,y,z)
      v: (4,) quaternion (w,x,y,z)
    Returns:
      A quaternion u * v.
    """
    q = jnp.array(
        [
            u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
            u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
            u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
            u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
        ]
    )
    return q


def quat_inv(q: jnp.ndarray) -> jnp.ndarray:
    """Calculates the inverse of quaternion q.
    Args:
      q: (4,) quaternion [w, x, y, z]
    Returns:
      The inverse of q, where qmult(q, inv_quat(q)) = [1, 0, 0, 0].
    """
    return q * jnp.array([1, -1, -1, -1])


def rotate(vec: jnp.ndarray, quat: jnp.ndarray):
    """Rotates a vector vec by a unit quaternion quat.
    Args:
      vec: (3,) a vector
      quat: (4,) a quaternion
    Returns:
      ndarray(3) containing vec rotated by quat.
    """
    return quat_mul(quat, quat_mul(jnp.array([0, *vec]), quat_inv(quat)))[1:4]


def quat_rot_axis(axis: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    """Provides a quaternion that describes rotating around axis v by angle.
    Args:
      axis: (3,) axis (x,y,z)
      angle: () float angle to rotate by
    Returns:
      A quaternion that rotates around v by angle
    """
    qx = axis[0] * jnp.sin(angle / 2)
    qy = axis[1] * jnp.sin(angle / 2)
    qz = axis[2] * jnp.sin(angle / 2)
    qw = jnp.cos(angle / 2)
    return jnp.array([qw, qx, qy, qz])


def quat_from_3x3(m: jnp.ndarray) -> jnp.ndarray:
    """Converts 3x3 rotation matrix to quaternion."""
    w = jnp.sqrt(1 + m[0, 0] + m[1, 1] + m[2, 2]) / 2.0
    x = (m[2][1] - m[1][2]) / (w * 4)
    y = (m[0][2] - m[2][0]) / (w * 4)
    z = (m[1][0] - m[0][1]) / (w * 4)
    return jnp.array([w, x, y, z])


def quat_to_3x3(q: jnp.ndarray) -> jnp.ndarray:
    """Converts quaternion to 3x3 rotation matrix."""
    d = jnp.dot(q, q)
    w, x, y, z = q
    s = 2 / d
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs

    return jnp.array(
        [
            jnp.array([1 - (yy + zz), xy - wz, xz + wy]),
            jnp.array([xy + wz, 1 - (xx + zz), yz - wx]),
            jnp.array([xz - wy, yz + wx, 1 - (xx + yy)]),
        ]
    )
