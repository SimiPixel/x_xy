from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrand

################
# Safe Functions


@partial(jnp.vectorize, signature="(k)->(1)")
def safe_norm(x):
    """Grad-safe for x=0.0. Norm along last axis."""
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
    """Execution- and Grad-safe for x=0.0. Normalizes along last axis."""
    assert x.ndim == 1

    is_zero = jnp.allclose(x, 0.0)
    return jax.lax.cond(
        is_zero,
        lambda x: jnp.zeros_like(x),
        lambda x: x / jnp.where(is_zero, 1.0, safe_norm(x)),
        x,
    )


##########################
# Small Quaternion Library

# APPROVED
# it does not matter if you use q1_inv x q2 OR q1 x q2_inv
def quat_angle_error(q, qhat):
    return jnp.abs(quat_angle(quat_mul(quat_inv(q), qhat)))


# APPROVED
# Engineering Log of 10.02.23 reveals that
# this is a useless operation
# delete it
# TODO
@partial(jnp.vectorize, signature="(4)->(4)")
def quat_positive_w(q):
    return q
    return jax.lax.cond(q[0] < 0.0, lambda q: -q, lambda q: q, q)


def quat_unit_quats_like(array):
    if array.shape[-1] != 4:
        raise Exception()

    return jnp.ones(array.shape[:-1])[..., None] * jnp.array([1.0, 0, 0, 0])


# APPROVED
def quat_wrap_to_pi(phi):
    return (phi + jnp.pi) % (2 * jnp.pi) - jnp.pi


# APPROVED
@partial(jnp.vectorize, signature="(4),(4)->(4)")
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
    # EL 20.02.23 not normalizing changes the result when
    # training the network. However, it becomes better?
    # TODO
    return safe_normalize(q)


# APPROVED
def quat_inv(q: jnp.ndarray) -> jnp.ndarray:
    """Calculates the inverse of quaternion q.
    Args:
      q: (4,) quaternion [w, x, y, z]
    Returns:
      The inverse of q, where qmult(q, inv_quat(q)) = [1, 0, 0, 0].
    """
    return q * jnp.array([1.0, -1, -1, -1])


# APPROVED
@partial(jnp.vectorize, signature="(3),(4)->(3)")
def rotate(vec: jnp.ndarray, quat: jnp.ndarray):
    """Rotates a vector vec by a unit quaternion quat.
    Args:
      vec: (3,) a vector
      quat: (4,) a quaternion
    Returns:
      ndarray(3) containing vec rotated by quat.
    """
    return quat_mul(quat, quat_mul(jnp.array([0, *vec]), quat_inv(quat)))[
        1:4
    ] * safe_norm(vec)


# APPROVED
@partial(jnp.vectorize, signature="(3),(4)->(3)")
def safe_rotate(vec: jax.Array, quat: jax.Array):
    """This is the function my RNNO library originally used."""
    raise Exception("try `rotate` first; Only use if really needed.")
    is_zero = jnp.allclose(vec, 0.0)
    # https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
    return jnp.where(is_zero, vec, jnp.where(is_zero, vec, rotate(vec, quat)))


def rotate_matrix(mat: jax.Array, quat: jax.Array):
    E = quat_to_3x3(quat)
    return E @ mat @ E.T


# APPROVED
@partial(jnp.vectorize, signature="(3),()->(4)")
def quat_rot_axis(axis: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    """Provides a quaternion that describes rotating around axis v by angle.
    Args:
      axis: (3,) axis (x,y,z)
      angle: () float angle to rotate by
    Returns:
      A quaternion that rotates around v by angle
    """
    axis = safe_normalize(axis)
    qx = axis[0] * jnp.sin(angle / 2)
    qy = axis[1] * jnp.sin(angle / 2)
    qz = axis[2] * jnp.sin(angle / 2)
    qw = jnp.cos(angle / 2)
    return jnp.array([qw, qx, qy, qz])


# APPROVED
@partial(jnp.vectorize, signature="(3,3)->(4)")
def quat_from_3x3(m: jnp.ndarray) -> jnp.ndarray:
    """Converts 3x3 rotation matrix to quaternion."""
    w = jnp.sqrt(1 + m[0, 0] + m[1, 1] + m[2, 2]) / 2.0
    x = (m[2][1] - m[1][2]) / (w * 4)
    y = (m[0][2] - m[2][0]) / (w * 4)
    z = (m[1][0] - m[0][1]) / (w * 4)
    return jnp.array([w, x, y, z])


# APPROVED
@partial(jnp.vectorize, signature="(4)->(3,3)")
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


# APPROVED
def quat_random(key: jrand.PRNGKey, batch_shape: tuple[int] = ()) -> jax.Array:
    """Provides a random quaternion, sampled uniformly"""
    shape = batch_shape + (4,)
    return safe_normalize(jrand.normal(key, shape))


# APPROVED
@partial(jnp.vectorize, signature="(3)->(4)", excluded=(1, 2))
def quat_euler(angles, intrinsic=True, convention="xyz"):
    xunit = jnp.array([1.0, 0.0, 0.0])
    yunit = jnp.array([0.0, 1.0, 0.0])
    zunit = jnp.array([0.0, 0.0, 1.0])

    axes_map = {
        "x": xunit,
        "y": yunit,
        "z": zunit,
    }

    q1 = quat_rot_axis(axes_map[convention[0]], angles[0])
    q2 = quat_rot_axis(axes_map[convention[1]], angles[1])
    q3 = quat_rot_axis(axes_map[convention[2]], angles[2])

    if intrinsic:
        return quat_mul(q1, quat_mul(q2, q3))
    else:
        return quat_mul(q3, quat_mul(q2, q1))


# APPROVED
@partial(jnp.vectorize, signature="(4)->()")
def quat_angle(q):
    phi = 2 * jnp.arctan2(safe_norm(q[1:])[0], q[0])
    return quat_wrap_to_pi(phi)


# APPROVED
@partial(jnp.vectorize, signature="(4)->(3),()")
def quat_to_rot_axis(q):
    angle = quat_angle(q)
    axis = safe_normalize(q[1:])
    return axis, angle


###################
# Matrix Operations


def inv_approximate(
    a: jnp.ndarray, a_inv: jnp.ndarray, tol: float = 1e-12, maxiter: int = 10
) -> jnp.ndarray:
    """Use Newton-Schulz iteration to solve ``A^-1``.
    Args:
        a: 2D array to invert
        a_inv: approximate solution to A^-1
        tol: tolerance for convergance, ``norm(residual) <= tol``.
        maxiter: maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    Returns:
        A^-1 inverted matrix
    """

    def cond_fn(value):
        # TODO: test whether it's better for convergence to check
        # ||I - Xn @ A || > tol - this is certainly faster and results seem OK
        _, k, err = value
        return (err > tol) & (k < maxiter)

    def body_fn(value):
        a_inv, k, _ = value
        a_inv_new = 2 * a_inv - a_inv @ a.T @ a_inv
        return a_inv_new, k + 1, jnp.linalg.norm(a_inv_new - a_inv)

    # ensure ||I - X0 @ A|| < 1, in order to guarantee convergence
    r0 = jnp.eye(a.shape[0]) - a @ a_inv
    a_inv = jnp.where(jnp.linalg.norm(r0) > 1, 0.5 * a.T / jnp.trace(a @ a.T), a_inv)

    a_inv, *_ = jax.lax.while_loop(cond_fn, body_fn, (a_inv, 0, 1.0))

    return a_inv
