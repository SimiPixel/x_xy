import time
from pathlib import Path
from typing import Optional

import imageio
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from vispy import app, scene
from vispy.scene import MatrixTransform

from x_xy import maths
from x_xy.base import Box, Geometry, GeometryCollection


@jax.jit
def _transform_4x4(pos, rot, com):
    E = maths.quat_to_3x3(rot)
    M = jnp.eye(4)
    M = M.at[:3, :3].set(E)
    pos = pos + E.T @ com
    T = jnp.eye(4)
    T = T.at[3, :3].set(pos)
    return M @ T


def transform_4x4(pos, rot, com=jnp.zeros((3,))):
    return np.asarray(_transform_4x4(pos, rot, com))


class Renderer:
    def __init__(
        self,
        geom_coll: GeometryCollection,
        show_cs=True,
        size=(1280, 720),
        camera: scene.cameras.BaseCamera = scene.TurntableCamera(
            elevation=30, distance=6
        ),
        headless: Optional[bool] = None,
        **kwargs,
    ):
        if headless:
            import vispy

            try:
                vispy.use("egl")
            except RuntimeError:
                try:
                    vispy.use("osmesa")
                except RuntimeError:
                    print("headless mode requires either `egl` or `osmesa`")
        self.headless = headless

        self.canvas = scene.SceneCanvas(
            keys="interactive", size=size, show=True, **kwargs
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = camera
        self.show_cs = show_cs
        self.geom_coll = geom_coll
        self._populate()

    def _create_visual_element(self, geom: Geometry, **kwargs):
        if isinstance(geom, Box):
            return scene.visuals.Box(
                geom.dim_x, geom.dim_z, geom.dim_y, parent=self.view.scene, **kwargs
            )
        raise NotImplementedError()

    def _populate(self):
        self._can_mutate = False
        self.visuals = []
        self._geoms_cs = []

        if self.show_cs:
            scene.visuals.XYZAxis(parent=self.view.scene)

        for geoms_per_link in self.geom_coll.geoms:

            if self.show_cs:
                self._geoms_cs.append(scene.visuals.XYZAxis(parent=self.view.scene))

            visuals_per_link = []
            for geom in geoms_per_link:
                visuals_per_link.append(
                    self._create_visual_element(geom, **geom.vispy_kwargs)
                )
            self.visuals.append(visuals_per_link)

    def update(self, data_pos: jax.Array, data_rot: jax.Array):
        self.data_pos = data_pos
        self.data_rot = data_rot
        self._update_scene()
        self._can_mutate = True

    def render(self) -> np.ndarray:
        """RGBA Array of Shape = (M, N, 4)"""
        return self.canvas.render(alpha=True)

    def _get_link_data(self, link_idx: int):
        rot = self.data_rot[link_idx]
        pos = self.data_pos[link_idx]
        return rot, pos

    def _get_transform_matrix(self, link_idx, geom_idx=None):
        rot, pos = self._get_link_data(link_idx)

        if geom_idx is not None:
            return transform_4x4(pos, rot, self.geom_coll.geoms[link_idx][geom_idx].CoM)
        else:
            return transform_4x4(pos, rot)

    def _update_scene(self):
        for link_idx in range(len(self.visuals)):
            if self.show_cs:
                transform_matrix = self._get_transform_matrix(link_idx)
                cs = self._geoms_cs[link_idx]
                if self._can_mutate:
                    cs.transform.matrix = transform_matrix
                else:
                    cs.transform = MatrixTransform(transform_matrix)

            for geom_idx in range(len(self.visuals[link_idx])):
                transform_matrix = self._get_transform_matrix(link_idx, geom_idx)
                visual = self.visuals[link_idx][geom_idx]
                if self._can_mutate:
                    visual.transform.matrix = transform_matrix
                else:
                    visual.transform = MatrixTransform(transform_matrix)


def _parse_timestep(timestep: float, fps: int, N: int):
    assert 1 / timestep > fps, "The `fps` is too high for the simulated timestep"
    fps_simu = int(1 / timestep)
    assert (fps_simu % fps) == 0, "The `fps` does not align with the timestep"
    T = N * timestep
    step = int(fps_simu / fps)
    return T, step


def _data_checks(renderer, data_pos, data_rot):
    assert (
        data_pos.ndim == data_rot.ndim == 3
    ), "Expected shape = (n_timesteps, n_links, 3/4)"
    n_links = len(renderer.geom_coll.geoms)
    assert (
        data_pos.shape[1] == data_rot.shape[1] == n_links
    ), "Number of links does not match"


def animate(
    path: Path,
    renderer: Renderer,
    data_pos: jax.Array,
    data_rot: jax.Array,
    timestep: float,
    fps: int = 50,
    format: str = "GIF",
):
    # TODO Allow for .mp4
    assert format == "GIF", "Currently the only implemented option is `GIF`"
    if not renderer.headless:
        print("Warning: Animate function expected a `Renderer(headless=True)`")

    _data_checks(renderer, data_pos, data_rot)

    N = data_pos.shape[0]
    T, step = _parse_timestep(timestep, fps, N)

    frames = []
    for t in tqdm.tqdm(range(0, N, step), "Rendering frames.."):
        renderer.update(data_pos[t], data_rot[t])
        frames.append(renderer.render())

    print(f"DONE. Converting frames to {path}.gif (might take a second)")
    imageio.mimsave(path, frames, format="GIF", fps=fps)


class Launcher:
    def __init__(
        self,
        renderer: Renderer,
        data_pos: jax.Array,
        data_rot: jax.Array,
        timestep: float,
        fps: int = 50,
    ) -> None:
        _data_checks(renderer, data_pos, data_rot)
        self._data_pos = data_pos
        self._data_rot = data_rot
        self._renderer = renderer

        self.N = data_pos.shape[0]
        self.T, self.step = _parse_timestep(timestep, fps, self.N)
        self.timestep = timestep
        self.fps = fps

    def reset(self):
        self.reached_end = False
        self.time = 0
        self.t = 0
        self.starttime = time.time()
        self._update_renderer()

    def _update_renderer(self):
        self._renderer.update(self._data_pos[self.t], self._data_rot[self.t])

    def _on_timer(self, event):
        if self.time > self.T:
            self.reached_end = True

        if self.reached_end:
            return

        self._update_renderer()

        self.t += self.step
        self.time += self.step * self.timestep
        self.realtime = time.time()
        self.current_fps = (self.time / (self.realtime - self.starttime)) * self.fps

        print("FPS: ", int(self.current_fps), f"Target FPS: {self.fps}")

    def start(self):
        self.reset()

        self._timer = app.Timer(
            1 / self.fps,
            connect=self._on_timer,
            start=True,
        )

        app.run()
