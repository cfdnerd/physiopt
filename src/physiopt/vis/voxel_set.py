import polyscope as ps
import polyscope.imgui as psim
import numpy as np
from enum import Enum
from typing import Tuple

from ps_utils.structures import (
    CUBE_VERTICES_NP,
    CUBE_TRIANGLES_NP,
)
from ps_utils.ui import get_enum_maps, KEY_HANDLER

import torch


class BrushMode(Enum):
    ADD = "add"
    REMOVE = "remove"


BRUSH_MODE_MAP, BRUSH_MODE_INVMAP, BRUSH_MODE_NAMES, _ = get_enum_maps(BrushMode)


DEFAULT_HOVER_ADD_COLOR = np.array([0.0, 1.0, 0.0])
DEFAULT_HOVER_ERASE_COLOR = np.array([1.0, 0.0, 0.0])
DEFAULT_SELECTED_COLOR = np.array([0.0, 0.0, 1.0])
DEFAULT_BASE_COLOR = np.array([1.0, 1.0, 1.0])

DEFAULT_SELECTION_RADIUS = 4.0
SELECTION_RADIUS_SENTIVITY = 0.5
MIN_SELECTION_RADIUS = 0.001
MAX_SELECTION_RADIUS = 32.0
DEFAULT_BRUSH_MODE = BrushMode.ADD


# TODO: remove because this is a duplicate
@torch.no_grad()
def create_voxel_set_np(
    coords: np.ndarray,
    voxel_res: int,
    bbox_min: float,
    bbox_max: float,
    name: str = "voxel_set",
    offset: np.ndarray | None = None,
    enabled: bool = True,
) -> Tuple[ps.SurfaceMesh, np.ndarray, np.ndarray]:
    # self.voxels = coord_bbox_filter(self.voxels, self.res)

    vertex_offsets = np.repeat(coords, 8, axis=0)
    cube_vertices = (
        np.tile(CUBE_VERTICES_NP, (len(coords), 1)) - 0.5
    )  # Rescale voxels (each center with point cloud)
    vertices = cube_vertices + vertex_offsets
    vertices = (bbox_max - bbox_min) * (1.0 / float(voxel_res)) * vertices + bbox_min

    if offset is not None:
        vertices += offset

    # 8 for 8 vertices
    triangles_offsets = np.repeat(
        np.tile((8 * np.arange(len(coords)))[:, None], ((1, 3))),
        len(CUBE_TRIANGLES_NP),
        axis=0,
    )
    faces = np.tile(CUBE_TRIANGLES_NP, (len(coords), 1)) + triangles_offsets

    ps_voxels = ps.register_surface_mesh(
        name,
        vertices,
        faces,
        enabled=enabled,
    )
    ps_voxels.set_edge_width(0.0)

    return ps_voxels, vertices, faces


class VoxelSet:

    def __init__(
        self,
        coords: np.ndarray,
        voxel_res: int,
        bbox_min: float,
        bbox_max: float,
        name: str = "voxel_set",
        offset: np.ndarray | None = None,
        selection_mask: np.ndarray | None = None,
    ):

        self.square_brush = True
        self.selection_radius = DEFAULT_SELECTION_RADIUS
        self.brush_mode = DEFAULT_BRUSH_MODE
        self.square_brush = True
        self.last_selected_voxel_id = -1
        self.selection_changed = False
        self.name = name

        self.coords = coords
        self.selection_mask = (
            selection_mask
            if selection_mask is not None
            else np.zeros(len(coords), dtype=bool)
        )

        self.ps_voxels, self.vertices, self.faces = create_voxel_set_np(
            coords=coords,
            voxel_res=voxel_res,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            name=name,
            offset=offset,
        )

        self.ps_voxels.add_color_quantity(
            name + "selection",
            np.tile(DEFAULT_BASE_COLOR[None, :], (len(self.faces), 1)),
            defined_on="faces",
            enabled=True,
        )

        self.selection_buffer = self.ps_voxels.get_quantity_buffer(
            name + "selection", "colors"
        )

        self.update_selection_buffer()

        self.ps_voxels.set_hover_callback(self.hover_callback)

    def update_selection_buffer(self, within_radius: np.ndarray | None = None):
        current_selection = np.tile(DEFAULT_BASE_COLOR[None, :], (len(self.coords), 1))
        current_selection[self.selection_mask] = DEFAULT_SELECTED_COLOR

        if within_radius is not None:
            current_selection[within_radius] = (
                DEFAULT_HOVER_ADD_COLOR
                if self.brush_mode == BrushMode.ADD
                else DEFAULT_HOVER_ERASE_COLOR
            )

        current_selection = np.repeat(current_selection, 12, axis=0)

        self.selection_buffer.update_data_from_host(current_selection)

    def hover_callback(self, mesh_element: ps.MeshElement, index: int):
        if mesh_element == ps.MeshElement.VERTEX.value:
            voxel_id = index // 8
        elif mesh_element == ps.MeshElement.FACE.value:
            voxel_id = index // 12
        else:
            return

        hovered_voxel = self.coords[voxel_id]

        if self.square_brush:
            within_radius = (np.abs(hovered_voxel[None, :] - self.coords)).max(
                1
            ) < self.selection_radius
        else:
            within_radius = ((hovered_voxel[None, :] - self.coords) ** 2).sum(
                1
            ) < self.selection_radius**2

        if (
            psim.IsMouseClicked(0)
            and psim.GetIO().KeyAlt
            and voxel_id != self.last_selected_voxel_id
        ):
            if self.brush_mode == BrushMode.REMOVE:
                self.selection_mask &= ~within_radius
            else:
                self.selection_mask |= within_radius

            self.last_selected_voxel_id = voxel_id
            self.selection_changed = True

        self.update_selection_buffer(within_radius)

    def gui(self) -> bool:
        update = False

        psim.SeparatorText(f"{self.name}##voxel_set")

        io = psim.GetIO()

        # Use wheel to increase/decrease brush radius
        if io.MouseWheel != 0 and io.KeyAlt:
            self.selection_radius += SELECTION_RADIUS_SENTIVITY * float(io.MouseWheel)
        self.selection_radius = max(
            MIN_SELECTION_RADIUS,
            min(self.selection_radius, MAX_SELECTION_RADIUS),
        )

        # Switch selection
        if KEY_HANDLER("s"):
            self.brush_mode = BRUSH_MODE_INVMAP[1 - BRUSH_MODE_MAP[self.brush_mode]]

        psim.Text(f"Radius: {self.selection_radius}")

        psim.SameLine()
        if psim.Button(f"Reset##voxel_set_{self.name}"):
            self.selection_mask = np.zeros_like(self.selection_mask)
            self.update_selection_buffer()
            update |= True

        psim.SameLine()
        if psim.Button(f"Invert##voxel_set_{self.name}"):
            self.selection_mask = ~self.selection_mask
            self.update_selection_buffer()
            update |= True

        # Notify parent if something changed
        update |= self.selection_changed
        self.selection_changed = False
        return update

    def step(self) -> bool:
        self.update_selection_buffer()

    def set_enabled(self, val: bool = True):
        self.ps_voxels.set_enabled(val)
