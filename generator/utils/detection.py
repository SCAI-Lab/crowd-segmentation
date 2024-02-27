from typing import Optional

import numpy as np

from dataset.utils.utils import bresenham


class Detection:
    def __init__(
        self, centre: np.ndarray, width: float, length: float, rotation: float, index: int
    ) -> None:
        self.centre = centre  # Given in local coordinate frame
        self.width = width
        self.length = length
        self.rotation = rotation
        self.velocity: Optional[np.ndarray] = None
        self.corner: Optional[np.ndarray] = None
        self.index: int = index

    def get_corner_pnts(self) -> None:
        rotation_matrix = np.array(
            [
                [np.cos(self.rotation), np.sin(self.rotation)],
                [-np.sin(self.rotation), np.cos(self.rotation)],
            ]
        )

        offset = np.array([self.length / 2, self.width / 2])

        corner = []
        for x in [1, -1]:
            for y in [-1, 1]:
                shift = np.zeros((2, 2))
                np.fill_diagonal(shift, [x, y])
                pnt = self.centre + np.matmul(rotation_matrix, np.matmul(shift, offset))
                corner.append(pnt)

        self.corner = np.array(corner)

    def inside_radius(self, max_distance: float) -> bool:
        if self.corner is None:
            self.get_corner_pnts()

        if np.linalg.norm(self.centre) <= max_distance:
            return True
        return bool(np.any(np.linalg.norm(self.corner, axis=1) <= max_distance))

    def inside_radius_global(self, max_distance: float, position: np.ndarray) -> bool:
        if self.corner is None:
            self.get_corner_pnts()

        if np.linalg.norm(self.centre - position[:2]) <= max_distance:
            return True

        return bool(np.any(np.linalg.norm(self.corner - position[:2], axis=1) <= max_distance))

    def get_occupied_indecies(self, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
        if self.corner is None:
            self.get_corner_pnts()

        x_idx = (
            np.digitize(self.corner[:, 0], x_grid, right=True) - 1
        )  # Reduce by one to receive correct index
        y_idx = (
            np.digitize(self.corner[:, 1], y_grid, right=True) - 1
        )  # Reduce by one to receive correct index

        idx = np.concatenate((x_idx, y_idx)).reshape(2, -1).T
        idx = np.clip(idx, 0, 255)

        line_1 = bresenham(idx[0], idx[1])
        line_2 = bresenham(idx[2], idx[3])
        indecies = []
        for start_pnt, end_pnt in zip(line_1, line_2):
            indecies += bresenham(start_pnt, end_pnt)

        return np.unique(np.array(indecies), axis=0)

    def transform_to_global_coordinates(
        self,
        local_transformation_matrix: np.ndarray,
        global_transformation_matrix: np.ndarray,
        rotation: float,
    ) -> None:
        expanded_pos = np.concatenate((self.centre, np.array([0, 1])))

        transformed_pos = np.matmul(
            global_transformation_matrix, np.matmul(local_transformation_matrix, expanded_pos)
        )

        self.centre = transformed_pos[0:2]
        self.rotation += rotation

        # if self.velocity is not None:
        #     expanded_vel = np.concatenate((self.centre, np.array([0, 1])))

        #     transformed_vel = np.matmul(
        #         global_transformation_matrix, np.matmul(local_transformation_matrix, expanded_vel)
        #     )

        #     self.velocity = transformed_vel[0:2]

    def is_cell_occupied(
        self, cell_idx: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray
    ) -> bool:
        occupied_cells = self.get_occupied_indecies(x_grid, y_grid)
        return bool(np.any(np.all(occupied_cells == cell_idx, axis=1)))
