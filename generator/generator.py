import argparse
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from scipy.stats import norm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logging

from dataset.scene_class import SceneClassColour
from generator.utils.utilities import (
    cartesian_to_polar,
    euler_to_rot,
    polar_to_cartesian,
    quat_to_euler,
)
from generator.utils.detection import Detection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class Generator:
    def __init__(self, opts: argparse.Namespace) -> None:
        self.dataformat = opts.dataformat
        self.input_path = f"{opts.input}/labels" if self.dataformat == "JRDB" else opts.input
        self.output_path = opts.output
        self.gen_label = opts.label

        local_path = os.path.dirname(os.path.abspath(__file__))
        self.config = f"{local_path}/../config/labels.yml"
        file = open(self.config, "r")
        cfg = yaml.safe_load(file)

        self.timestep = cfg["timestep"]
        self.height = cfg["num_pixels"]
        self.width = cfg["num_pixels"]
        self.max_distance = cfg["max_distance"]
        self.time_interval = cfg["time_interval"]

        self.decay_type = cfg["decay_type"]
        self.decay = cfg["decay"]
        self.max_weight = cfg["max_weight"]
        self.input_occupancy_type = cfg["input_occupancy_type"]
        self.occupancy_variance = cfg["occupancy_variance"]
        # self.weight_decrease = self.max_weight * self.timestep / self.time_interval

        self.velocity_threshold = cfg["velocity_threshold"]
        self.angle_threshold = cfg["angle_threshold"] * np.pi / 180
        self.labelling_occupancy_type = cfg["labelling_occupancy_type"]
        self.search_radius = cfg["search_radius"]
        self.detection_threshold = cfg["detection_threshold"]

        self.origin = np.array([0, 0], dtype=int)
        self.origin_changed: bool = False
        self.curr_origin = np.array([0, 0], dtype=int)
        self.full_img_shift = np.array([0, 0], dtype=int)
        self.origin_coordinates = np.array([-7.0, -7.0])
        self.distance_between_pixel = 2 * self.max_distance / self.height

        self.save_all = False  # True

        self.wheelchair_path: list[np.ndarray] = []
        self.force_correct_decay = False

        if self.dataformat == "CrowdBot":
            with open(self.input_path, "rb") as np_data:
                self.data = np.load(np_data, allow_pickle=True).item()
            try:
                with open(opts.input_tf, "rb") as f:
                    self.data_tf = np.load(f, allow_pickle=True).item()
            except Exception:
                raise Exception("A correct input for the tf frames has to be supplied")

            try:
                with open(opts.input_vel, "rb") as f:
                    self.data_vel = np.load(f, allow_pickle=True).item()
            except Exception:
                raise Exception("A correct input for the vel frames has to be supplied")

            diff_timestamp = np.mean(
                self.data_tf["timestamp"][1:] - self.data_tf["timestamp"][:-1]
            )
            self.timestep = np.round(diff_timestamp, decimals=3)

        self.datasize = (
            len(os.listdir(self.input_path)) - 1
            if self.dataformat == "JRDB"
            else len(self.data.keys())
        )

        self.check_directories()

        self.generate_images()
        if self.gen_label:
            self.generate_label()

    def check_directories(self) -> None:
        output_input = f"{self.output_path}/input"
        os.makedirs(output_input, exist_ok=True)

        output_label = f"{self.output_path}/label"
        os.makedirs(output_label, exist_ok=True)

        output_overview = f"{self.output_path}/overview"
        os.makedirs(output_overview, exist_ok=True)

        if self.save_all:
            data_full_dir = f"{self.output_path}/full_data"
            os.makedirs(data_full_dir, exist_ok=True)

    def generate_images(self) -> None:
        logger.info("Generating input images")
        self.img = np.zeros((self.height, self.width))
        img_speed = []
        for index in range(self.datasize):
            start = time.time()
            if self.decay_type == "linear":
                if (
                    self.decay != self.max_weight * self.timestep / self.time_interval
                    and self.force_correct_decay
                ):
                    raise Exception(
                        f"""
                        Decay doesnt match weight and timestep. 
                        Current decay: {self.decay}, 
                        Current max weight: {self.max_weight}, 
                        Current timestep: {self.timestep}, 
                        Current timeinterval: {self.time_interval},
                        Desired decay: {self.max_weight* self.timestep/self.time_interval},
                    """
                    )
                self.img = np.clip(
                    self.img - self.decay, 0, 255
                )  # Darken image to remove oldest data
            else:
                raise Exception("Other methods are not implemented yet")

            obstacles = self.get_data(index)

            self.update_img(obstacles)

            img = self.img[
                self.curr_origin[0] : self.curr_origin[0] + self.height,
                self.curr_origin[1] : self.curr_origin[1] + self.width,
            ]

            logger.debug(
                f"Origin {self.origin}:{self.origin + np.array([self.height,self.width]).astype(int)}"
            )
            logger.debug(
                f"Full image shift {self.full_img_shift}, {self.full_img_shift*self.distance_between_pixel}"
            )
            logger.debug(f"Current origin {self.curr_origin}")

            input_img = Image.fromarray(img.astype(np.uint8), mode="L")
            input_img.save(f"{self.output_path}/input/{str(index).zfill(6)}.png")
            logger.info(f"Progress: {index}/{self.datasize}")
            end = time.time()
            img_speed.append(end - start)

        print(f"Average computation speed: {np.mean(img_speed)}")

    def get_data(self, index: int, velocity: bool = False) -> list[Detection]:
        if self.dataformat == "JRDB":
            file_name = str(index).zfill(6)
            file = open(f"{self.input_path}/{file_name}.json")
            data = json.load(file)
            pointcloud_name = f"{file_name}.pcd"
            detections = data["detections"][pointcloud_name]

            translation = data["robot"]["translation"]
            pose = data["robot"]["quaternion"]

            transform = np.zeros((4, 4))
            transform[:, 3] = translation
            euler = quat_to_euler(pose)
            transform[0:3, 0:3] = euler_to_rot(euler)

            if index == 0:
                self.global_translation = translation[:3]
                self.global_transform = np.linalg.inv(transform)

            self.wheelchair_path.append(translation[:3])
            self.update_img_size(np.array(translation[:3]))

            obstacles: list[Detection] = []
            for obstacle_data in detections:
                box = obstacle_data["box"]
                character_index = int(obstacle_data["label_id"].split(":")[1])
                obstacle_position = np.array([box["cx"], box["cy"]])
                detection = Detection(
                    centre=obstacle_position,
                    length=box["l"],
                    width=box["w"],
                    rotation=box["rot_z"],
                    index=character_index,
                )
                if detection.inside_radius(self.max_distance):
                    if velocity:
                        vel = np.array(obstacle_data["attributes"]["velocity"][0:2])
                        detection.velocity = vel
                    if index != 0:
                        detection.transform_to_global_coordinates(
                            local_transformation_matrix=transform,
                            global_transformation_matrix=self.global_transform,
                            rotation=euler[2],
                        )

                    obstacles.append(detection)

        elif self.dataformat == "CrowdBot":
            data = self.data[index]

            translation = self.data_tf["position"][index]

            if index == 0:
                self.global_translation = translation

            self.wheelchair_path.append(translation)
            self.update_img_size(translation=translation)

            obstacles: list[Detection] = []
            for idx in range(data.shape[0]):
                # obstacle index data [x, y, z, l, w, h, rotation, char_idx]
                obstacle_data = data[idx, :]

                detection = Detection(
                    centre=obstacle_data[:2],
                    length=obstacle_data[3],
                    width=obstacle_data[4],
                    rotation=obstacle_data[6],
                    index=obstacle_data[-1],
                )

                if detection.inside_radius_global(
                    max_distance=self.max_distance, position=translation
                ):
                    if velocity:
                        ped_idx = obstacle_data[-1]
                        ped_data = self.data_vel[ped_idx]

                        if (
                            "orientation_list" not in ped_data.keys()
                        ):  # Not enought data for smoothed velocity data
                            curr_pos = obstacle_data[:2]

                            prev_data = self.data[index - 1] if index != 0 else data
                            prev_detections = prev_data[:, 7]

                            previously_detected = prev_detections == obstacle_data[7]
                            if np.any(previously_detected):
                                prev_pos = prev_data[previously_detected].T[:2].reshape(-1)
                            else:
                                prev_pos = curr_pos

                            vel = (curr_pos - prev_pos) / self.timestep

                            detection.velocity = vel[:2]

                        else:
                            timestamp = self.data_tf["timestamp"][index]

                            try:
                                ped_vel_idx = np.argmin(
                                    np.abs(ped_data["timestamp_list"] - timestamp)
                                )
                            except Exception:
                                raise Exception("Timestamp not found")

                            ped_vel = ped_data["orientation_list"][ped_vel_idx]

                            detection.velocity = ped_vel[:2]

                    obstacles.append(detection)

        else:
            raise Exception("Other dataformats are not yet implemented.")

        return obstacles

    def update_img_size(self, translation: np.ndarray) -> None:
        img_size = np.array(list(self.img.shape))
        self.get_updated_origin(translation=translation)

        if not self.origin_changed:
            return

        new_img_size = np.max(
            (
                img_size[:2],
                self.curr_origin + np.array([self.height, self.width]),
                self.origin + np.array([self.height, self.width]),
            ),
            axis=0,
        )

        # if np.all(new_img_size == img_size):
        #     return

        start_idx = (
            np.zeros(2).astype(int) if np.all(self.full_img_shift >= 0) else self.full_img_shift
        )

        new_img = (
            np.zeros((new_img_size[0], new_img_size[1]))
            if len(img_size) == 2
            else np.zeros((new_img_size[0], new_img_size[1], 3))
        )

        if len(img_size) == 2:
            new_img[
                start_idx[0] : start_idx[0] + img_size[0],
                start_idx[1] : start_idx[1] + img_size[1],
            ] = self.img
        else:
            new_img[
                start_idx[0] : start_idx[0] + img_size[0],
                start_idx[1] : start_idx[1] + img_size[1],
                :,
            ] = self.img

        self.img = new_img
        # self.origin = new_origin

    def get_updated_origin(self, translation: np.ndarray) -> None:
        global_translation = translation - self.global_translation
        shift = np.round(global_translation[0:2] / self.distance_between_pixel).astype(int)
        # self.origin_coordinates += shift * self.distance_between_pixel

        # full_img_shift = np.array([0, 0]) # Shift to keep indecies of image positive

        if np.any(self.curr_origin != self.origin.astype(int) + shift):
            # origin_shift = np.zeros(3)
            # origin_shift[:2] += new_origin*self.distance_between_pixel
            # self.global_translation += origin_shift
            self.origin_changed = True
            self.curr_origin = self.origin.astype(int) + shift
        else:
            self.origin_changed = False

        if np.any(self.curr_origin < 0):
            self.origin = np.where(
                self.curr_origin < 0, self.origin + np.abs(self.curr_origin), self.origin
            )
            self.full_img_shift = np.where(
                self.curr_origin < 0,
                self.full_img_shift + np.abs(self.curr_origin),
                self.full_img_shift,
            )
            self.curr_origin = np.where(
                self.curr_origin < 0, np.zeros(2).astype(int), self.curr_origin
            )

    def update_img(self, obstacle: list[Detection]) -> None:
        border = True if self.input_occupancy_type == "bounding_box" else False
        x_grid, y_grid = self.get_cell_grid(border=border)

        for obs in obstacle:
            if self.input_occupancy_type == "bounding_box":
                occupied_cells = obs.get_occupied_indecies(x_grid=x_grid, y_grid=y_grid)
                occupied_cells = occupied_cells[
                    np.all(occupied_cells < list(self.img.shape), axis=1)
                ]
                occupied_cells = occupied_cells - self.curr_origin
                for cell in occupied_cells:
                    self.img[tuple(cell)] += self.max_weight

            elif self.input_occupancy_type == "gaussian_distribution":
                mean = obs.centre
                pdf_x = norm.pdf(x_grid, mean[0], np.sqrt(self.occupancy_variance))
                pdf_x = pdf_x.reshape(-1, 1) / np.max(
                    pdf_x
                )  # Scale density function s.t. max value is 1

                pdf_y = norm.pdf(y_grid, mean[1], np.sqrt(self.occupancy_variance))
                pdf_y = pdf_y.reshape(-1, 1) / np.max(
                    pdf_y
                )  # Scale density function s.t. max value is 1

                pdf_2d = np.matmul(pdf_x, pdf_y.T)

                weight_increase = np.round(pdf_2d * self.max_weight)
                self.img[
                    self.curr_origin[0] : self.curr_origin[0] + self.height,
                    self.curr_origin[1] : self.curr_origin[1] + self.width,
                ] += weight_increase

            else:
                raise Exception("Other methods are not implemented yet")

        self.img = np.clip(self.img, 0, 255)

    def get_cell_grid(
        self, border: bool = True, label: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        x_grid_start = (
            self.origin_coordinates[0] - self.full_img_shift[0] * self.distance_between_pixel
        )
        y_grid_start = (
            self.origin_coordinates[1] - self.full_img_shift[1] * self.distance_between_pixel
        )

        # if label:
        #     x_grid_start += self.curr_origin[0]* self.distance_between_pixel
        #     y_grid_start += self.curr_origin[1]* self.distance_between_pixel

        x_grid_end = x_grid_start + self.distance_between_pixel * self.img.shape[0]
        grid_size_x = self.img.shape[0] + 1 if border else self.img.shape[0]
        x_grid = np.linspace(start=x_grid_start, stop=x_grid_end, num=grid_size_x, endpoint=border)

        y_grid_end = y_grid_start + self.distance_between_pixel * self.img.shape[1]
        grid_size_y = self.img.shape[1] + 1 if border else self.img.shape[1]
        y_grid = np.linspace(start=y_grid_start, stop=y_grid_end, num=grid_size_y, endpoint=border)

        # Reduce size of grid to visible img size
        reduced_grid_size_x = self.height + 1 if border else self.height + 0
        reduced_grid_size_y = self.width + 1 if border else self.width + 0
        x_grid = x_grid[self.curr_origin[0] : self.curr_origin[0] + reduced_grid_size_x]
        y_grid = y_grid[self.curr_origin[1] : self.curr_origin[1] + reduced_grid_size_y]

        logger.debug(f"Current origin {self.curr_origin}")
        logger.debug(f"Current grid size: ({len(x_grid)}, {len(y_grid)})")
        logger.debug(f"X grid {x_grid[0]}:{x_grid[-1]}, Y grid {y_grid[0]}:{y_grid[-1]}")
        x_grid_start = (
            self.origin_coordinates[0] - self.full_img_shift[0] * self.distance_between_pixel
        )
        y_grid_start = (
            self.origin_coordinates[1] - self.full_img_shift[1] * self.distance_between_pixel
        )
        logger.debug(f"X grid {x_grid_start}:{x_grid_end}, Y grid {y_grid_start}:{y_grid_end}")
        return x_grid, y_grid

    def generate_label(self) -> None:
        logger.info("Generating label images")
        label_speed = []

        # Reset values set during input generation
        self.img = np.zeros((self.height, self.width, 3))
        self.origin = np.array([0, 0], dtype=int)
        self.curr_origin = np.array([0, 0], dtype=int)
        self.full_img_shift = np.array([0, 0], dtype=int)

        total_time = int(self.time_interval / self.timestep)

        obstacle_data = np.full((total_time, 0), None)

        for index in range(self.datasize):
            start = time.time()
            obstacles = self.get_data(index=index, velocity=True)
            if len(obstacles) > 0:
                max_obstacle_idx_new = int(np.max([obs.index for obs in obstacles]))

            data_dict = {}

            if len(obstacles) > 0:
                max_obstacle_idx_new = int(np.max([obs.index for obs in obstacles]))

                num_obstacles = max(max_obstacle_idx_new + 1, obstacle_data.shape[1])

                obstacle_data_new = np.full((total_time, num_obstacles, 4), None)

                obstacle_data = np.roll(
                    obstacle_data, shift=-1, axis=0
                )  # roll data to have data to remove at the bottom
                if obstacle_data.shape[1] != 0 and obstacle_data.shape[2] == 4:
                    obstacle_data_new[: total_time - 1, : obstacle_data.shape[1], :] = (
                        obstacle_data[: total_time - 1, :, :]
                    )

                new_data = np.full((num_obstacles, 4), None)
                for obs in obstacles:
                    new_data[int(obs.index), :] = np.array(
                        [obs.centre[0], obs.centre[1], obs.velocity[0], obs.velocity[1]]
                    )

                obstacle_data_new[-1, :, :] = new_data
                obstacle_data = obstacle_data_new

                new_img = np.zeros((self.height, self.width, 3))
                x_grid, y_grid = self.get_cell_grid(border=True)
                logger.debug(
                    f"X grid label {x_grid[0]}:{x_grid[-1]}, Y grid label {y_grid[0]}:{y_grid[-1]}"
                )
                x_val = (x_grid[:-1] + x_grid[1:]) / 2  # Get centre of each cell
                y_val = (y_grid[:-1] + y_grid[1:]) / 2
                xy_coords = np.array(np.meshgrid(x_val, y_val)).T.reshape(-1, 2)

                idx = np.linspace(0, self.height, self.height, endpoint=False)
                xy_idx = np.array(np.meshgrid(idx, idx)).T.reshape(-1, 2).astype(int)

                pos_data = obstacle_data[
                    :, :, :2
                ].T  # shape(pos_x/pos_y, num_obstacles, total_time)

                in_search_radius = np.zeros(
                    (xy_coords.shape[0], num_obstacles, total_time), dtype=bool
                )

                for t in range(pos_data.shape[2]):
                    for obs in range(pos_data.shape[1]):
                        if np.any(pos_data[:, obs, t] == None):
                            continue
                        if self.labelling_occupancy_type == "proximity":
                            distance_to_cell = (xy_coords - pos_data[:, obs, t]).astype(float)
                            in_search_radius[:, obs, t] = (
                                np.linalg.norm(distance_to_cell, axis=1) <= self.search_radius
                            )
                        else:
                            raise Exception("Other methods have not been implemented yet.")

                accessible = np.any(np.any(in_search_radius, axis=-1), axis=-1)
                new_img[accessible.reshape(self.height, self.width)] = list(
                    SceneClassColour.ACCESSIBLE.value
                )

                unique_detections = np.sum(np.any(in_search_radius, axis=-1), axis=-1)
                # ToDo: if people close by in one frame identify them as one person

                if self.save_all:
                    for i in xy_idx:
                        x, y = i
                        save_idx = np.argwhere(np.all(xy_idx == [x, y], axis=1))
                        data_dict[f"{x}:{y}"] = {
                            "class": (
                                SceneClassColour.ACCESSIBLE.name
                                if unique_detections[save_idx]
                                else SceneClassColour.UNASSIGNED.name
                            ),
                            "vel": np.array([]),
                        }

                idx_to_check = unique_detections >= self.detection_threshold
                img_idx = xy_idx[idx_to_check]

                idx_with_detection = np.argwhere(idx_to_check)

                vel_data = obstacle_data[:, :, 2:]

                for detection_index, image_index in zip(idx_with_detection, img_idx):
                    vel_idx = np.argwhere(in_search_radius[detection_index, :, :].T)
                    unique_vel_idx = list(set(vel_idx.T[1, :]))
                    detected_velocities = np.zeros((len(unique_vel_idx), 2))
                    for idx, i in enumerate(unique_vel_idx):
                        detection_times = vel_idx[vel_idx.T[1, :] == i].T[0, :]
                        obs_vel_data = np.array([vel_data[t, i, :].T for t in detection_times])
                        if not np.any(obs_vel_data is None) and len(obs_vel_data) > 0:
                            try:
                                vel_cartesian = np.mean(obs_vel_data, axis=0)
                            except Exception:
                                vel_cartesian = 0
                        detected_velocities[idx] = cartesian_to_polar(vel_cartesian)

                    # mean_velocity = np.mean(detected_velocities, axis=0)
                    # variance_velocity = np.var(detected_velocities, axis=0)

                    x, y = image_index

                    if np.sum(np.abs(detected_velocities[:, 0]) <= self.velocity_threshold) >= 2:
                        new_img[x, y, :] = list(SceneClassColour.STATIC.value)
                        if self.save_all:
                            data_dict[f"{x}:{y}"]["class"] = SceneClassColour.STATIC.name
                            data_dict[f"{x}:{y}"]["vel"] = np.array([[0, 0]])
                        logger.debug("STATIC")

                    else:
                        non_zero_angles = detected_velocities[
                            np.abs(detected_velocities[:, 0]) > self.velocity_threshold, 1
                        ] % (2 * np.pi)
                        sorted_angles = np.sort(
                            non_zero_angles
                        )  # Sort all angles with non-zero magnitude

                        sections = []
                        for i, ang in enumerate(sorted_angles):
                            if i != 0 and ang <= sections[-1][0] + self.angle_threshold:
                                sections[-1].append(ang)
                            elif (
                                i != 0
                                and ang + self.angle_threshold > 2 * np.pi
                                and (ang + self.angle_threshold) % (2 * np.pi) > sections[0][-1]
                            ):
                                sections[0].insert(0, ang)
                            else:
                                sections.append([ang])

                        full_sections = []
                        full_sections_data = []
                        for sec in sections:  # Remove all sections with less then 2 elements
                            if len(sec) >= 2:
                                full_sections.append(sec)

                                if self.save_all:
                                    mag_data = detected_velocities[
                                        np.searchsorted(detected_velocities[:, 1], sec)
                                    ]
                                    mag_mean = np.mean(mag_data)
                                    rad_mean = np.mean(sec)
                                    full_sections_data.append([mag_mean, rad_mean])

                        if len(full_sections) == 1:
                            new_img[x, y, :] = list(SceneClassColour.ONEDIRECTIONAL.value)
                            if self.save_all:
                                data_dict[f"{x}:{y}"][
                                    "class"
                                ] = SceneClassColour.ONEDIRECTIONAL.name
                            logger.debug("ONEDIRECTIONAL")

                        elif (
                            len(full_sections) == 2
                            and np.abs(
                                (full_sections[0][0] - full_sections[1][0]) % (2 * np.pi) - np.pi
                            )
                            <= self.angle_threshold
                        ):
                            new_img[x, y, :] = list(SceneClassColour.TWODIRECTIONAL.value)
                            if self.save_all:
                                data_dict[f"{x}:{y}"][
                                    "class"
                                ] = SceneClassColour.TWODIRECTIONAL.name
                            logger.debug("TWODIRECTIONAL")

                        elif len(full_sections) == 2:
                            new_img[x, y, :] = list(SceneClassColour.CROSSING.value)
                            if self.save_all:
                                data_dict[f"{x}:{y}"]["class"] = SceneClassColour.CROSSING.name

                        elif len(full_sections) > 2 or len(sorted_angles) > 5:
                            new_img[x, y, :] = list(SceneClassColour.MIXED.value)
                            if self.save_all:
                                data_dict[f"{x}:{y}"]["class"] = SceneClassColour.MIXED.name
                            logger.debug("MIXED")

                        else:
                            new_img[x, y, :] = list(SceneClassColour.ACCESSIBLE.value)
                            logger.debug("ACCESSIBLE")

                        if self.save_all:
                            data_dict[f"{x}:{y}"]["vel"] = np.array(full_sections_data)

                    if False:
                        for v in detected_velocities:
                            vel = polar_to_cartesian(v)
                            pos_x = np.array([0, vel[0]])
                            pos_y = np.array([0, vel[1]])
                            plt.plot(pos_x, pos_y)
                        plt.title(f"{SceneClassColour(tuple(new_img[x, y, :]))}")
                        plt.legend(
                            [
                                f"{i}: radius {detected_velocities[i][0]:.3f}, angle {detected_velocities[i][1]:.3f} "
                                for i in range(len(detected_velocities))
                            ]
                        )
                        plt.savefig(f"{self.output_path}/quiver/{index}_{x}_{y}.png")
                        plt.close()
            else:
                new_img = np.zeros((self.height, self.width, 3))

            if False:
                fig = plt.figure()
                wheelchair_path = np.array(self.wheelchair_path)
                plt.plot(wheelchair_path[:, 0], wheelchair_path[:, 1])
                plt.savefig(f"{self.output_path}/path/{index}.png")
                plt.close()

            prev_img = self.img[
                self.curr_origin[0] : self.curr_origin[0] + self.height,
                self.curr_origin[1] : self.curr_origin[1] + self.width,
                :,
            ]
            empty = np.all(prev_img == np.zeros(3, dtype=int), axis=2)
            accessible = np.all(prev_img == list(SceneClassColour.ACCESSIBLE.value), axis=2)
            prev_img[empty | accessible] = new_img[empty | accessible]
            self.img[
                self.curr_origin[0] : self.curr_origin[0] + self.height,
                self.curr_origin[1] : self.curr_origin[1] + self.width,
                :,
            ] = prev_img

            label_img = Image.fromarray(new_img.astype(np.uint8), mode="RGB")
            label_img.save(f"{self.output_path}/label/{str(index).zfill(6)}.png")

            overview_img = Image.fromarray(self.img.astype(np.uint8), mode="RGB")
            overview_img.save(f"{self.output_path}/overview/{str(index).zfill(6)}.png")

            if self.save_all:
                with open(f"{self.output_path}/full_data/{str(index).zfill(6)}.npy", "wb") as f:
                    np.save(f, data_dict)

            logger.info(f"Progress: {index}/{self.datasize}")
            end = time.time()
            label_speed.append(end - start)

        print(f"Average label computation speed: {np.mean(label_speed)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    input_folder = "/scai_data/data01/daav/isaac_sim_synthetic_data/label_generation/training_data/random-movement/rosbag2_2024_02_08-11_16_33"  # "/scai_data/data01/daav/isaac_sim_synthetic_data/label_generation/training_data/random-movement/rosbag2_2024_02_08-13_35_43"  #"/scai_data/data01/daav/CrowdBot/lidar_odom_test3_processed"
    file_name = "defaced_2021-12-03-19-12-00_filtered_lidar_odom"  # "huang-lane-2019-02-12_0_all_transforms"
    tf_name = "tfqolo"
    tf_folder_name = "tf_qolo"
    dataformat = "JRDB"

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input file path",
        default=(
            f"{input_folder}/alg_res/tracks/{file_name}.npy"
            if dataformat == "CrowdBot"
            else f"{input_folder}/lidar_3d"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path",
        default=(
            f"{input_folder}/label_generation/{file_name}"
            if dataformat == "CrowdBot"
            else f"{input_folder}"
        ),
    )

    parser.add_argument(
        "--label",
        "-l",
        type=bool,
        help="Decide if labels should also be generated or only input",
        default=True,
    )

    parser.add_argument(
        "--dataformat",
        "-d",
        type=str,
        help="Dataformat of input data (JRDB, CrowdBot, ...)",
        default=dataformat,
    )
    parser.add_argument(
        "--input_tf",
        "-itf",
        type=str,
        help="Input path to tf frames (required if dataformat == CrowdBot)",
        default=f"{input_folder}/source_data/{tf_folder_name}/{file_name}_{tf_name}_sampled.npy",
    )
    parser.add_argument(
        "--input_vel",
        "-iv",
        type=str,
        help="Set input path for velocity data (required if dataformat == CrowdBot)",
        default=f"{input_folder}/alg_res/tracks/velocities/vel_{file_name}.npy",
    )

    opts = parser.parse_args()

    generator = Generator(opts)
