import numpy as np


def bresenham(pos_1: np.ndarray, pos_2: np.ndarray) -> list[list[int]]:
    x1, y1 = pos_1
    x2, y2 = pos_2

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    slope = dy > dx

    if slope:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    error = dx // 2
    y = y1
    ystep = 1 if y1 < y2 else -1

    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if slope else [x, y]
        points.append(coord)
        error -= dy
        if error < 0:
            y += ystep
            error += dx

    return points


def quat_to_euler(quat: list[float]) -> np.ndarray:
    if len(quat) != 4:
        raise Exception(f"Length of quaternion incorrect. Length is {len(quat)} instead of 4")

    x, y, z, w = quat

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def euler_to_rot(euler: np.ndarray) -> np.ndarray:
    if len(euler) != 3:
        raise Exception(f"Length of euler vector incorrect. Length is {len(euler)} instead of 3")

    roll, pitch, yaw = euler

    R = np.array(
        [
            [
                np.cos(pitch) * np.cos(yaw),
                np.sin(roll) * np.sin(pitch) * np.cos(yaw) - np.sin(yaw) * np.cos(roll),
                np.sin(pitch) * np.cos(roll) * np.cos(yaw) + np.sin(roll) * np.sin(yaw),
            ],
            [
                np.sin(yaw) * np.cos(pitch),
                np.sin(roll) * np.sin(pitch) * np.sin(yaw) + np.cos(roll) * np.cos(yaw),
                np.sin(pitch) * np.sin(yaw) * np.cos(roll) - np.sin(roll) * np.cos(yaw),
            ],
            [
                -np.sin(pitch),
                np.sin(roll) * np.cos(pitch),
                np.cos(roll) * np.cos(pitch),
            ],
        ]
    )

    return R


def quat_to_rot(quat: list[float]) -> np.ndarray:
    euler = quat_to_euler(quat=quat)
    return euler_to_rot(euler=euler)


def cartesian_to_polar(coords: np.ndarray) -> np.ndarray:
    if coords.shape[0] != 2:
        raise Exception(f"Coordinates have to have shape 2. Current shape: {coords.shape}")
    x = coords[0]
    y = coords[1]

    mag = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x) % (2 * np.pi)

    return np.array([mag, angle])


def polar_to_cartesian(coords: np.ndarray) -> np.ndarray:
    if coords.shape[0] != 2:
        raise Exception(f"Coordinates have to have shape 2. Current shape: {coords.shape}")
    rho = coords[0]
    phi = coords[1]

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    return np.array([x, y])
