from collections import namedtuple
from enum import Enum


class SceneClassEnum(Enum):
    UNASSIGNED = 0
    STATIC = 1
    UNIDIRECTIONAL = 2
    BIDIRECTIONAL = 3
    MIXED = 4
    ACCESSIBLE = 5
    CROSSING = 6


class SceneClassColour(Enum):
    UNASSIGNED = (0, 0, 0)  # Black
    STATIC = (255, 0, 0)  # Red
    UNIDIRECTIONAL = (255, 255, 0)  # Yellow
    BIDIRECTIONAL = (0, 255, 0)  # Green
    MIXED = (0, 255, 255)  # Teal
    ACCESSIBLE = (0, 0, 255)  # Blue
    CROSSING = (255, 0, 255)


SceneClass = namedtuple("SceneClass", ["name", "id", "color"])

classes = [
    SceneClass(
        SceneClassEnum.UNASSIGNED.name,
        SceneClassEnum.UNASSIGNED.value,
        SceneClassColour.UNASSIGNED.value,
    ),
    SceneClass(
        SceneClassEnum.STATIC.name, SceneClassEnum.STATIC.value, SceneClassColour.STATIC.value
    ),
    SceneClass(
        SceneClassEnum.UNIDIRECTIONAL.name,
        SceneClassEnum.UNIDIRECTIONAL.value,
        SceneClassColour.UNIDIRECTIONAL.value,
    ),
    SceneClass(
        SceneClassEnum.BIDIRECTIONAL.name,
        SceneClassEnum.BIDIRECTIONAL.value,
        SceneClassColour.BIDIRECTIONAL.value,
    ),
    SceneClass(
        SceneClassEnum.MIXED.name, SceneClassEnum.MIXED.value, SceneClassColour.MIXED.value
    ),
    SceneClass(
        SceneClassEnum.ACCESSIBLE.name,
        SceneClassEnum.ACCESSIBLE.value,
        SceneClassColour.ACCESSIBLE.value,
    ),
    SceneClass(
        SceneClassEnum.CROSSING.name,
        SceneClassEnum.CROSSING.value,
        SceneClassColour.CROSSING.value,
    ),
]

train_id_to_color = [[c.color] for c in classes]
