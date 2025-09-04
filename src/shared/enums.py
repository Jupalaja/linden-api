from enum import Enum


class InteractionType(str, Enum):
    USER = "user"
    MODEL = "model"
    TOOL = "tool"
