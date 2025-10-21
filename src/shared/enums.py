from enum import Enum


class InteractionType(str, Enum):
    USER = "user"
    MODEL = "model"
    TOOL = "tool"

class SourceType(Enum):
    FILE = "FILE"
    WEB_PAGE = "WEB_PAGE"
    QA_PAIR = "QA_PAIR"
