#####################################################################################################
#
# Base class for the hub that is responsible for managing branch related operations.
# Users must provide their own implementation of this class as each branch requires different data.
#
#####################################################################################################

from abc import ABC
from typeguard import typechecked

@typechecked
class Hub(ABC):
    def __init__(self) -> None:
        return