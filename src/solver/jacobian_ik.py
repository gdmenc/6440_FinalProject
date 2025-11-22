import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple

from src.core.skeleton import Skeleton
from src.core.node import Node
from src.core.transform import Mat4x4

class JacobianIK:
    def __init__(self, 
                damping: float = 0.1, 
                step_size: float = 0.5, 
                max_iterations: int = 10,
                threshold: float = 0.1
    ) -> None:
        self.damping = damping
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.threshold = threshold