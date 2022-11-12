"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import numpy as np # for np ops
from typing import Tuple

def check_if_buggy_region(pos:Tuple[int], visited:np.array) -> bool:
    if visited[pos]:
        return 0
    if pos >= (7,2) or pos >= (7,7) or pos >= (2,7):
        return 50
    if pos <= (2,2):
        return 50
    
    return 0    