"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import numpy as np # for np ops
from typing import Tuple

def check_if_buggy_region(pos:Tuple[int], visited:np.array) -> bool:
    if visited[pos]:
        return 0,visited
    if pos[0] <= 2:
        if pos[1] <= 2:
            visited[:2,:2] = 1
            return 50,visited
        if pos[1] >= 7:
            visited[:2,7:] = 1
            return 50,visited
    
    if pos[0] >= 7:
        if pos[1] <= 2:
            visited[7:,:2] = 1
            return 50,visited
        if pos[1] >= 7:
            visited[7:,7:] = 1
            return 50,visited
    return 0,visited