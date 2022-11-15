"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import numpy as np # for np ops
from typing import Tuple

def check_if_buggy_region(pos:Tuple[int], visited:np.array) -> bool:
    if visited[pos[0], pos[1]]:
        return 0,visited
    if pos[0] <= 2:
        if pos[1] <= 2:
            visited[:3,:3] = 1
            return 1,visited
        if pos[1] >= 8:
            visited[:3,7:] = 1
            return 1,visited
    
    if pos[0] >= 8:
        if pos[1] <= 2:
            visited[6:,:3] = 1
            return 1,visited
        if pos[1] >= 8:
            visited[7:,7:] = 1
            return 1,visited
    return 0,visited
