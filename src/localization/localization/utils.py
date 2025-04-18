import os

import cv2
import numpy as np
import yaml
from geometry_msgs.msg import Point, Pose, Quaternion
from nav_msgs.msg import MapMetaData, OccupancyGrid


def load_map(yaml_path: str) -> OccupancyGrid:
    """Load a map from a yaml file as an OccupancyGrid message."""
    folder = os.path.dirname(yaml_path)
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    img = cv2.imread(os.path.join(folder, cfg["image"]), cv2.IMREAD_GRAYSCALE)
    # Flip image vertically
    img = cv2.flip(img, 0)
    # Compute occupancy probability
    if cfg.get("negate", 0):
        p = img / 255.0
    else:
        p = (255.0 - img) / 255.0
    free = p <= cfg["free_thresh"]
    occ = p >= cfg["occupied_thresh"]
    data = np.full(img.shape, -1, dtype=np.int8)
    data[free] = 0
    data[occ] = 100
    # Create occupancy grid message
    msg = OccupancyGrid()
    msg.header.frame_id = cfg.get("frame_id", "map")
    meta = MapMetaData(
        resolution=cfg["resolution"],
        width=img.shape[1],
        height=img.shape[0],
        origin=Pose(
            position=Point(x=cfg["origin"][0], y=cfg["origin"][1], z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=1.0, w=0.0),
        ),
    )
    msg.info = meta
    msg.data = data.flatten().tolist()
    # with open("debug.txt", "w") as f:
    #     f.write(f"{msg}\n")
    return msg
