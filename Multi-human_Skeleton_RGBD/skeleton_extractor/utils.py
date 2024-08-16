from pathlib import Path
import sys
from loguru import logger
from typing import Optional
import time
import os
import numpy as np

def get_project_dir_path() -> Path:
    # logger.debug(f"debug:{__file__}\n{Path(__file__).parents}")
    for parent in Path(__file__).parents:
        if (parent / "src/hst_combo/Multi-human_Skeleton_RGBD").exists():
            # parent is workspace directory
            return parent / "src/hst_combo/Multi-human_Skeleton_RGBD"
        
    assert Exception("Cannot find the source code path")


def get_pose_model_dir() -> Path:
    # Recursively ascend the parent directories of this file's path looking for
    # the .venv folder.
    proj_path = get_project_dir_path()
    return proj_path / "models"


logger.remove(0)
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | <cyan>{file}:{line}</cyan> - <level>{message}</level>",
)


def time_logger(pre: float | None = None) -> float | None:
    cur = time.time()
    if pre is None:
        logger.info("Computational Analysis Starts")
        return cur
    else:
        logger.info(f"It took {cur-pre} seconds")
        return None


def delete_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        logger.info("Delete all files in the folder successfully.")
    except OSError:
        logger.warning("Error occurred while deleting files.")
