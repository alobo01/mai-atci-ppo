import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import imageio.v2 as imageio # Use v2 for mimwrite
import numpy as np

# Type Aliases
NpArray = np.ndarray
Loggable = Union[int, float, str, bool, None] # Added None for robustness

def overlay_text(frame: NpArray, text: str, color: Tuple[int,int,int]=(255,255,255)) -> NpArray:
    """
    Draws text with a semi-transparent background box on the frame.

    Args:
        frame: The image frame (numpy array) to draw on.
        text: The string to draw.
        color: Text color in BGR format (OpenCV default).

    Returns:
        The frame with text overlay.
    """
    out_frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    # Get text size to create a background box
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    margin = 5
    box_top_left = (margin, margin)
    box_bottom_right = (margin + text_width + margin, margin + text_height + baseline + margin)
    
    # Create a black rectangle for the background with some transparency
    sub_img = out_frame[box_top_left[1]:box_bottom_right[1], box_top_left[0]:box_bottom_right[0]]
    black_rect = np.zeros(sub_img.shape, dtype=frame.dtype)
    alpha = 0.5 # Transparency factor
    res = cv2.addWeighted(sub_img, 1 - alpha, black_rect, alpha, 0)
    out_frame[box_top_left[1]:box_bottom_right[1], box_top_left[0]:box_bottom_right[0]] = res

    # Position for the text itself (inside the box)
    text_origin = (margin + margin // 2, margin + text_height + margin // 2)
    cv2.putText(out_frame, text, text_origin, font, font_scale, color, thickness, cv2.LINE_AA)
    return out_frame

def save_video(frames: List[NpArray], filename: Union[str, Path], fps: int = 30) -> None:
    """
    Saves a list of numpy array frames as an MP4 video.

    Args:
        frames: A list of frames (each frame is a HxWxC numpy array).
        filename: Path to save the video.
        fps: Frames per second for the video.
    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    # Ensure frames are in uint8 format if they are not already
    processed_frames = [f.astype(np.uint8) if f.dtype != np.uint8 else f for f in frames]
    imageio.mimwrite(filename, processed_frames, fps=fps, quality=8) # quality (0-10, higher is better)

def save_metrics(metrics_data: Dict[str, List[Loggable]], filepath: Union[str, Path]) -> None:
    """
    Saves a dictionary of metrics (lists of loggable items) to a JSON file.

    Args:
        metrics_data: Dictionary where keys are metric names and values are lists of data.
        filepath: Path to the JSON file to save.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics_data, f, indent=4)

def load_metrics(filepath: Union[str, Path]) -> Dict[str, List[Loggable]]:
    """
    Loads metrics dictionary from a JSON file.
    Returns a default structure if the file doesn't exist or is invalid.

    Args:
        filepath: Path to the JSON file.

    Returns:
        A dictionary of metrics.
    """
    filepath = Path(filepath)
    default_metrics: Dict[str, List[Loggable]] = {"steps": [], "avg_episodic_reward": [], "avg_episode_length": []}
    if not filepath.is_file():
        return default_metrics
    
    try:
        with open(filepath, "r") as f:
            loaded_data = json.load(f)
            # Basic validation
            if isinstance(loaded_data, dict) and \
               all(key in loaded_data for key in default_metrics.keys()) and \
               all(isinstance(loaded_data[key], list) for key in default_metrics.keys()):
                return loaded_data
            else:
                print(f"Warning: Metrics file {filepath} has unexpected format. Returning default.")
                return default_metrics
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {filepath}. Returning default metrics.")
        return default_metrics

def save_timings(timing_summary: Dict[str, Dict[str, float]], filepath: Union[str, Path], step: int) -> None:
    """
    Appends timing summary for the current step to a JSON Lines file.

    Args:
        timing_summary: Dictionary from Timing.summary().
        filepath: Path to the JSON Lines file.
        step: The current training step.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    log_entry = {"step": step, **timing_summary}
    with open(filepath, "a") as f: # Append mode
        f.write(json.dumps(log_entry) + "\n")