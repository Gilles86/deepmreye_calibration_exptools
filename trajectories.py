"""Trajectory generation for DeepMReye calibration.

Ported from getFixLocations.m and getFixLocations_pursuit.m.
"""
import numpy as np


def _as_xy(win_size_deg):
    """Return (win_x, win_y) from a scalar (square) or (x, y) pair."""
    if np.isscalar(win_size_deg):
        return float(win_size_deg), float(win_size_deg)
    win_x, win_y = win_size_deg
    return float(win_x), float(win_y)


def generate_fixation_grid(win_size_deg, n_locs):
    """Create an evenly-spaced grid of fixation positions in degrees.

    Port of getFixLocations.m.

    Parameters
    ----------
    win_size_deg : float or (float, float)
        Calibration window size in degrees. A scalar gives a square window;
        a (width, height) pair gives independent x/y extents.
    n_locs : list of int
        [n_x, n_y] grid dimensions.

    Returns
    -------
    xy : ndarray, shape (n_x * n_y, 2)
        Shuffled fixation positions in degrees.
    """
    win_x, win_y = _as_xy(win_size_deg)
    x = np.linspace(-win_x / 2, win_x / 2, n_locs[0])
    y = np.linspace(-win_y / 2, win_y / 2, n_locs[1])
    xx, yy = np.meshgrid(x, y, indexing='ij')
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    np.random.shuffle(xy)
    return xy


def generate_pursuit_trajectory(win_size_deg, angles, amplitudes_deg,
                                duration, framerate):
    """Build a pseudorandom-walk pursuit trajectory.

    Port of getFixLocations_pursuit.m. Each trial moves the fixation
    cross from one waypoint to the next via linear interpolation.

    Parameters
    ----------
    win_size_deg : float or (float, float)
        Calibration window size in degrees. A scalar gives a square window;
        a (width, height) pair bounds x and y independently.
    angles : array-like
        Movement angles in radians.
    amplitudes_deg : array-like
        Movement amplitudes in degrees.
    duration : float
        Duration per segment in seconds.
    framerate : float
        Monitor refresh rate in Hz.

    Returns
    -------
    trajectories : list of ndarray
        Each element is an array of shape (n_frames, 2) with x/y positions
        in degrees. First element is a stationary trial at (0, 0).
    """
    n_frames = int(round(duration * framerate))

    # Build all angle x amplitude combos (repeating angles for each amplitude)
    mov_angles = np.tile(angles, len(amplitudes_deg))
    mov_amp = np.repeat(amplitudes_deg, len(angles))

    win_x, win_y = _as_xy(win_size_deg)
    calib_half_x = win_x / 2
    calib_half_y = win_y / 2
    n_trials = len(mov_angles) + 1  # +1 for the starting position

    # Retry from scratch until a valid trajectory is found
    while True:
        remaining_angles = list(mov_angles)
        remaining_amp = list(mov_amp)
        waypoints = [[0.0, 0.0]]
        stuck = 0

        while len(waypoints) < n_trials:
            if stuck > 50:
                break

            idx = np.random.randint(len(remaining_angles))
            angle = remaining_angles[idx]
            amp = remaining_amp[idx]

            new_x = waypoints[-1][0] + amp * np.cos(angle)
            new_y = waypoints[-1][1] + amp * np.sin(angle)

            if (abs(new_x) > calib_half_x or abs(new_y) > calib_half_y):
                stuck += 1
                continue

            waypoints.append([new_x, new_y])
            remaining_angles.pop(idx)
            remaining_amp.pop(idx)
            stuck = 0

        if len(waypoints) == n_trials:
            break

    waypoints = np.array(waypoints)

    # Interpolate between waypoints for each movement segment
    trajectories = []
    for i in range(1, len(waypoints)):
        # linspace from previous to current, n_frames+1 points, drop first
        x_interp = np.linspace(waypoints[i - 1, 0], waypoints[i, 0],
                               n_frames + 1)[1:]
        y_interp = np.linspace(waypoints[i - 1, 1], waypoints[i, 1],
                               n_frames + 1)[1:]
        trajectories.append(np.column_stack([x_interp, y_interp]))

    # Prepend a stationary trial at (0, 0)
    stationary = np.zeros((n_frames, 2))
    trajectories.insert(0, stationary)

    return trajectories
