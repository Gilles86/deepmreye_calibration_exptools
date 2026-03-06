"""DeepMREye calibration experiment.

Presents fixation, smooth pursuit, and free picture-viewing tasks to
generate training data for DeepMReye. Ported from the Psychtoolbox
version by MN (September 2021).

Requires: exptools2, psychopy, numpy, pandas, matplotlib
"""
import os
import os.path as op
import argparse
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from psychopy.event import waitKeys
from psychopy.visual import Line, ImageStim, TextStim

from exptools2.core import Trial
from exptools2.core import PylinkEyetrackerSession
from trajectories import generate_fixation_grid, generate_pursuit_trajectory


# ------------------------------------------------------------------ #
#                           Trial classes                             #
# ------------------------------------------------------------------ #

class InstructionTrial(Trial):
    """Display instruction text for a fixed duration."""

    def __init__(self, session, trial_nr, txt, duration):
        phase_durations = [duration]
        super().__init__(
            session, trial_nr, phase_durations,
            phase_names=['instruction'],
            parameters={'trial_type': 'instruction'},
            verbose=True,
        )
        self.txt_stim = TextStim(self.session.win, text=txt, height=0.8)

    def draw(self):
        self.txt_stim.draw()
        self.session.frame_data.append({
            'onset': self.session.clock.getTime(),
            'x_deg': np.nan,
            'y_deg': np.nan,
            'trial_nr': self.trial_nr,
            'trial_type': 'instruction',
        })


class FixationTrial(Trial):
    """Show the fixation cross at a static target position."""

    def __init__(self, session, trial_nr, target_pos, duration):
        phase_durations = [duration]
        super().__init__(
            session, trial_nr, phase_durations,
            phase_names=['fixation'],
            parameters={
                'trial_type': 'fixation',
                'target_x': target_pos[0],
                'target_y': target_pos[1],
            },
            verbose=True,
        )
        self.target_pos = target_pos

    def draw(self):
        self.session.draw_cross(self.target_pos)
        self.session.frame_data.append({
            'onset': self.session.clock.getTime(),
            'x_deg': self.target_pos[0],
            'y_deg': self.target_pos[1],
            'trial_nr': self.trial_nr,
            'trial_type': 'fixation',
        })


class PursuitTrial(Trial):
    """Move the fixation cross along a pre-computed trajectory."""

    def __init__(self, session, trial_nr, trajectory, duration):
        phase_durations = [duration]
        super().__init__(
            session, trial_nr, phase_durations,
            phase_names=['pursuit'],
            parameters={'trial_type': 'pursuit'},
            verbose=True,
        )
        self.trajectory = trajectory
        self.frame_idx = 0

    def draw(self):
        idx = min(self.frame_idx, len(self.trajectory) - 1)
        pos = self.trajectory[idx]
        self.session.draw_cross(pos)
        self.session.frame_data.append({
            'onset': self.session.clock.getTime(),
            'x_deg': pos[0],
            'y_deg': pos[1],
            'trial_nr': self.trial_nr,
            'trial_type': 'pursuit',
        })
        self.frame_idx += 1


class PictureViewingTrial(Trial):
    """Display an image for free viewing."""

    def __init__(self, session, trial_nr, image_stim, image_name, duration):
        phase_durations = [duration]
        super().__init__(
            session, trial_nr, phase_durations,
            phase_names=['free_viewing'],
            parameters={
                'trial_type': 'free_viewing',
                'image': image_name,
            },
            verbose=True,
        )
        self.image_stim = image_stim

    def draw(self):
        self.image_stim.draw()
        self.session.frame_data.append({
            'onset': self.session.clock.getTime(),
            'x_deg': np.nan,
            'y_deg': np.nan,
            'trial_nr': self.trial_nr,
            'trial_type': 'free_viewing',
        })


# ------------------------------------------------------------------ #
#                           Session class                             #
# ------------------------------------------------------------------ #

class DeepMReyeCalibSession(PylinkEyetrackerSession):
    """Calibration session for DeepMReye."""

    def __init__(self, output_str, output_dir=None, settings_file=None,
                 eyetracker_on=False, calibrate_eyetracker=False):
        super().__init__(output_str, output_dir=output_dir,
                         settings_file=settings_file,
                         eyetracker_on=eyetracker_on)
        self.show_eyetracker_calibration = calibrate_eyetracker
        self.frame_data = []

        # Create fixation cross stimuli (two lines)
        dm = self.settings['deepmreye']
        cross_size = dm['cross_size_deg']
        lw = dm['cross_line_width']
        self.cross_v = Line(
            self.win, start=(0, -cross_size), end=(0, cross_size),
            lineWidth=lw, lineColor='black', units='deg',
        )
        self.cross_h = Line(
            self.win, start=(-cross_size, 0), end=(cross_size, 0),
            lineWidth=lw, lineColor='black', units='deg',
        )

    def draw_cross(self, pos):
        """Draw the fixation cross at *pos* (x, y) in degrees."""
        self.cross_v.pos = pos
        self.cross_h.pos = pos
        self.cross_v.draw()
        self.cross_h.draw()

    def create_trials(self, include_pictures=True):
        """Build the full list of trials."""
        dm = self.settings['deepmreye']

        # Calibration window: height in deg / 1.25 (matches original logic)
        height_deg = 2 * np.degrees(
            np.arctan(
                (self.monitor.getWidth()
                 * self.win.size[1] / self.win.size[0])
                / (2 * self.monitor.getDistance())
            )
        )
        win_size_deg = height_deg / 1.25

        # --- Fixation grid ---
        fix_xy = generate_fixation_grid(
            win_size_deg, dm['fixation']['n_locs'],
        )

        # --- Pursuit trajectories ---
        angles = np.deg2rad(
            np.arange(dm['pursuit']['angles_start'], 360,
                       dm['pursuit']['angles_step'])
        )
        pursuit_trajs = generate_pursuit_trajectory(
            win_size_deg, angles,
            dm['pursuit']['amplitudes_deg'],
            dm['pursuit']['duration'],
            self.actual_framerate,
        )

        # --- Images ---
        script_dir = op.dirname(op.abspath(__file__))
        img_dir = op.join(script_dir, dm['pictures']['path'])
        all_images = sorted(glob.glob(op.join(img_dir, 'image_*')))
        n_pics = min(dm['pictures']['n_pics'], len(all_images))
        chosen = list(np.random.choice(all_images, size=n_pics, replace=False))

        # Compute image size: height / 3 in pixels, convert to degrees
        pic_size_pix = self.win.size[1] / 3 * 2  # diameter = 2/3 of height
        pix_per_deg = self.win.size[1] / height_deg
        pic_size_deg = pic_size_pix / pix_per_deg

        image_stims = []
        image_names = []
        for img_path in chosen:
            stim = ImageStim(
                self.win, image=img_path,
                size=(pic_size_deg, pic_size_deg), units='deg',
            )
            image_stims.append(stim)
            image_names.append(op.basename(img_path))

        # --- Build trial list ---
        self.trials = []
        trial_nr = 0
        instr_dur = dm['instruction_duration']

        # Fixation block
        self.trials.append(InstructionTrial(
            self, trial_nr,
            'Fixation task\n\nAlways fixate at the black cross',
            instr_dur,
        ))
        trial_nr += 1

        for xy in fix_xy:
            self.trials.append(FixationTrial(
                self, trial_nr, xy, dm['fixation']['duration'],
            ))
            trial_nr += 1

        # Pursuit block
        self.trials.append(InstructionTrial(
            self, trial_nr,
            'Smooth-pursuit task\n\nAlways fixate at the moving black cross',
            instr_dur,
        ))
        trial_nr += 1

        for traj in pursuit_trajs:
            self.trials.append(PursuitTrial(
                self, trial_nr, traj, dm['pursuit']['duration'],
            ))
            trial_nr += 1

        # Picture viewing block
        if include_pictures:
            self.trials.append(InstructionTrial(
                self, trial_nr,
                'Free viewing task\n\nExplore the following images however you like',
                instr_dur,
            ))
            trial_nr += 1

            for stim, name in zip(image_stims, image_names):
                self.trials.append(PictureViewingTrial(
                    self, trial_nr, stim, name, dm['pictures']['duration'],
                ))
                trial_nr += 1

    def run(self):
        """Run the full experiment."""
        if self.eyetracker_on and self.show_eyetracker_calibration:
            self.calibrate_eyetracker()

        sync_key = self.settings['mri'].get('sync', 's')

        # Wait for scanner trigger (also accept 'q' to quit)
        stim = TextStim(self.win,
                        text=f"Waiting for scanner trigger ('{sync_key}')...\n\n"
                             "Press 'q' to quit.")
        stim.draw()
        self.win.flip()
        key = waitKeys(keyList=[sync_key, 'q'])
        if 'q' in key:
            self.close()
            return

        self.start_experiment()

        if self.eyetracker_on:
            self.start_recording_eyetracker()

        for trial in self.trials:
            trial.run()

        # Show end message
        self.display_text(
            'Well done!',
            duration=self.settings['deepmreye']['wait_after_run'],
        )

        self.close()

    def close(self):
        """Save per-frame target positions and trajectory plot, then close."""
        if self.closed:
            return

        # Save per-frame target positions as TSV
        if self.frame_data:
            df = pd.DataFrame(self.frame_data)
            tsv_path = op.join(self.output_dir,
                               self.output_str + '_target_positions.tsv')
            df.to_csv(tsv_path, sep='\t', index=False)

            # Plot target trajectory
            fig, ax = plt.subplots(figsize=(15, 5))
            ax.plot(df['onset'], df['x_deg'], label='X')
            ax.plot(df['onset'], df['y_deg'], label='Y')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Position (deg)')
            ax.set_title('Target position over time')
            ax.legend()
            fig.savefig(op.join(self.output_dir,
                                self.output_str + '_trajectory.pdf'))
            plt.close(fig)

        super().close()


# ------------------------------------------------------------------ #
#                            Entry point                              #
# ------------------------------------------------------------------ #

def _apply_debug_settings(settings):
    """Override settings for a quick debug run (~30s)."""
    dm = settings['deepmreye']
    dm['fixation']['n_locs'] = [3, 3]        # 9 instead of 100
    dm['fixation']['duration'] = 0.5
    dm['pursuit']['angles_step'] = 90         # 4 angles × 1 amp = 4 + 1 stationary
    dm['pursuit']['amplitudes_deg'] = [4]
    dm['pursuit']['duration'] = 0.5
    dm['pictures']['n_pics'] = 2
    dm['pictures']['duration'] = 1.0
    dm['instruction_duration'] = 2
    dm['wait_after_run'] = 2
    settings['window']['fullscr'] = False
    settings['window']['size'] = [800, 600]
    return settings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DeepMReye calibration experiment',
    )
    parser.add_argument('subject', type=int, help='Subject number')
    parser.add_argument('session', type=int, help='Session number')
    parser.add_argument('--settings', default='settings.yml',
                        help='Path to settings YAML file')
    parser.add_argument('--use_eyetracker', action='store_true',
                        help='Enable Eyelink eyetracker')
    parser.add_argument('--debug', action='store_true',
                        help='Short debug run (~30s, windowed)')
    parser.add_argument('--no_pictures', action='store_true',
                        help='Skip the free-viewing picture block')
    args = parser.parse_args()

    output_str = f'sub-{args.subject:02d}_ses-{args.session}_task-deepmreyecalib'
    settings_path = op.join(op.dirname(op.abspath(__file__)), args.settings)

    calib_session = DeepMReyeCalibSession(
        output_str,
        output_dir=op.join(op.dirname(op.abspath(__file__)), 'logs'),
        settings_file=settings_path,
        eyetracker_on=args.use_eyetracker,
        calibrate_eyetracker=args.use_eyetracker,
    )

    if args.debug:
        calib_session.settings = _apply_debug_settings(calib_session.settings)

    calib_session.create_trials(include_pictures=not args.no_pictures)
    calib_session.run()
    calib_session.quit()
