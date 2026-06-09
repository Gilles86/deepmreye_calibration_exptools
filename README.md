# DeepMREye Calibration (exptools2)

PsychoPy calibration script for [DeepMReye](https://github.com/DeepMReye/DeepMReye), built on the [exptools2](https://github.com/VU-Cog-Sci/exptools2) framework. Ported from the original [Psychtoolbox version](https://github.com/DeepMReye/Calibration).

## Tasks

The full run takes approximately **5 minutes 7 seconds**:

| Task | Trials | Duration | Subtotal |
|------|--------|----------|----------|
| Fixation | 100 (10×10 grid) | 1.5s each | ~150s |
| Smooth pursuit | 73 (24 directions × 3 speeds + 1 stationary) | 1.5s each | ~110s |
| Free viewing | 10 images | 3s each | ~30s |
| Instructions + end screen | 4 | — | ~17s |
| **Total** | | | **~307s** |

## Usage

```bash
python deepmreye_calib.py <subject> <session>
```

- The script waits for a **`t` keypress** to start (i.e. the MRI scanner trigger).
- Press **`q`** at any time to quit.
- `--use_eyetracker` enables the Eyelink eyetracker.
- `--no_pictures` skips the free-viewing picture block.

At the Spinoza 7T (projector where only the bottom of the screen is visible):

```bash
python deepmreye_calib.py 1 1 --settings settings_spinoza.yml
```

### Debug mode

Short run (~17s) in a windowed display for testing:

```bash
python deepmreye_calib.py 1 1 --debug
```

## Requirements

- Python 3.9+
- [PsychoPy](https://www.psychopy.org/)
- [exptools2](https://github.com/VU-Cog-Sci/exptools2)
- NumPy, Pandas, Matplotlib

## Configuration

Settings live in YAML files passed via `--settings` (default `settings.yml`).
`settings_spinoza.yml` is preconfigured for the Spinoza 7T projector. Key sections:

- `window` — screen resolution, fullscreen
- `monitor` — physical screen width (cm) and viewing distance (cm)
- `mri` — trigger key (`sync: t`), TR, simulated mode
- `deepmreye` — task durations, grid size, pursuit parameters, image count

### Partially-visible screens (`visible_fraction`)

When only the bottom portion of the projected screen is visible (e.g. the head
coil clips the top), set `deepmreye.visible_fraction` to the visible fraction
(`0.7` = bottom 70%). The calibration window's **height** is then shrunk to the
visible band and **all stimuli are shifted down** so everything stays in view;
the horizontal extent is left unchanged, so horizontal eccentricity matches the
full-screen case. The logged target positions reflect the actual on-screen
(shifted) coordinates, so the DeepMReye labels remain correct. Default `1.0` =
full screen visible.

## MRI acquisition tips

A few things to keep in mind when acquiring DeepMReye training data (courtesy of M. Nau):

- **Eyes must be visible in the images.** Make sure the FOV covers the eyes.
- **Phase-encoding direction: P>>A.** This stretches the eyes towards the front of the head rather than squishing them, which is better for DeepMReye.
- **More subjects > more data per subject.** When training the model, scanning additional subjects tends to help more than acquiring extra runs per subject.
- **Training data does not need to come from study participants.** You can scan lab members or students with your specific sequence and use that data to train the model.
- **Consider your task.** What kind of eye movements do you expect in your study? That should inform which calibration conditions are most important to include.
- **Use a pretrained model first.** You can test one of the [pretrained DeepMReye models](https://github.com/DeepMReye/DeepMReye) on pilot data before collecting dedicated training data.

## Output

Files are saved to `logs/`:

| File | Content |
|------|---------|
| `*_events.tsv` | Trial onsets, MRI pulses, key responses |
| `*_target_positions.tsv` | Per-frame target (x, y) in degrees |
| `*_trajectory.pdf` | Target trajectory plot |
| `*_frames.pdf` | Frame interval plot |
| `*_expsettings.yml` | Full settings used |

## References

Frey M.\*, Nau M.\*, Doeller C.F. (2021). Magnetic resonance-based eye tracking using deep neural networks. *Nature Neuroscience*.
