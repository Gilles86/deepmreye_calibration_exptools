# DeepMREye Calibration (exptools2)

PsychoPy calibration script for [DeepMReye](https://github.com/DeepMReye/DeepMReye), built on the [exptools2](https://github.com/VU-Cog-Sci/exptools2) framework. Ported from the original [Psychtoolbox version](https://github.com/DeepMReye/Calibration).

## Tasks

1. **Fixation** -- fixation cross jumps to 100 pseudorandom screen locations (10x10 grid)
2. **Smooth pursuit** -- fixation cross moves along a pseudorandom walk (24 directions x 3 speeds)
3. **Free viewing** -- participant freely views images

## Requirements

- Python 3.9+
- [PsychoPy](https://www.psychopy.org/)
- [exptools2](https://github.com/VU-Cog-Sci/exptools2)
- NumPy, Pandas, Matplotlib

## Usage

```bash
python deepmreye_calib.py --subject 01 --run 1
```

Press `t` to start (scanner trigger), `q` to quit at any time.

### Debug mode

Short run (~17s) in a windowed display for testing:

```bash
python deepmreye_calib.py --subject test --run 1 --debug
```

## Configuration

All settings are in `settings.yml`. Key sections:

- `window` -- screen resolution, fullscreen
- `monitor` -- physical screen width (cm) and viewing distance (cm)
- `mri` -- trigger key, TR, simulated mode
- `deepmreye` -- task durations, grid size, pursuit parameters, image count

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
