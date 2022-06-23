# About
Stereo 3-band parametric equalizer for jack with fader and mono controls.

# Usage
Estripe runs an OSC server at port 5510, and can receive messages at the
following paths:

| Path | Msg Type | Parameter |
| ---- | -------- | ---------- |
| /fader | _float_ | volume in dB |
| /mono | _int_ | 0 for stereo, 1 for mono |
| /fil/k/type | _string_ | filter type |
| /fil/k/freq | _float_ | frequency in Hz |
| /fil/k/gain | _float_ | gain in dB |
| /fil/k/q | _float_ | Q value |

Where _k_ ranges from 1 to 3 and specifies which of the three filters you are
changing the parameter for. The 'filter type' message can be _off_ to disable
the filter, or _lowshelf_, _lowpass_, _peak_, _hishelf_ and _hipass_. The filter
coefficients are from [RBJ's filter
cookbook](https://shepazu.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html).
