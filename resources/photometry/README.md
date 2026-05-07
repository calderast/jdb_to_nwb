# Photometry resources

Configuration files that drive photometry processing and NWB metadata. Each file
is described below. To add a new indicator or device, update the relevant files
as described in each section.

---

## `photometry_devices.yaml`

Hardware catalog: excitation sources (LEDs and their wavelengths), optic fibers,
and photodetectors used by the Berke Lab. Referenced by name in session metadata
and used to populate NWB device metadata.

Add a new device entry here if you use a new optic fiber, or we change photodetectors/excitation sources.


## `virus_info.yaml`

Running list of all indicators used for fiber photometry (and any opsins used for
optogenetic experiments). Each entry provides the construct name, a description,
and the manufacturer.

Add a new indicator here whenever a new virus is used. The `name` field must match
the name used in `photometry_mappings.yaml` and in session metadata.

**Current indicators:**

| Name | Construct | Sensor type |
|---|---|---|
| dLight1.3b | AAVDJ-CAG-dLight1.3b | Dopamine (green) |
| dLight3.8 | AAV-DJ-CAG-dLight3.8 | Dopamine (green) |
| rDA3m (AAV9) | AAV9-hsyn-rDA3m | Dopamine (red) |
| rDA3m (rAAV) | rAAV-hsyn-rDA3m | Dopamine (red) |
| gACh4h | AAV-hSyn-ACh3.8 | Acetylcholine (green) |


## `photometry_mappings.yaml`

Maps each indicator to:
- **Signal and reference wavelengths** (nm) — used to assign loaded channels to
  the correct wavelength key in the `PhotometrySignalBundle`
- **Default processing preset** — which preset from `processing_presets.yaml` is
  applied automatically unless overridden in session metadata
- **Compatible excitation sources** — which LEDs from `photometry_devices.yaml`
  are valid for this indicator

**Current mappings:**

| Indicator | Signal | Reference | Default preset |
|---|---|---|---|
| dLight1.3b | 470 nm | 405 nm (isosbestic) | `dlight_isosbestic` |
| dLight3.8 | 470 nm | 405 nm (isosbestic) | `dlight_isosbestic` |
| gACh4h | 470 nm | 405 nm (ratiometric) | `gach_ratiometric` |
| rDA3m (AAV9) | 565 nm | — | `classic_filtering` |
| rDA3m (rAAV) | 565 nm | — | `classic_filtering` |

Add a new entry here whenever a new indicator is added to `virus_info.yaml`.


## `processing_presets.yaml`

Defines the signal processing pipeline. Each preset specifies one method per step:

```
smoothing → baseline → normalization → correction
```

**Available methods per step:**

| Step | Methods |
|---|---|
| smoothing | `rolling_mean`, `lowpass`, `none` |
| baseline | `airpls`, `double_exp`, `highpass`, `none` |
| normalization | `median_zscore`, `mean_zscore`, `none` |
| correction | `isosbestic_lasso`, `ratiometric`, `none` |

**Current presets:**

| Preset | Pipeline | Default for |
|---|---|---|
| `dlight_isosbestic` | rolling mean → airPLS → median zscore → isosbestic Lasso | dLight1.3b, dLight3.8 |
| `dlight_isosbestic_tight_baseline` | rolling mean → airPLS (lambda=1e5 for tighter baseline) → median zscore → isosbestic Lasso | — |
| `dlight_isosbestic_double_exp` | rolling mean → double exp → median zscore → isosbestic Lasso | — |
| `dlight_isosbestic_no_baseline` | rolling mean → none → median zscore → isosbestic Lasso | — |
| `gach_ratiometric` | lowpass → highpass → mean zscore → ratiometric | gACh4h |
| `classic_filtering` | lowpass → highpass → mean zscore | rDA3m (AAV9), rDA3m (rAAV) |

**Customizing processing in session metadata:**

Use `processing_presets` to select a non-default preset for a given indicator:
```yaml
processing_presets:
  dLight1.3b: dlight_isosbestic_double_exp
```

Use `processing_overrides` to tune method parameters without changing the preset:
```yaml
processing_overrides:
  baseline:
    lambda: 1.0e+9        # airPLS: higher = smoother baseline
    mode: dff             # airpls/double_exp: "subtract" or "dff" = (F-F0)/F0
  smoothing:
    window_fraction: 0.05 # rolling_mean: window = Fs * this value
  correction:
    alpha: 0.00001        # isosbestic_lasso: lower = tighter Lasso fit
```

New presets can be added directly in this file and referenced by name in metadata.
Presets support a `param_overrides` key to bake in non-default parameters without
affecting the shared `method_defaults`:
```yaml
my_custom_preset:
  smoothing: rolling_mean
  baseline: airpls
  normalization: median_zscore
  correction: isosbestic_lasso
  param_overrides:
    baseline:
      lambda: 1.0e+7
```

---

## `crop_photometry.ipynb`

Notebook to help find the correct `phot_end_time_mins` value for a session where
the photometry recording needs to be cropped (e.g. the fiber was unplugged before
the recording ended). Cropping applies to all photometry signals via the `phot_end_time_mins` field in session metadata.
