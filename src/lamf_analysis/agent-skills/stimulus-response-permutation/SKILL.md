---
name: stimulus-response-permutation
description: Generate onset-aligned interpolated stimulus-response traces and efficient cyclic-shift permutation tests for neural time series.
---

# Stimulus response with cyclic-shift permutations

Use this skill when analyzing neural responses aligned to repeated stimuli,
especially when recordings have variable frame timing or when response
significance must be tested without breaking the within-cycle temporal
structure.

## Align traces with LAMF ETR

Use `lamf_analysis.ophys.general_utilities.event_triggered_response` rather
than manually interpolating event windows. Request interpolation explicitly and
use an explicit output sampling rate.

```python
aligned = event_triggered_response(
    data=pd.DataFrame({"time": timestamps, "events": trace}),
    t="time",
    y="events",
    event_times=stimulus_times,
    t_start=request_start,
    t_end=request_end,
    output_sampling_rate=20.0,
    include_endpoint=True,
    output_format="wide",
    interpolate=True,
)
```

- Work on an explicit target grid. For a 750 ms image cycle at 20 Hz, use the
  half-open grid `np.arange(0.0, 0.75, 0.05)`: 15 bins from 0.00 through
  0.70 s. Do not include 0.75 s because it is the next image onset.
- Request a padded ETR interval, then select the desired coordinates from the
  returned ETR grid. ETR checks for support on both sides of an event, so a
  strictly post-onset request can otherwise become missing at its boundary.
- Validate returned coordinates with `np.allclose`; never infer time from the
  number of samples.
- For an isolated missing source sample, interpolate that source trace before
  ETR. Do not silently replace an entirely invalid trace.


## Define cycle-based response statistics

For cycles shaped `(neurons, trials, time)`:

1. Compute each neuron’s mean trace across trials.
2. Subtract its cycle mean from the mean trace.
3. Derive `T_obs = max(abs(deviation))`, directional positive/negative
   amplitudes, and peak latencies from the same explicit grid.
4. Use directional null exceedances for positive and negative peaks and apply
   the chosen multiple-comparison correction to the primary two-sided
   statistic.

Keep response metrics in physical units and use z-scored quantities only when
the reference distribution is clearly specified.

## Efficient cyclic-shift null

The null shifts each trial independently around its own stimulus cycle. This
preserves trial amplitudes and cycle structure while disrupting a
stimulus-locked temporal pattern.

### Required vectorization pattern

- Build the small rolled lookup tensor once per neuron block:
  `(trial, shift, neuron, time)`.
- Draw **one** `(permutation, trial)` shift array per block and apply it to all
  neurons in the block. Thus every neuron is evaluated against the same random
  shifts, reducing random-number overhead and making comparisons coherent.
- Batch permutations so the temporary selected tensor
  `(permutation, trial, neuron, time)` stays within a fixed RAM budget.
- Stream exceedance counts for observed `T_obs`, positive amplitude, and
  negative amplitude. Do not retain or send `neurons × permutations × 3` null
  arrays when only p-values are needed.
- Recompute full null traces only for the small number of cells shown in
  shuffle-validation figures.

```python
# cycles_block: (neurons, trials, time), float32
rolled = np.stack(
    [np.roll(cycles_block, shift=s, axis=2) for s in range(n_time)],
    axis=2,
).transpose(1, 2, 0, 3)  # (trials, shifts, neurons, time)

shifts = rng.integers(0, n_time, size=(batch_size, n_trials), dtype=np.int16)
picked = rolled[trial_index, shifts]  # (permutations, trials, neurons, time)
deviation = picked.mean(axis=1, dtype=np.float32) - cycle_mean[None, :, None]
t_exceedances += np.count_nonzero(
    np.max(np.abs(deviation), axis=2) >= t_obs[None, :], axis=0
)
```

Use a plus-one corrected p-value:

```python
p_value = (1 + exceedances) / (1 + num_permutations)
```

## Parallel scheduling on one machine

Parallelize neuron blocks within a session with a bounded process pool. For
multiple independent sessions (for example, mice), parallelizing **across
sessions** is beneficial on a single machine, but do not nest large process
pools.


## Verification checklist

- Confirm all expected grid values are present and finite.
- Test that serial and parallel block execution produce identical exceedance
  counts with a fixed seed.
- Compare streaming exceedance counts once against a retained-null reference on
  synthetic data before removing retained null arrays.
- Verify p-value directionality and the half-open cycle boundary.
- Confirm heatmap axes end at the right bin edge (0.75 s for the example
  above), while response sample coordinates end at 0.70 s.
