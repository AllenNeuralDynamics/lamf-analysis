---
name: circular-statistics
description: Summarize phase-locked peaks or troughs on a repeating stimulus cycle.
---

# Circular statistics

- Use a half-open, explicit grid `[0, T)`. Including a duplicated `T` endpoint
  treats the next stimulus onset as part of the current cycle and distorts
  wrap-around latencies.
- Declare the cycle period `T` and anchor: `t=0 -> 0 rad`. Convert event times
  with `theta = 2*pi*t/T`.
- For selected phases `theta`, report `n`, circular mean
  `angle(mean(exp(1j*theta)))`, resultant length `R = abs(mean(exp(1j*theta)))`,
  and the mean phase converted back to `[0, T)`.
- Summarize positive and negative peak phases separately. Include only cells
  meeting the corresponding directional criterion.
- For a circular Gaussian or wrapped latency distance, use
  `((t - center + T/2) % T) - T/2`, not ordinary linear subtraction.

## Repeated clean-image responses

- For 750 ms image cycles, retain only non-change, non-omission image flashes
  with `flashes_since_change > 4` and `flashes_since_omission > 2`.
- Extract one cycle per clean flash, average across flashes, then center the
  mean trace: `d_obs = r_obs - mean(r_obs)`. Compute empirical `A_pos`,
  `A_neg`, and signed peak latencies from this same trace and grid.
- Use the cyclic-shift permutation test for significance; the Gaussian fit is
  descriptive only. Fit a positive component only if `p_pos < alpha` and a
  negative component only if `p_neg < alpha`; otherwise leave that component
  absent (`NaN`).
- A centered strong positive response can make below-mean points significant,
  and vice versa. `both_sig` is an operational label, not evidence of a
  genuinely biphasic response. Inspect `sign_index` and
  `min(A_pos, A_neg) / max(A_pos, A_neg)` before categorizing shape.

## Descriptive circular-Gaussian fit

- Fit `d_fit = G_pos - G_neg`, with
  `G(t) = amp * exp(-0.5 * circ_dist(t, center)^2 / sigma^2)`. Constrain
  amplitudes nonnegative and initialize centers at empirical signed latencies.
- Bound widths. For a 0.75 s, 20 Hz cycle, use `sigma` in `[0.025, 0.25]` s
  (half a sample through one third of the cycle). Persist flags for either
  bound: boundary-piled widths are not reliable width estimates.
- Save component amplitudes, wrapped centers, widths, `fit_R2`, fit RMSE, and
  `log(sigma_pos / sigma_neg)` only when both components exist.
