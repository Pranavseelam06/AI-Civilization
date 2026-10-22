# Experiment 12: temporary earlier-eating test

Date: October 22, 2026.

Compared the frozen Experiment 12 best checkpoint with and without an inference
rule that chooses EAT whenever an agent has food underfoot and energy at most
25, across the same 30 worlds (seeds 500000–500029), each for 2,000 ticks.
Both variants sample the model action first; only the intervention then
replaces eligible actions. There are no additional random draws. World and
policy RNG seeds start matched; trajectories and later random-number usage
can diverge after interventions. No reward, training, checkpoint, or viewer
behavior changed. The normal control exactly reproduced every episode metric
from the preceding long-survival audit, and checkpoint SHA-256 stayed unchanged.

| Metric | Unchanged policy | Temporary rule |
| --- | ---: | ---: |
| Mean final survivors / 12 | 8.20 | 8.80 |
| Mean lifespan capped at 2,000 ticks | 1,653.6 | 1,704.7 |
| Worlds retaining all 12 agents | 0 / 30 | 1 / 30 |
| Total food consumed | 22,651 | 23,415 |
| Food-energy waste | 0 | 0 |

The override changed 14,782 sampled actions. It improved final survivor counts
in 17 worlds, tied in five, and worsened them in eight. The paired mean gain
was 0.60 survivors and 51.0 ticks of capped lifespan. A 95% percentile bootstrap
interval for the survivor gain is [-0.20, 1.43] survivors, using
10,000 paired-world resamples with RNG seed 20261022; it includes zero.

The result is a modest, variable exploratory improvement, not a reliable fix
established by this test. These worlds were already inspected to motivate the
intervention, so independent evaluation is still needed. Zero waste is expected
for the forced meals because eating at energy 25 or less fully uses the 25-unit
food gain, but altered trajectories could still have changed ordinary policy
meals; measured waste was zero across all meals in both variants.

The rule does not teach the model to eat sooner and has not been enabled in
the viewer. A useful next step is to test this same fixed threshold on new
world seeds before deciding whether to change training or keep a rule.

## Status: rejected and removed

On October 22, 2026, the user rejected the modest gain and requested removal.
The temporary rule, comparison runner, associated tests, and generic override
hook were removed. The rule was never enabled in the viewer or trained into
checkpoint weights. Existing local results in
`runs/experiment_12_early_eating` remain as the experiment record.
