# Experiment 12: food availability before deaths

Date: October 22, 2026.

Replayed the same 30 worlds (seeds 500000–500029) for 2,000 ticks with the
unchanged Experiment 12 best checkpoint, recording each agent's final 30
pre-action states before death, including energy, position, food visibility,
action, and movement outcome. All 114 per-agent death events and their exact
ticks matched the original audit. Checkpoint SHA-256 stayed unchanged.

## Verified observations

- Across 3,420 pre-death agent-ticks, no food was visible in 56.7% of states.
- Visible food was within the optimistic remaining energy budget in 39.4% of states.
- No food anywhere on the map was within that budget in 15.1% of states.
- Only 15 of 114 dying agents saw no food throughout their entire final 30 ticks.
- 44 of 114 dying agents skipped food directly underfoot at least once in that window.
- Food underfoot was skipped 204 times: 124 at energy above 20, 56 at energy
  11–20, 15 at energy 6–10, and 9 at energy 1–5.
- Of those skipped meals, 203 actions were movement requests and one was STAY.
- Four EAT actions occurred in these death windows, each at energy 4.
- There were 238 blocked movement requests and only nine foodless STAY actions.
- One agent even had food underfoot at energy 1 immediately before dying.

## Interpretation and limits

The observations rule out a blanket explanation that there was never nearby
food before deaths: many agents had opportunities to eat, including directly
underfoot. Delaying eating until very low energy is a plausible contributor,
and simply encouraging more movement is not the clearest next change.
However, this audit selects dying agents and their final 30 ticks; it does not
measure eating decisions across all surviving trajectories or prove the cause
of individual deaths. The four low-energy meals are not a population-wide
estimate of the policy's eating threshold.

Visibility is the policy's actual 5x5 square. Distances are Manhattan lower
bounds ignoring obstacles, occupied cells, competition, and future regrowth.
A food cell qualifies as within budget when distance is strictly less than
current energy, leaving time to move and then EAT. Qualifying does not guarantee
successful access. No food within budget now also does not rule out future regrowth.

A useful next controlled test is to replay these same worlds with a temporary
inference override that eats food underfoot at energy 25 or less, comparing
survival and waste with the unchanged model. That would test whether earlier
eating helps before changing rewards or retraining; it has not been run here.

## Reproduce

```bash
python -m experiments.death_food_audit
```

The command requires the original `runs/experiment_12_long_survival/deaths.csv`
and an absent `runs/experiment_12_death_food_audit` output directory, refusing
to overwrite existing results. Local output includes `death_windows.csv`,
`protocol.json`, and `summary.json`; no model weights or simulation rules change.
