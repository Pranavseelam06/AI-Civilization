# Experiment log

Each entry is one sentence; results are exploratory and evaluation seeds differ unless a matched comparison is stated.

- **Experiment 1 — October 20, 2026:** A three-update, unmasked PPO smoke test ran successfully but showed directional bias and performed below random movement in later evaluation.
- **Experiment 2 — October 20, 2026:** Twenty unmasked PPO updates with +5 per living tick increased corner concentration and performed worse than Experiment 1 on matched evaluation seeds.
- **Experiment 3 — October 20, 2026:** Twenty masked PPO updates eliminated boundary violations but did not solve corner concentration or improve survival.
- **Experiment 4 — October 20, 2026:** Moving the +5 survival bonus to episode end and training for 200 updates produced strong food-direction responses and 3.2 survivors per episode across ten new worlds, with both reward timing and training duration changed.
- **Experiment 5 — October 20, 2026:** Adding explicit EAT and training for 200 updates produced 5.97 survivors per episode while wasting 48.8% of consumed food energy on evaluation seeds 60000–60029.
- **Experiment 6 — October 20, 2026:** Rewarding energy restored divided by 25 for 200 updates reduced survival to 1.97 agents per episode and increased wasted food energy to 58.0% on the same seeds as Experiment 5.
- **Experiment 7 — October 20, 2026:** After 200 updates with signed food rewards, Experiment 7 wasted 16.8% of food energy versus Experiment 5’s 48.5% across 30 matched worlds, with average survival of 392 versus 383 ticks but fewer final survivors (4.97 versus 5.70).

- **Experiment 8 — October 20, 2026:** After 200 updates with hunger-based eating rewards, food-energy waste fell to 0.5% across 30 matched worlds, but 4.50 agents survived versus Experiment 5’s 6.10, with eating probabilities responding strongly to hunger in a controlled probe.

- **Experiment 9 — October 20, 2026:** After 200 updates with smooth positive hunger rewards, agents reverted to near-unconditional eating and averaged 2.37 survivors with 57.1% food-energy waste versus Experiment 8’s 4.77 survivors and 0.5% waste across 30 matched worlds; the change was reverted to Experiment 8 and its checkpoint archived.

- **Experiment 10 — October 20, 2026:** After 200 updates with negative hunger rewards clipped to zero, mean final survivors rose to 5.13 from Experiment 8’s 3.70 across 30 matched worlds, but food-energy waste rose to 46.3% from 0.7%; it was later archived when Experiment 8 was restored.

- **Experiment 11 — October 20, 2026:** Adding a gradual missed-meal penalty below 30 energy to restored-energy food rewards produced 3.67 survivors versus Experiment 6’s 2.17 across 30 matched worlds, but wasted 52.6% of food energy and remained below Experiment 8’s 4.03 survivors; it was then archived and the code restored to Experiment 8.

- **Experiment 12 — October 20, 2026:** Extending Experiment 8’s unchanged training to 1,000 updates with checkpoints every 100 selected the 700-update stage on validation worlds, which averaged 11.23 survivors versus Experiment 8’s 4.50 with no observed food-energy waste on 30 separate test worlds.

- **Experiment 13 — October 20, 2026:** Repeating the unchanged longer-training setup with three new training seeds produced validation-selected models averaging 11.07, 10.60, and 9.93 survivors versus Experiment 8’s 4.00 on 30 separate matched test worlds, with food-energy waste no greater than 0.006%.

## Additional diagnostic experiments

- **Food-direction probe — October 20, 2026:** Controlled food relocation exposed weak food responsiveness in Experiment 3 and strong direction-specific responses in Experiment 4.
- **Forced exploration — October 20, 2026:** Replacing Experiment 4's foodless STAY choices with random legal movement across 30 matched worlds yielded 302 rather than 305 ticks of mean survival and provided no survival benefit.

- **Experiment 12 — Longer survival audit — October 22, 2026:** Extending the frozen best checkpoint to 2,000 ticks on 30 new worlds reduced average survivors from 11.37 at tick 500 to 8.20 at tick 2,000 with zero food-energy waste, revealing continued mortality beyond the training horizon.

- **Experiment 12 — Pre-death food audit — October 22, 2026:** Replaying the same 30 extended worlds exactly reproduced 114 deaths and found that 44 dying agents skipped underfoot food during their final 30 ticks, suggesting delayed eating as a contributor without establishing causality.

- **Experiment 12 — Temporary earlier-eating test — October 22, 2026:** Forcing underfoot meals at energy ≤25 on the same 30 extended worlds raised mean survivors from 8.20 to 8.80 with zero waste, winning 17 pairs and losing eight, a modest exploratory gain whose bootstrap interval includes zero; the user rejected the rule and its implementation was removed on October 22, 2026.

- **Food carrying — Pipeline smoke test — October 22, 2026:** Adding opt-in one-slot storage and PICKUP passed pickup/eating/reset/replay tests and a two-update training/save/load/evaluation smoke test, while all 13 old experiments remained runnable and the Experiment 12 long-survival control matched exactly, with no carrying-policy improvement established.

- **Experiment 14 — October 22, 2026:** Extending the unchanged one-slot carrying setup from 200 to 1,000 updates raised mean survivors from 4.63 to 10.47 out of 12 and reduced food-energy waste from 4.34% to zero on 30 reused evaluation worlds, with 29 paired wins and one tie; the updated checkpoint is active and the original is archived.

- **Experiment 14 — Sample versus argmax — October 22, 2026:** Comparing unchanged weights on 30 reused worlds found sampling averaged 10.47 survivors versus argmax’s 5.87 with zero food-energy waste for both, while argmax lost 27 paired worlds and had more contested moves, so sampling remains the default.
