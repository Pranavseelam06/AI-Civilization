# Experiment log

Each entry is one sentence; results are exploratory and evaluation seeds differ unless a matched comparison is stated.

- **Experiment 1 — October 20, 2026:** A three-update, unmasked PPO smoke test ran successfully but showed directional bias and performed below random movement in later evaluation.
- **Experiment 2 — October 20, 2026:** Twenty unmasked PPO updates with +5 per living tick increased corner concentration and performed worse than Experiment 1 on matched evaluation seeds.
- **Experiment 3 — October 20, 2026:** Twenty masked PPO updates eliminated boundary violations but did not solve corner concentration or improve survival.
- **Experiment 4 — October 20, 2026:** Moving the +5 survival bonus to episode end and training for 200 updates produced strong food-direction responses and 3.2 survivors per episode across ten new worlds, with both reward timing and training duration changed.
- **Experiment 5 — October 20, 2026:** Adding explicit EAT and training for 200 updates produced 5.97 survivors per episode while wasting 48.8% of consumed food energy on evaluation seeds 60000–60029.
- **Experiment 6 — October 20, 2026:** Rewarding energy restored divided by 25 for 200 updates reduced survival to 1.97 agents per episode and increased wasted food energy to 58.0% on the same seeds as Experiment 5.
- **Experiment 7 — October 20, 2026:** After 200 updates with signed food rewards, Experiment 7 wasted 16.8% of food energy versus Experiment 5’s 48.5% across 30 matched worlds, with average survival of 392 versus 383 ticks but fewer final survivors (4.97 versus 5.70).

## Additional diagnostic experiments

- **Food-direction probe — October 20, 2026:** Controlled food relocation exposed weak food responsiveness in Experiment 3 and strong direction-specific responses in Experiment 4.
- **Forced exploration — October 20, 2026:** Replacing Experiment 4's foodless STAY choices with random legal movement across 30 matched worlds yielded 302 rather than 305 ticks of mean survival and provided no survival benefit.
