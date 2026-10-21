"""Select each training run's stage using validation only, before testing."""

import argparse
import csv
import json
from pathlib import Path
import shutil
from statistics import mean


def summarize_stages(rows, stage_dirs, expected_episodes=10):
    roots = {str(Path(p).resolve()) for p in stage_dirs}
    groups = {}
    for row in rows:
        checkpoint = Path(row.get('checkpoint','')).resolve()
        if row['policy'] == 'ppo' and str(checkpoint.parent) in roots:
            groups.setdefault(str(checkpoint),[]).append(row)
    summaries = []
    seed_set = None
    for checkpoint,group in sorted(groups.items()):
        seeds = [int(r['world_seed']) for r in group]
        if len(seeds) != expected_episodes or len(set(seeds)) != len(seeds):
            raise ValueError('Every checkpoint needs the expected number of distinct validation seeds')
        if seed_set is not None and seed_set != set(seeds):
            raise ValueError('Checkpoints must share the same validation seed set')
        seed_set = set(seeds)
        food = sum(int(r['food_consumed']) for r in group)
        summaries.append({
            'run':Path(checkpoint).parent.name,
            'updates':int(Path(checkpoint).stem.removeprefix('update_')),
            'checkpoint':checkpoint,
            'mean_survivors':mean(int(r['survivors']) for r in group),
            'mean_survival_ticks':mean(float(r['mean_survival_ticks']) for r in group),
            'food_energy_waste_fraction':sum(float(r['food_energy_wasted']) for r in group)/(25*food) if food else 0,
        })
    if {str(Path(r['checkpoint']).parent) for r in summaries} != roots:
        raise ValueError('A requested training run has no validation results')
    return summaries


def selection_rank(row):
    return (-row['mean_survivors'], row['food_energy_waste_fraction'],
            -row['mean_survival_ticks'], row['updates'], row['checkpoint'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes-csv',type=Path,required=True)
    parser.add_argument('--stage-dirs',type=Path,nargs='+',required=True)
    parser.add_argument('--expected-episodes',type=int,default=10)
    parser.add_argument('--best-output',type=Path,required=True)
    args = parser.parse_args()
    with args.episodes_csv.open() as source:
        rows = list(csv.DictReader(source))
    try:
        summary = summarize_stages(rows,args.stage_dirs,args.expected_episodes)
    except ValueError as exc:
        parser.error(str(exc))
    selected = []
    for root in args.stage_dirs:
        group = [r for r in summary if Path(r['checkpoint']).parent == root.resolve()]
        winner = min(group,key=selection_rank)
        winner = {**winner,'copy':str(root.parent/f'{root.name}_best.pt')}
        selected.append(winner)
    overall = min(selected,key=selection_rank)
    targets = [Path(r['copy']) for r in selected] + [args.best_output]
    if len(set(p.resolve() for p in targets)) != len(targets) or any(p.exists() for p in targets):
        parser.error('Output checkpoint paths must be distinct and must not already exist')
    for winner in selected:
        shutil.copyfile(winner['checkpoint'],winner['copy'])
    shutil.copyfile(overall['checkpoint'],args.best_output)
    folder = args.episodes_csv.parent
    with (folder/'stages.csv').open('w',newline='') as output:
        writer = csv.DictWriter(output,fieldnames=list(summary[0]))
        writer.writeheader();writer.writerows(summary)
    validation_seeds=sorted({int(r['world_seed']) for r in rows})
    (folder/'selection.json').write_text(json.dumps({
        'selected_runs':selected,'overall_validation_choice':overall,
        'overall_copy':str(args.best_output),'validation_seeds':validation_seeds,
        'criterion':'Maximize final survivors, minimize food-energy waste, maximize survival time, prefer earlier update',
        'locked_before_test':True},indent=2))
    for winner in selected:
        print(f"{winner['run']}: selected {winner['updates']} updates, "
              f"{winner['mean_survivors']:.2f}/12 validation survivors, "
              f"{100*winner['food_energy_waste_fraction']:.2f}% waste -> {winner['copy']}")
    print(f"Overall validation choice: {overall['run']} -> {args.best_output}")


if __name__=='__main__':
    main()
