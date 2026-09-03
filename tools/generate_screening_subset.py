"""Generate a fixed, stratified 20% KITTI train subset for Stage 1 screening.

E0 / E1 / E2 must all train on exactly the same images, so this subset is
generated once, written to a split file, and reused on every subsequent call
unless --regenerate is explicitly passed.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / 'data' / 'KITTIDataset'
IMAGESETS_DIR = DATASET_ROOT / 'ImageSets'
LABEL_DIR = DATASET_ROOT / 'training' / 'label_2'

CANDIDATE_SEEDS = [42, 123, 3407, 2026, 777]
TARGET_RATIO = 0.20
BUCKET_BOUNDS = {'near': (0.0, 20.0), 'middle': (20.0, 40.0), 'far': (40.0, float('inf'))}


def depth_bucket(depth):
    if depth < 20.0:
        return 'near'
    if depth < 40.0:
        return 'middle'
    return 'far'


def difficulty_of(height_px, occluded, truncated):
    if height_px > 40 and occluded <= 0 and truncated <= 0.15:
        return 'easy'
    if height_px > 25 and occluded <= 1 and truncated <= 0.30:
        return 'moderate'
    if height_px > 25 and occluded <= 2 and truncated <= 0.50:
        return 'hard'
    return None


def load_image_ids(split_name):
    path = IMAGESETS_DIR / f'{split_name}.txt'
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def parse_car_objects(image_id):
    """Return list of dicts: depth, height_px, occluded, truncated, difficulty."""
    label_path = LABEL_DIR / f'{image_id}.txt'
    objects = []
    for line in label_path.read_text().splitlines():
        fields = line.strip().split()
        if not fields or fields[0] != 'Car':
            continue
        truncated = float(fields[1])
        occluded = int(float(fields[2]))
        bbox_top, bbox_bottom = float(fields[5]), float(fields[7])
        height_px = bbox_bottom - bbox_top
        depth = float(fields[13])
        objects.append({
            'depth': depth,
            'bucket': depth_bucket(depth),
            'height_px': height_px,
            'occluded': occluded,
            'truncated': truncated,
            'difficulty': difficulty_of(height_px, occluded, truncated),
        })
    return objects


def build_image_index(image_ids):
    """Parse every image once; return per-image object lists."""
    index = {}
    for image_id in image_ids:
        index[image_id] = parse_car_objects(image_id)
    return index


def aggregate_stats(image_ids, index):
    num_images = len(image_ids)
    all_objects = [obj for image_id in image_ids for obj in index[image_id]]
    num_cars = len(all_objects)
    depths = [obj['depth'] for obj in all_objects]
    bucket_counts = {bucket: sum(1 for obj in all_objects if obj['bucket'] == bucket)
                      for bucket in ('near', 'middle', 'far')}
    bucket_pct = {bucket: (100.0 * count / num_cars if num_cars else 0.0)
                  for bucket, count in bucket_counts.items()}
    difficulty_counts = {level: sum(1 for obj in all_objects if obj['difficulty'] == level)
                          for level in ('easy', 'moderate', 'hard')}

    def mean(values):
        return sum(values) / len(values) if values else float('nan')

    def median(values):
        if not values:
            return float('nan')
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 0:
            return (ordered[mid - 1] + ordered[mid]) / 2.0
        return ordered[mid]

    def std(values):
        if not values:
            return float('nan')
        avg = mean(values)
        return (sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5

    return {
        'number_of_images': num_images,
        'number_of_car_objects': num_cars,
        'cars_per_image_mean': mean([len(index[image_id]) for image_id in image_ids]),
        'depth_bucket_counts': bucket_counts,
        'depth_bucket_pct': bucket_pct,
        'depth_mean': mean(depths),
        'depth_median': median(depths),
        'depth_std': std(depths),
        'difficulty_counts': difficulty_counts,
    }


def dominant_bucket(objects):
    if not objects:
        return 'no_car'
    counts = {'near': 0, 'middle': 0, 'far': 0}
    for obj in objects:
        counts[obj['bucket']] += 1
    return max(counts, key=counts.get)


def stratified_sample(image_ids, index, target_size, seed):
    groups = {}
    for image_id in image_ids:
        groups.setdefault(dominant_bucket(index[image_id]), []).append(image_id)
    rng = random.Random(seed)
    total = len(image_ids)
    allocation = {}
    for group, members in groups.items():
        allocation[group] = round(target_size * len(members) / total)
    # Fix rounding drift against target_size.
    drift = target_size - sum(allocation.values())
    largest_group = max(groups, key=lambda group: len(groups[group]))
    allocation[largest_group] += drift

    selected = []
    for group, members in groups.items():
        shuffled = members[:]
        rng.shuffle(shuffled)
        take = max(0, min(allocation.get(group, 0), len(shuffled)))
        selected.extend(shuffled[:take])
    # Guard against any residual size mismatch from clamping.
    if len(selected) < target_size:
        remaining = [image_id for image_id in image_ids if image_id not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[:target_size - len(selected)])
    elif len(selected) > target_size:
        rng.shuffle(selected)
        selected = selected[:target_size]
    return sorted(selected, key=lambda value: int(value))


def distribution_error(full_stats, subset_stats):
    return sum(abs(full_stats['depth_bucket_pct'][bucket] - subset_stats['depth_bucket_pct'][bucket])
               for bucket in ('near', 'middle', 'far'))


def format_table(full_stats, subset_stats, subset_label='Subset'):
    lines = []
    header = f"{'':<22}{'Full Train':>14}{subset_label:>16}"
    lines.append(header)
    lines.append(f"{'Images':<22}{full_stats['number_of_images']:>14}{subset_stats['number_of_images']:>16}")
    lines.append(f"{'Car objects':<22}{full_stats['number_of_car_objects']:>14}{subset_stats['number_of_car_objects']:>16}")
    for bucket, label in (('near', 'Depth 0-20 m'), ('middle', 'Depth 20-40 m'), ('far', 'Depth 40m+')):
        full_cell = f"{full_stats['depth_bucket_counts'][bucket]} ({full_stats['depth_bucket_pct'][bucket]:.1f}%)"
        sub_cell = f"{subset_stats['depth_bucket_counts'][bucket]} ({subset_stats['depth_bucket_pct'][bucket]:.1f}%)"
        lines.append(f"{label:<22}{full_cell:>14}{sub_cell:>16}")
    for level in ('easy', 'moderate', 'hard'):
        lines.append(f"{level.capitalize():<22}{full_stats['difficulty_counts'][level]:>14}{subset_stats['difficulty_counts'][level]:>16}")
    lines.append(f"{'Mean depth':<22}{full_stats['depth_mean']:>14.2f}{subset_stats['depth_mean']:>16.2f}")
    lines.append(f"{'Median depth':<22}{full_stats['depth_median']:>14.2f}{subset_stats['depth_median']:>16.2f}")
    lines.append(f"{'Std depth':<22}{full_stats['depth_std']:>14.2f}{subset_stats['depth_std']:>16.2f}")
    return '\n'.join(lines)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-out', default=str(IMAGESETS_DIR / 'train_screening_20.txt'))
    parser.add_argument('--stats-out', default=str(ROOT / 'experiment_results' / 'screening' / 'subset' / 'subset_statistics.json'))
    parser.add_argument('--target-ratio', type=float, default=TARGET_RATIO,
                         help='Fraction of the full train split to sample (default: 0.20, Stage 1).')
    parser.add_argument('--regenerate', action='store_true',
                         help='Force regeneration even if the split file already exists.')
    args = parser.parse_args()
    target_ratio = args.target_ratio

    split_out = Path(args.split_out)
    stats_out = Path(args.stats_out)
    stats_out.parent.mkdir(parents=True, exist_ok=True)

    full_ids = load_image_ids('train')
    full_index = build_image_index(full_ids)
    full_stats = aggregate_stats(full_ids, full_index)
    target_size = round(full_stats['number_of_images'] * target_ratio)
    subset_label = f'{target_ratio * 100:.0f}%'

    if split_out.exists() and not args.regenerate:
        existing_ids = [line.strip() for line in split_out.read_text().splitlines() if line.strip()]
        subset_index = {image_id: full_index[image_id] for image_id in existing_ids}
        subset_stats = aggregate_stats(existing_ids, subset_index)
        print(f'Reusing existing split file: {split_out}')
        print(f'File hash (sha256): {file_hash(split_out)}')
        print(f'Image count: {len(existing_ids)}')
        print()
        print(format_table(full_stats, subset_stats, subset_label))
        if stats_out.exists():
            print(f'\nExisting statistics file: {stats_out}')
        return

    best_seed, best_ids, best_stats, best_error = None, None, None, float('inf')
    candidate_errors = {}
    for seed in CANDIDATE_SEEDS:
        candidate_ids = stratified_sample(full_ids, full_index, target_size, seed)
        candidate_index = {image_id: full_index[image_id] for image_id in candidate_ids}
        candidate_stats = aggregate_stats(candidate_ids, candidate_index)
        error = distribution_error(full_stats, candidate_stats)
        candidate_errors[seed] = error
        if error < best_error:
            best_seed, best_ids, best_stats, best_error = seed, candidate_ids, candidate_stats, error

    split_out.parent.mkdir(parents=True, exist_ok=True)
    split_out.write_text('\n'.join(best_ids) + '\n')

    warnings = []
    for bucket in ('near', 'middle', 'far'):
        diff = abs(full_stats['depth_bucket_pct'][bucket] - best_stats['depth_bucket_pct'][bucket])
        if diff >= 5.0:
            warnings.append(f'Depth bucket "{bucket}" differs by {diff:.2f} percentage points (>= 5.0 threshold).')

    print(f'Original train images: {full_stats["number_of_images"]}')
    print(f'Subset images: {len(best_ids)}')
    print(f'Sampling ratio: {100.0 * len(best_ids) / full_stats["number_of_images"]:.1f}%')
    print(f'Candidate seeds tried: {CANDIDATE_SEEDS}')
    print(f'Candidate distribution errors (sum |pct diff| over near/middle/far): {candidate_errors}')
    print(f'Selected seed: {best_seed} (SCREENING_SPLIT_SEED)')
    print()
    print(format_table(full_stats, best_stats, subset_label))
    if warnings:
        print('\nWARNINGS:')
        for warning in warnings:
            print(f'  - {warning}')
    else:
        print('\nNo distribution warnings (all depth-bucket deltas < 5.0 percentage points).')

    report = {
        'target_ratio': target_ratio,
        'target_size_formula': f'round(original_train_images * {target_ratio})',
        'original_train_images': full_stats['number_of_images'],
        'screening_subset_images': len(best_ids),
        'sampling_ratio_pct': 100.0 * len(best_ids) / full_stats['number_of_images'],
        'candidate_seeds': CANDIDATE_SEEDS,
        'candidate_distribution_errors': candidate_errors,
        'selected_seed': best_seed,
        'split_file': str(split_out),
        'split_file_sha256': file_hash(split_out),
        'warnings': warnings,
        'full_train_statistics': full_stats,
        'screening_subset_statistics': best_stats,
    }
    with stats_out.open('w') as stream:
        json.dump(report, stream, indent=2)
    print(f'\nSaved split file: {split_out}')
    print(f'Saved statistics: {stats_out}')


if __name__ == '__main__':
    main()
