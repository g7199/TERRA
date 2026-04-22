"""Amazon 2014 5-core reviews -> LOO sequences + id mapping."""
import argparse, os, json, gzip
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True)
ap.add_argument('--raw_dir', default='data')
ap.add_argument('--out_dir', default='data/processed')
ap.add_argument('--min_k', type=int, default=5)
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
raw_path = os.path.join(args.raw_dir, f'reviews_{args.dataset}_5.json.gz')
if not os.path.exists(raw_path):
    raise FileNotFoundError(
        f'Missing {raw_path}. Download from http://jmcauley.ucsd.edu/data/amazon/')

user_item_time = defaultdict(list)
with gzip.open(raw_path, 'rt') as f:
    for line in f:
        r = json.loads(line)
        u, a, t = r.get('reviewerID'), r.get('asin'), r.get('unixReviewTime')
        if u is None or a is None or t is None: continue
        user_item_time[u].append((a, int(t)))

print(f'[{args.dataset}] raw users: {len(user_item_time)}')


def kcore(uit, k):
    while True:
        uc = {u: len(v) for u, v in uit.items()}
        ic = defaultdict(int)
        for items in uit.values():
            for a, _ in items: ic[a] += 1
        drop_u = {u for u, c in uc.items() if c < k}
        drop_i = {a for a, c in ic.items() if c < k}
        if not drop_u and not drop_i: return uit
        new = {}
        for u, items in uit.items():
            if u in drop_u: continue
            kept = [(a, t) for a, t in items if a not in drop_i]
            if len(kept) >= k: new[u] = kept
        uit = new


filtered = kcore(user_item_time, args.min_k)
print(f'[{args.dataset}] after {args.min_k}-core: users={len(filtered)}')

final = {u: [a for a, _ in sorted(items, key=lambda x: x[1])] for u, items in filtered.items()}

user_sorted = sorted(final.keys())
user_to_int = {u: i + 1 for i, u in enumerate(user_sorted)}
all_items = sorted({a for items in final.values() for a in items})
asin_to_int = {a: i + 1 for i, a in enumerate(all_items)}

seq_path = os.path.join(args.out_dir, f'{args.dataset}.txt')
with open(seq_path, 'w') as f:
    for u in user_sorted:
        seq = ' '.join(str(asin_to_int[a]) for a in final[u])
        f.write(f"{user_to_int[u]} {seq}\n")

mapping_path = os.path.join(args.out_dir, f'{args.dataset}_mapping.json')
with open(mapping_path, 'w') as f:
    json.dump({
        'asin_to_int': asin_to_int,
        'int_to_asin': {i: a for a, i in asin_to_int.items()},
        'user_to_int': user_to_int,
        'int_to_user': {i: u for u, i in user_to_int.items()},
    }, f)

print(f'[{args.dataset}] {len(final)} users, {len(asin_to_int)} items -> {seq_path}, {mapping_path}')
