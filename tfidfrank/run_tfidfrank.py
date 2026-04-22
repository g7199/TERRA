"""Leak-free TF-IDFRank evaluation. Val α* selection, then test report."""
import sys, os, argparse, json, gzip, time, csv
import numpy as np
import torch
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True)
ap.add_argument('--backbone', required=True,
                choices=['GRU4Rec','SASRec','FMLPRec','BERT4Rec','DuoRec','BSARec'])
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--ckpt', required=True)
ap.add_argument('--seq_file', required=True)
ap.add_argument('--mapping_file', required=True)
ap.add_argument('--reviews_file', required=True)
ap.add_argument('--out_csv', default='results/tfidfrank.csv')
ap.add_argument('--svd_dim', type=int, default=500)
ap.add_argument('--window', type=int, default=5, help='W: number of recent items for content profile')
ap.add_argument('--gpu', type=str, default='0')
args = ap.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'backbone'))

ALPHAS = [round(i * 0.05, 2) for i in range(21)]
EVAL_KS = [5, 10, 20]


def load_sequences(path):
    user_seq = []
    max_item = 0
    with open(path) as f:
        for line in f:
            toks = line.strip().split(' ')
            items = [int(x) for x in toks[1:]]
            user_seq.append(items)
            if items:
                max_item = max(max_item, max(items))
    return user_seq, max_item


def get_val_data(user_seq):
    data, hist = [], {}
    for uid, items in enumerate(user_seq):
        if len(items) < 3: continue
        data.append((uid, items[-2], len(items) - 2))
        hist[uid] = set(items[:-2])
    return data, hist


def get_test_data(user_seq):
    data, hist = [], {}
    for uid, items in enumerate(user_seq):
        if len(items) < 3: continue
        data.append((uid, items[-1], len(items) - 1))
        hist[uid] = set(items[:-1])
    return data, hist


def build_item_reviews(reviews_path, asin2int, user2int, user_seq, exclude_val):
    exclude = set()
    for uid, seq in enumerate(user_seq):
        if len(seq) < 3: continue
        exclude.add((uid, seq[-1]))
        if exclude_val: exclude.add((uid, seq[-2]))
    item_texts = defaultdict(list)
    with gzip.open(reviews_path, 'rt') as f:
        for line in f:
            r = json.loads(line)
            asin = r.get('asin'); text = r.get('reviewText', ''); reviewer = r.get('reviewerID', '')
            if asin not in asin2int or not text.strip(): continue
            iid = asin2int[asin]
            if reviewer in user2int and (user2int[reviewer] - 1, iid) in exclude:
                continue
            item_texts[iid].append(text)
    n_items = max(asin2int.values()) + 1
    return [" ".join(item_texts[i])[:5000] if i in item_texts else "" for i in range(n_items)]


def build_profile(docs, svd_dim):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    v = TfidfVectorizer(max_features=10000, stop_words="english", min_df=3, max_df=0.3)
    X = v.fit_transform(docs)
    svd = TruncatedSVD(n_components=svd_dim, random_state=42)
    Xr = svd.fit_transform(X).astype(np.float32)
    norm = np.linalg.norm(Xr, axis=1, keepdims=True) + 1e-8
    return torch.from_numpy(Xr / norm)


BACKBONE_DEFAULTS = {
    'gru4rec':  {'num_attention_heads': 2, 'gru_hidden_size': 64},
    'sasrec':   {'num_attention_heads': 1},
    'fmlprec':  {'num_attention_heads': 2},
    'bert4rec': {'num_attention_heads': 2, 'mask_ratio': 0.2},
    'duorec':   {'num_attention_heads': 2, 'tau': 1.0, 'lmd': 0.1, 'lmd_sem': 0.1, 'ssl': 'us_x', 'sim': 'dot'},
    'bsarec':   {'num_attention_heads': 1, 'c': 5, 'alpha': 0.7},
}


def get_backbone_scores(user_seq, ckpt_path, max_item, model_type, device, for_val=False):
    from model import MODEL_DICT
    import argparse as _ap
    d = dict(
        model_type=model_type, max_seq_length=50, hidden_size=64,
        num_hidden_layers=2, hidden_act='gelu',
        attention_probs_dropout_prob=0.5, hidden_dropout_prob=0.5,
        initializer_range=0.02,
        item_size=max_item + 1, num_users=len(user_seq) + 1,
        batch_size=256, cuda_condition=True, no_cuda=False,
    )
    d.update(BACKBONE_DEFAULTS[model_type])
    ns = _ap.Namespace(**d)
    model = MODEL_DICT[model_type](args=ns)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state: state = state['state_dict']
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    item_emb = model.item_embeddings.weight
    n = len(user_seq)
    all_scores = torch.zeros(n, item_emb.shape[0], dtype=torch.float32)
    with torch.no_grad():
        for s in range(0, n, 256):
            e = min(s + 256, n)
            batch = []
            for uid in range(s, e):
                seq = user_seq[uid][:-2] if for_val else user_seq[uid][:-1]
                pad = [0] * max(0, 50 - len(seq))
                batch.append((pad + seq)[-50:])
            inp = torch.tensor(batch, dtype=torch.long, device=device)
            out = model(inp)[:, -1, :]
            all_scores[s:e] = torch.matmul(out, item_emb.T).cpu()
    del model; torch.cuda.empty_cache()
    return all_scores


def evaluate(eval_data, user_history, scores, item_prof, train_seqs, device, window):
    ci = item_prof.to(device)
    n_items = ci.shape[0]
    results = {a: {k: {'hits': [], 'ndcgs': []} for k in EVAL_KS} for a in ALPHAS}
    for uid, gt, _ in eval_data:
        s = scores[uid].clone().to(device)[:n_items]
        for h in user_history.get(uid, set()):
            if h < n_items: s[h] = -1e9
        s[0] = -1e9
        seq = train_seqs.get(uid, [])
        recent = seq[-window:]
        valid = [r for r in recent if 0 <= r < n_items]
        if valid:
            cs = torch.matmul(ci[valid], ci.T).mean(dim=0)
        else:
            cs = torch.zeros(n_items, device=device)
        mask = s > -1e8
        if not mask.any(): continue
        cf = (s - s[mask].min()) / (s[mask].max() - s[mask].min() + 1e-8); cf[~mask] = -1e9
        if cs[mask].max() > cs[mask].min():
            cc = (cs - cs[mask].min()) / (cs[mask].max() - cs[mask].min() + 1e-8)
        else:
            cc = torch.zeros_like(cs)
        cc[~mask] = -1e9
        for a in ALPHAS:
            _, topk = torch.topk((1 - a) * cf + a * cc, max(EVAL_KS))
            ranked = topk.cpu().tolist()
            for k in EVAL_KS:
                tk = ranked[:k]
                if gt in tk:
                    results[a][k]['hits'].append(1)
                    results[a][k]['ndcgs'].append(1.0 / np.log2(tk.index(gt) + 2))
                else:
                    results[a][k]['hits'].append(0)
                    results[a][k]['ndcgs'].append(0.0)
    return results


def select_alpha(val_results):
    best_n10, best_alpha = -1, 0.0
    for a in ALPHAS:
        n10 = np.mean(val_results[a][10]['ndcgs'])
        if n10 > best_n10:
            best_n10 = n10
            best_alpha = a
    return best_alpha


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()

    with open(args.mapping_file) as f:
        mp = json.load(f)
    asin2int = mp['asin_to_int']
    user2int = mp['user_to_int']

    user_seq, max_item = load_sequences(args.seq_file)
    val_data, val_hist = get_val_data(user_seq)
    test_data, test_hist = get_test_data(user_seq)
    train_val  = {uid: s[:-2] for uid, s in enumerate(user_seq) if len(s) >= 3}
    train_test = {uid: s[:-1] for uid, s in enumerate(user_seq) if len(s) >= 3}

    print(f'[{args.dataset}/{args.backbone}/s{args.seed}] users={len(user_seq)}, items={max_item}, val={len(val_data)}, test={len(test_data)}')

    t_prof = time.time()
    val_docs  = build_item_reviews(args.reviews_file, asin2int, user2int, user_seq, exclude_val=True)
    test_docs = build_item_reviews(args.reviews_file, asin2int, user2int, user_seq, exclude_val=False)
    val_prof  = build_profile(val_docs,  args.svd_dim)
    test_prof = build_profile(test_docs, args.svd_dim)
    print(f'  leak-free profiles built in {time.time()-t_prof:.0f}s')

    t_sc = time.time()
    val_scores  = get_backbone_scores(user_seq, args.ckpt, max_item, args.backbone.lower(), device, for_val=True)
    test_scores = get_backbone_scores(user_seq, args.ckpt, max_item, args.backbone.lower(), device, for_val=False)
    print(f'  backbone scoring in {time.time()-t_sc:.0f}s')

    vr = evaluate(val_data,  val_hist,  val_scores,  val_prof,  train_val,  device, args.window)
    best_a = select_alpha(vr)
    tr = evaluate(test_data, test_hist, test_scores, test_prof, train_test, device, args.window)

    base_n10 = float(np.mean(tr[0.0][10]['ndcgs']))
    best_n10 = float(np.mean(tr[best_a][10]['ndcgs']))
    imp_pct = (best_n10 / base_n10 - 1) * 100 if base_n10 > 0 else 0.0
    row = {
        'dataset': args.dataset, 'backbone': args.backbone, 'seed': args.seed,
        'best_alpha': f'{best_a:.2f}',
        'ndcg10_base':  f'{base_n10:.4f}',
        'ndcg10_best':  f'{best_n10:.4f}',
        'ndcg5':        f'{float(np.mean(tr[best_a][5]["ndcgs"])):.4f}',
        'ndcg20':       f'{float(np.mean(tr[best_a][20]["ndcgs"])):.4f}',
        'hr5':          f'{float(np.mean(tr[best_a][5]["hits"])):.4f}',
        'hr10':         f'{float(np.mean(tr[best_a][10]["hits"])):.4f}',
        'hr20':         f'{float(np.mean(tr[best_a][20]["hits"])):.4f}',
        'improve_pct':  f'{imp_pct:+.2f}',
        'elapsed_sec':  f'{time.time()-t0:.1f}',
    }
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    write_header = not os.path.exists(args.out_csv)
    with open(args.out_csv, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)
    print('[result]', row)


if __name__ == '__main__':
    main()
