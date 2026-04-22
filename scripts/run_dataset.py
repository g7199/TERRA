"""End-to-end: preprocess -> train backbone -> TF-IDFRank eval."""
import argparse, os, sys, subprocess, shlex

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True)
ap.add_argument('--backbone', required=True,
                choices=['GRU4Rec', 'SASRec', 'FMLPRec', 'BERT4Rec', 'DuoRec', 'BSARec'])
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--gpu', type=str, default='0')
ap.add_argument('--skip_preprocess', action='store_true')
ap.add_argument('--skip_train', action='store_true')
ap.add_argument('--data_dir', default='data')
ap.add_argument('--proc_dir', default='data/processed')
ap.add_argument('--ckpt_dir', default='backbone/output')
ap.add_argument('--results_csv', default='results/tfidfrank.csv')
ap.add_argument('--epochs', type=int, default=200)
args = ap.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = args.gpu

seq_file = os.path.join(args.proc_dir, f'{args.dataset}.txt')
mapping_file = os.path.join(args.proc_dir, f'{args.dataset}_mapping.json')
raw_reviews = os.path.join(args.data_dir, f'reviews_{args.dataset}_5.json.gz')

if not args.skip_preprocess and not (os.path.exists(seq_file) and os.path.exists(mapping_file)):
    subprocess.check_call([sys.executable, 'preprocess/preprocess.py',
                           '--dataset', args.dataset,
                           '--raw_dir', args.data_dir,
                           '--out_dir', args.proc_dir], env=env)

os.makedirs(args.ckpt_dir, exist_ok=True)
train_name = f'{args.backbone}_{args.dataset}_seed{args.seed}'
ckpt = os.path.join(args.ckpt_dir, f'{train_name}.pt')

if not args.skip_train and not os.path.exists(ckpt):
    model_type = args.backbone.lower()
    cmd = [sys.executable, 'main.py',
           '--data_name', args.dataset,
           '--data_dir', f'../{args.proc_dir}/',
           '--output_dir', f'../{args.ckpt_dir}',
           '--model_type', model_type,
           '--train_name', train_name,
           '--gpu_id', args.gpu,
           '--seed', str(args.seed),
           '--epochs', str(args.epochs)]
    if model_type == 'bert4rec':
        cmd += ['--mask_ratio', '0.2']
    if model_type == 'bsarec':
        cmd += ['--c', '5', '--alpha', '0.7']
    subprocess.check_call(cmd, cwd='backbone', env=env)

if not os.path.exists(ckpt):
    raise FileNotFoundError(f'Missing checkpoint: {ckpt}')

subprocess.check_call([sys.executable, 'tfidfrank/run_tfidfrank.py',
                       '--dataset', args.dataset,
                       '--backbone', args.backbone,
                       '--seed', str(args.seed),
                       '--ckpt', ckpt,
                       '--seq_file', seq_file,
                       '--mapping_file', mapping_file,
                       '--reviews_file', raw_reviews,
                       '--out_csv', args.results_csv,
                       '--gpu', args.gpu], env=env)

print(f'[done] {args.results_csv}')
