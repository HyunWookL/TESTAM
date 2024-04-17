import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = None, type = str, required = True)
parser.add_argument('--device', default = -1, type = int)
parser.add_argument('--exp_id', dest='i', default = 0, help = 'experiment identifier for who want to run experiment multiple time')
args = parser.parse_args()

device = 'cuda:' + str(args.device)

BATCH_DICT = {
        'EXPY-TKY': 8,
        'METR-LA': 16,
        'PEMS-BAY': 32
        }

if args.device < 0:
    print("Device ID is not Specified!")
    print("Continue training with CPU...")
    device = 'cpu'

if args.dataset not in BATCH_DICT.keys():
    raise ValueError("We do not have default setting for the custom dataset. Please specify the datsets from METR-LA, PEMS-BAY, or EXPY-TKY")

if not os.path.exists('experiment/{}_{}'.format(args.dataset, args.i)):
    os.makedirs('experiment/{}_{}'.format(args.dataset, args.i))
if args.dataset == 'EXPY-TKY':
    log = "python -u train.py --batch_size 8 --seq_length 6 --dropout 0.0 --seed -1 --save ./experiment/{}_{}/TESTAM --data ./data/{} --adjdata ./data/{}/adj_mx.pkl --device {}"
else:
    batch_size = BATCH_DICT[args.dataset]
    log = "python -u train.py --batch_size 16 --dropout 0.0 --seed -1 --save ./experiment/{}_{}/TESTAM --data ./data/{} --adjdata ./data/{}/adj_mx.pkl --device {} --n_warmup_steps 4000"
print(log.format(args.dataset, args.i, args.dataset, args.dataset, device, args.dataset, args.i))
os.system(log.format(args.dataset, args.i, args.dataset, args.dataset, device, args.dataset, args.i))


