import torch
import numpy as np
import argparse
import time
import util
from engine import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--save',type=str,default=None,help='save path')
parser.add_argument('--load_path',type=str,default=None,help='load path')
args = parser.parse_args()




def main():
    #load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    args.num_nodes = len(sensor_ids)
    args.gcn_bool = True
    args.addaptadj = True

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None



    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, 0., device)


    if args.load_path is None:
        raise ValueError
    else:
        engine.model.load_state_dict(torch.load(args.load_path, map_location = args.device))
        engine.model.eval()


    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]
    output_gates = []
    output_ind = []

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds, gate, ind_out = engine.model(testx, gate_out = True)
        outputs.append(preds.squeeze())
        output_gates.append(gate)
        output_ind.append(ind_out)

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]

    yhat_gates = torch.cat(output_gates, dim = 0)[:realy.size(0),...]
    yhat_ind = torch.cat(output_ind, dim = 0)[:realy.size(0),...]
    yhat_ind = scaler.inverse_transform(yhat_ind)
    tmp = yhat_gates.argmax(dim = -1)
    print("Gates!")
    for i in range(3):
        print((tmp == i).sum())
        cur_ind = yhat_ind[:,:,-1,i]
        metrics = util.metric(cur_ind, realy[:,:,-1])
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(args.seq_length, metrics[0], metrics[1], metrics[2]))
        print('On average over {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'.format(args.seq_length, *util.metric(yhat_ind[...,i], realy)))


    amae = []
    amape = []
    armse = []
    results = {'prediction': [], 'ground_truth':[], 'gate':[],}
    from copy import deepcopy as cp
    for i in range(args.seq_length):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        results['prediction'].append(cp(pred).cpu().numpy())
        results['ground_truth'].append(cp(real).cpu().numpy())
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(args.seq_length, np.mean(amae),np.mean(amape),np.mean(armse)))
    results['prediction'] = np.asarray(results['prediction'])
    results['ground_truth'] = np.asarray(results['ground_truth'])
    results['gate'] = np.asarray(cp(yhat_gates).cpu().numpy())
    results['indi'] = np.asarray(cp(yhat_ind).cpu().numpy())
    if args.save is not None:
        np.savez_compressed(args.save+"_prediction.npz", **results)


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
