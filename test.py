import torch
import numpy as np
import argparse
import time, os
import util
from engine import trainer

parser = argparse.ArgumentParser()                                                      
parser.add_argument('--device',type=str,default='cuda:0',help='')                       
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')          
parser.add_argument('--adjdata',type=str,default=None,help='adj data path')                                                                                
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')    
parser.add_argument('--out_dim',type=int,default=1,help='')                         
parser.add_argument('--nhid',type=int,default=32,help='')                               
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')              
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')          
parser.add_argument('--batch_size',type=int,default=64,help='batch size')               
parser.add_argument('--save',type=str,default=None,help='save path') 
parser.add_argument('--load_path', type = str, default = None)                          
                                                                                        
args = parser.parse_args()                                                              
                                                                                        
                                                                                        
def count_parameters(model):                                                            
    return sum(p.numel() for p in model.parameters() if p.requires_grad)                
                                                                                        
                                                                                        
def main():                                                                             
    device = torch.device(args.device)                                                  
    if args.adjdata:
        if os.path.exists(args.adjdata):
          sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype) 
          args.num_nodes = len(sensor_ids)                                                
        else:
          print("Invalid File Path; utliize user-provided args.num_nodes")
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)                                                                                
    scaler = dataloader['scaler']                                                       
                                                                                        
    print(args)                                                                         
                                                                                        
    engine = trainer(scaler, args.in_dim, args.out_dim, args.num_nodes, args.nhid, 0., device)
    if args.load_path is None:
        raise ValueError
    else:
        engine.model.load_state_dict(torch.load(args.load_path, map_location = args.device))
        engine.model.eval()


    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,:args.out_dim,:,:]
    output_gates = []
    output_ind = []

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds, gate, ind_out = engine.model(testx, gate_out = True)
        outputs.append(preds)
        output_gates.append(gate)
        output_ind.append(ind_out)

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]

    yhat_gates = torch.cat(output_gates, dim = 0)[:realy.size(0),...].permute(0,3,1,2,4).contiguous()
    yhat_ind = torch.cat(output_ind, dim = 0)[:realy.size(0),...].permute(0,3,1,2,4).contiguous()
    yhat_ind = scaler.inverse_transform(yhat_ind)
    tmp = yhat_gates.argmax(dim = -1)
    print("Gates!")
    for i in range(yhat_gates.size(-1)):
        print((tmp == i).sum())
        cur_ind = yhat_ind[:,:,:,-1,i]
        metrics = util.metric(cur_ind, realy[...,-1])
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(realy.size(-1), metrics[0], metrics[1], metrics[2]))
        print('On average over {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'.format(realy.size(-1), *util.metric(yhat_ind[...,i], realy)))


    amae = []
    amape = []
    armse = []
    results = {'prediction': [], 'ground_truth':[], 'gate':[],}
    from copy import deepcopy as cp
    for i in range(realy.size(-1)):
        pred = scaler.inverse_transform(yhat[...,i])
        real = realy[...,i]
        results['prediction'].append(cp(pred).cpu().numpy())
        results['ground_truth'].append(cp(real).cpu().numpy())
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(realy.size(-1), np.mean(amae),np.mean(amape),np.mean(armse)))
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
