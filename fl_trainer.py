# from utils import get_dataset, get_model, get_optimizer, get_loss, get_trainer
import argparse
import json
import torch
import numpy as np
import tqdm
from copy import deepcopy


from utils import init_random_seed, AverageMeter
from easydict import EasyDict
from datasets import get_dataset_pfl_cifar10
from pfl_models import get_model_and_optimizer, get_client_optimizer

def train_client(args, client_id, train_dataloader, client_model, client_weight, criterion):
    
    print(f"Start training client {client_id} in {args.local_epoch} epochs")    
    client_model.train()
    optimizer = get_client_optimizer(args.client_optimizer, client_model, args.client_learning_rate, args.client_weight_decay)
    
    
    with torch.set_grad_enabled(True):
            
        for epoch in range(args.local_epoch):
            # print(f"Training client {client_id} epoch {epoch}")
            
            pbar = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            running_loss, running_acc = AverageMeter(), AverageMeter()
            total_correct = 0
            for batch_idx, (data, target) in pbar:
                data, target = data.to(args.device), target.to(args.device)
                
                
                optimizer.zero_grad()
                
                output = client_model(data)
                loss = criterion(output, target, )
                loss.backward()
                optimizer.step()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct_pred = pred.eq(target.view_as(pred))
                correct = correct_pred.sum().item()
                total_correct += correct
                
                running_loss.update(loss.item()/ data.size(0), data.size(0))
                running_acc.update(correct / data.size(0), data.size(0))
                
                
                pbar.set_description(
                    f"Client {client_id} epoch {epoch} loss {running_loss.avg:.4f} acc {running_acc.avg:.4f}"
                )
    # pass

# def fed_avg_aggregator(args, global_model, client_models, client_weights):
#     print(f"Aggregating models with FedAvg {client_weights}")
    
#     weight_accumulator = {}
#     for name, params in global_model.state_dict().items():
#         weight_accumulator[name] = torch.zeros(params.size()).to(device)

#     for net_index, net in enumerate(client_models):
#         for name, data in net.state_dict().items():
#             weight_accumulator[name].add_(
#                 client_weights[net_index] * (data - global_model.state_dict()[name])
#             )
            
#     for name, params in global_model.state_dict().items():
#         update_per_layer = weight_accumulator[name]
#         if params.type() != update_per_layer.type():
#             params.add_(update_per_layer.to(torch.int64))
#         else:
#             params.add_(update_per_layer)

def fed_avg_aggregator(args, global_model, client_models, client_weights):
    print(f"Aggregating models with FedAvg {client_weights}")
    
    weight_accumulator = {name: torch.zeros(params.size()).to(args.device) for name, params in global_model.state_dict().items()}

    for net_index, net in enumerate(client_models):
        for name, data in net.state_dict().items():
            weight_accumulator[name].add_(client_weights[net_index] * (data - global_model.state_dict()[name]))
            
    for name, params in global_model.state_dict().items():
        params.add_(weight_accumulator[name].to(params.dtype))


def train_fl(args, train_dataloaders, valid_eval_dataloader, global_model, criterion):
    global_model.to(args.device)
    
    num_rounds = args.comm_round
    
    print("---"*20)
    print(f"Start training in {num_rounds} communication rounds")
    n_train_clients = len(train_dataloaders)
    
    best_valid_eval_acc = 0
    for com_round_id in range(num_rounds):
        select_clients = np.random.choice(n_train_clients, args.client_num_per_round, replace=False)
        
        train_sample_num = sum(len(train_dataloaders[client_id].dataset) for client_id in select_clients)

        client_weights = [len(train_dataloaders[client_id].dataset) / train_sample_num for client_id in select_clients]
        
        print(f"Communication round {com_round_id} select clients {select_clients} #train sample: {train_sample_num} client_weights: {client_weights}")
        
        client_models = [deepcopy(global_model).to(args.device) for _ in range(args.client_num_per_round)]
        
        for id_0, client_id in enumerate(select_clients):
            train_client(args, client_id, train_dataloaders[client_id], client_models[id_0], client_weights[id_0], criterion)
            
        fed_avg_aggregator(args, global_model, client_models, client_weights)
        
    
    print("End fl training")
        


def run_pfl(json_config):
    # load config file
    # args = json.load(open(json_config), object_hook=lambda d: SimpleNamespace(**d))
    args_dict = EasyDict(json.load(open(json_config, "r")))
    args = args_dict.fed_training
    
    wandb_instance = None
    print("Done loading config file")
    
    # load dataset
    train_datasets, train_dataloaders, valid_per_dataset, valid_eval_dataset, test_per_dataset, test_eval_dataset, valid_per_dataloader, valid_eval_dataloader, test_per_dataloader, test_eval_dataloader = get_dataset_pfl_cifar10(args)
    print("Done loading dataset")
    
    # load model
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}
    device = torch.device(args.device if use_cuda else "cpu") 
    args.device = device
    global_model, _, _, criterion = get_model_and_optimizer(args)
    
    
    print("Done loading model")
    
    
    # n_train_clients, n_valid_clients, n_test_clients = len(train_datasets), len(valid_per_dataset), len(test_per_dataset)
    
    train_fl(args, train_dataloaders, valid_eval_dataloader, global_model, criterion)
    
    # # load optimizerç
    # optimizer = get_optimizer(config, model)

    # # load loss function
    # criterion = get_loss(config)

    # load trainer
    
    
    # trainer = get_trainer(config, model, optimizer, criterion, train_dataset, test_dataset)

    # # run trainer
    # trainer.run()
    

if __name__ == "__main__":
    # get name of file json from command line
    parser = argparse.ArgumentParser(description="PyTorch CIFAR-10 Example")
    parser.add_argument('--config', type=str, default="./configs/pfl_cifar10_fedavg_052723.json", help='config file')
    # "/home/henry/learning2/pfl/configs/pfl_cifar10_fedavg_052723.json"
    args = parser.parse_args()
    
    run_pfl(json_config=args.config)