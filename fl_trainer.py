import os
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
import multiprocessing

from rich.console import Console


def valid_test_client(args, client_id, valid_eval_dataloader, global_model, criterion):
    # print(f"Start validating client {client_id}")
    with torch.set_grad_enabled(False):
        global_model.eval()
        running_loss, running_acc = AverageMeter(), AverageMeter()
        total_correct = 0
        for batch_idx, (data, target) in enumerate(valid_eval_dataloader):
            data, target = data.to(args.device), target.to(args.device)
            
            output = global_model(data)
            loss = criterion(output, target, )
            
            pred = output.argmax(dim=1, keepdim=True)
            correct_pred = pred.eq(target.view_as(pred))
            correct = correct_pred.sum().item()
            total_correct += correct
            
            running_loss.update(loss.item()/ data.size(0), data.size(0))
            running_acc.update(correct / data.size(0), data.size(0))
            
        # print(f"Client {client_id} valid loss {running_loss.avg:.4f} acc {running_acc.avg:.4f}")
        return running_acc.avg
    

def train_client(args, client_id, train_dataloader, client_model, client_weight, criterion):
    
    print(f"Start training client {client_id} in {args.local_epoch} epochs")    
    client_model.train()
    optimizer = get_client_optimizer(args.client_optimizer, client_model, args.client_learning_rate, args.client_weight_decay)
    
    with torch.set_grad_enabled(True):
            
        for epoch in range(args.local_epoch):
            
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
                
                pbar.set_description(f"Client {client_id} epoch {epoch} loss {running_loss.avg:.4f} acc {running_acc.avg:.4f}")
    return running_loss, running_acc


def fed_avg_aggregator(args, global_model, client_models, client_weights, logger):
    logger.log(f"Aggregating models with FedAvg: {' '.join([f'{weight:.4f}' for weight in client_weights])}")
    
    weight_accumulator = {name: torch.zeros(params.size()).to(args.device) for name, params in global_model.state_dict().items()}

    for net_index, net in enumerate(client_models):
        for name, data in net.state_dict().items():
            weight_accumulator[name].add_(client_weights[net_index] * (data - global_model.state_dict()[name]))
            
    for name, params in global_model.state_dict().items():
        params.add_(weight_accumulator[name].to(params.dtype))

def valid_test_process(args, map_clients_id, valid_eval_dataloader, global_model, criterion, logger):
    result_acc = []
    
    for client_id, valid_eval_dl in enumerate(valid_eval_dataloader):
        acc = valid_test_client(args, client_id, valid_eval_dl, global_model, criterion)    
        logger.log(f"Client {map_clients_id[client_id]} valid acc {acc:.4f}")
        result_acc.append(acc)
    return result_acc

def train_fl(args, clients_type, train_dataloaders, valid_eval_dataloader, test_eval_dataloader, global_model, criterion, logger):
    global_model.to(args.device)
    
    num_rounds = args.comm_round
    
    logger.log("---"*20)
    logger.log(f"Start training in {num_rounds} communication rounds")
    n_train_clients = len(train_dataloaders)
    map_clients = {
        "valid": [],
        "test": [],
    }
    for key, value in clients_type.items():
        if value == "Validation":
            map_clients["valid"].append(key)
        elif value == "Test":
            map_clients["test"].append(key)
    
        
    best_valid_eval_acc = 0
    
    for com_round_id in range(num_rounds):
        select_clients = np.random.choice(n_train_clients, args.client_num_per_round, replace=False)
        
        train_sample_num = sum(len(train_dataloaders[client_id].dataset) for client_id in select_clients)

        client_weights = [len(train_dataloaders[client_id].dataset) / train_sample_num for client_id in select_clients]
        
        logger.log(f"Communication round {com_round_id} select clients {select_clients} #train sample: {train_sample_num} client_weights: {' '.join([f'{weight:.3f}' for weight in client_weights])}")
        
        client_models = [deepcopy(global_model).to(args.device) for _ in range(args.client_num_per_round)]
        
        for id_0, client_id in enumerate(select_clients):
            running_loss, running_acc = train_client(args, client_id, train_dataloaders[client_id], client_models[id_0], client_weights[id_0], criterion)
            logger.log(f"Client {client_id} train loss {running_loss.avg:.4f} acc {running_acc.avg:.4f} correct {running_acc.sum:.4f} total {running_acc.count:.4f}")
            
        fed_avg_aggregator(args, global_model, client_models, client_weights, logger)
        logger.log(f"---"*10)
        logger.log(f"Validating model for validation clients")
        val_acc = valid_test_process(args, map_clients["valid"], valid_eval_dataloader, global_model, criterion, logger)
        
        logger.log(f"Validating model for test clients")
        test_acc = valid_test_process(args, map_clients["test"], test_eval_dataloader, global_model, criterion, logger)

        logger.log(f"Communication round {com_round_id} valid acc: {' '.join([f'{acc:.3f}' for acc in val_acc])} mean: {np.mean(val_acc):.3f}")
        logger.log(f"Communication round {com_round_id} test acc: {' '.join([f'{acc:.3f}' for acc in test_acc])} mean: {np.mean(test_acc):.3f}")
        logger.log("***"*20)
        
    logger.log(f"End fl training in {num_rounds} communication rounds")
        


def run_pfl(json_config):
    init_random_seed()
    
    # load config file
    # args = json.load(open(json_config), object_hook=lambda d: SimpleNamespace(**d))
    args_dict = EasyDict(json.load(open(json_config, "r")))
    args = args_dict.fed_training
    
    wandb_instance = None
    log_name = f"./logs/{args.dataset}_{args.model}_{args.client_num_per_round}_{args.comm_round}_{args.local_epoch}_{args.lr}_{args.seed}.html"
    
    logger = Console(record=True, log_path=False, log_time=False)
    
    logger.log(f"Done loading config file: {json_config}")
    
    args.logger = logger
    
    # load dataset
    clients_type, train_datasets, train_dataloaders, valid_per_dataset, valid_eval_dataset, test_per_dataset, test_eval_dataset, valid_per_dataloader, valid_eval_dataloader, test_per_dataloader, test_eval_dataloader = get_dataset_pfl_cifar10(args)
    logger.log("Done loading dataset")
    
    logger.log(f"Clients type: {clients_type}")
    
    # load model
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}
    device = torch.device(args.device if use_cuda else "cpu") 
    args.device = device
    global_model, _, _, criterion = get_model_and_optimizer(args)
    
    logger.log("Done loading model")
    
    
    # n_train_clients, n_valid_clients, n_test_clients = len(train_datasets), len(valid_per_dataset), len(test_per_dataset)
    
    train_fl(args, clients_type, train_dataloaders, valid_eval_dataloader, test_eval_dataloader, global_model, criterion, logger)
    
    os.makedirs(os.path.dirname(log_name), exist_ok=True)
    
    logger.save_html(log_name)
    

if __name__ == "__main__":
    # get name of file json from command line
    parser = argparse.ArgumentParser(description="PyTorch CIFAR-10 Example")
    parser.add_argument('--config', type=str, default="./configs/pfl_cifar10_fedavg_052723.json", help='config file')
    # "/home/henry/learning2/ pfl/configs/pfl_cifar10_fedavg_052723.json"
    args = parser.parse_args()
    
    run_pfl(json_config=args.config) # er4fg2m356