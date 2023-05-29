import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from utils import init_random_seed


def split_data_cifar100():
    # https://arxiv.org/pdf/2003.08082.pdf
    pass 

# using pytorch dataset class to load data for federated learning

def generate_dataset_clients(label_train_clients, n_clients=100, partition='hetero', partition_alpha=0.1, test_data=True): # hetero/ homo
    # label_train_clients; pair value (label, global_index from dataset)
    y_label = np.array([l for l, _ in label_train_clients])
    n_classes = len(np.unique(y_label))
    n_sample = len(y_label)

    if partition == 'hetero':
        min_required_size = 10
        net_dataidx_map = {}
        min_size = 0
        while min_size < min_required_size:
            idx_batch = [[] for _ in range(n_clients)]
            # for each class in the dataset
            for k in range(n_classes):
                idx_k = np.where(y_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(partition_alpha, n_clients))
                ## Balance
                proportions = np.array([p*(len(idx_j) < n_sample/ n_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            
            
        for j in range(n_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = [label_train_clients[idj][1]  for idj in idx_batch[j]]
            
    elif partition == "homo":
        idxs = np.random.permutation(n_sample)
        batch_idxs = np.array_split(idxs, n_clients)
        global_batch_idxs = []
        
        for j in range(n_clients):
            global_batch_idx = [label_train_clients[idj][1]  for idj in idx_batch[j]]
            global_batch_idxs.append(global_batch_idx)
            
        net_dataidx_map = {i: global_batch_idxs[i] for i in range(n_clients)}

    return net_dataidx_map

    total_sample = 0
    for j in range(n_clients):
        client_data_indices = idx_batch[j]
        # print(f"Client {j}: {len(client_data_indices)} samples")
        
        labels = y_label[client_data_indices]
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        
        total_sample += len(client_data_indices)
        print(f"Client {j}: #data: {np.sum(label_counts)} {dict(zip(unique_labels, label_counts))} ")
        # print(net_dataidx_map[j])
        # print("--------" * 10)
    print(f"total_sample: {total_sample}")
    
    if test_data:
        for j in range(n_clients):
            client_data_indices = idx_batch[j]
        # print(f"Client {j}: {len(client_data_indices)} samples")
        
            labels = y_label[client_data_indices]
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            
            total_sample += len(client_data_indices)
            # print(net_dataidx_map[j])
            # count value >= 50000 from global_batch_idxs[j]
            val_sample = np.count_nonzero(np.array(net_dataidx_map[j]) >= 50000)
            per_sample = np.count_nonzero(np.array(net_dataidx_map[j]) < 50000)
            print(f"Client {j}: #data: {np.sum(label_counts)} {dict(zip(unique_labels, label_counts))} ")
            print(f"val_sample: {val_sample} per_sample: {per_sample}")
            print("--------" * 10)
            # for jindex in global_batch_idxs[j]:
                
    return net_dataidx_map

def get_dataset_pfl_cifar10(args):
    
    # Cross-device case
    # train client, validation client, test client
    # Each validation/test client’s local examples are split into two local sets: a personalization set and an evaluation set
    
    _SHUFFLE_BUFFER_SIZE = 418  # The largest client has at most 418 examples.
    # EMNIST has 3400 clients. We split into 2500/400/500 for train/validation/test.
    TOTAL_CLIENT = 100
    # EMNIST has 100 clients. We split into 70/10/20 for train/validation/test.
    # EMNIST has 3400 clients. We split into 2500/400/500 for train/validation/test.
    
    cifar10_transform = transforms.Compose(
        [   transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5 ,0.5)),
        ]
    )
    transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    
    # Load CIFAR10 dataset
    cifar10_trainset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform_train, download=True)
    cifar10_testset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform_test, download=True)
    
    # combine train and test dataset
    cifar10_data = torch.utils.data.ConcatDataset([cifar10_trainset, cifar10_testset])
    
    # divide data into train clients and (validation + test) clients
    # 60% train clients, 40% (validation + test) clients
    # train_clients, val_test_data = torch.utils.data.random_split(cifar10_data, [int(len(cifar10_data) * 0.6), int(len(cifar10_data) * 0.4)])
    
    n_train_samples = int(len(cifar10_trainset) * 0.6)
    train_indices = np.random.choice(len(cifar10_trainset), n_train_samples, replace=False)
    
    val_test_indices = list(set(range(len(cifar10_trainset))) - set(train_indices))
   
    label_train_clients = [ (cifar10_trainset.targets[i], i) for i in train_indices]
    
    label_val_test_clients = [ (cifar10_trainset.targets[i], i) for i in val_test_indices] + [ (cifar10_testset.targets[i], len(cifar10_trainset) + i) for i in range(len(cifar10_testset))]

    DATA_RATIOS = (
        (0.7, 0.10, 0.2), # ratio of train, validation, test clien (train/ per/ per)
        (0, 0.3, 0.7) # ratio of train, validation, test client (0/ eval/ eval)
    )
    
    _NUM_TRAIN_CLIENTS = int(TOTAL_CLIENT * DATA_RATIOS[0][0])
    _NUM_VALID_CLIENTS = int(TOTAL_CLIENT * DATA_RATIOS[0][1])
    _NUM_TEST_CLIENTS = int(TOTAL_CLIENT * DATA_RATIOS[0][2])
    
    train_data_clients = generate_dataset_clients(label_train_clients, n_clients=_NUM_TRAIN_CLIENTS, partition='hetero', partition_alpha=0.5, test_data=False)
    val_test_clients = generate_dataset_clients(label_val_test_clients, n_clients=_NUM_VALID_CLIENTS + _NUM_TEST_CLIENTS, partition='hetero', partition_alpha=0.5, test_data=True)

    logger = args.logger
    
    train_datasets, train_loaders = [], []
    for idx, (_, client) in enumerate(train_data_clients.items()):
        client_dataset = torch.utils.data.Subset(cifar10_data, client)
        train_datasets.append(client_dataset)
        train_loaders += [torch.utils.data.DataLoader(client_dataset, batch_size=64, shuffle=True, num_workers=8)]
        logger.log(f"Train client number: {idx} has {len(client_dataset)} samples")
        
    per_val_datasets, per_val_loaders = [], [] # validation client
    eval_test_datasets, eval_test_loaders = [], [] # test client
    
    valid_per_dataset, valid_eval_dataset = [], []
    test_per_dataset, test_eval_dataset = [], []
    
    valid_per_dataloader, valid_eval_dataloader = [], []
    test_per_dataloader, test_eval_dataloader = [], []
    
    clients_type = {id: "Train" for id in range(_NUM_TRAIN_CLIENTS)}
    
    
    val_client_id = np.random.choice(len(val_test_clients), _NUM_VALID_CLIENTS, replace=False)
    # clients_type.update({id: "Validation" for id in val_client_id})
    
    total_valid_test = {
        "valid_per": 0,
        "valid_eval" : 0,
        "test_per": 0,
        "test_eval": 0
    }
    
    
    
    for idx, (_, client_data_idx) in enumerate(val_test_clients.items()):
        if idx in val_client_id:
            
            clients_type.update({_NUM_TRAIN_CLIENTS + idx: "Validation"})
            
            # get list of value < 50000 from client_data_idx
            per_dataset_index = [i for i in client_data_idx if i < 50000]
            eval_dataset_index = [i for i in client_data_idx if i >= 50000]
            
            per_dataset = torch.utils.data.Subset(cifar10_data, per_dataset_index)
            valid_per_dataset.append(per_dataset)
            valid_per_dataloader += [torch.utils.data.DataLoader(per_dataset, batch_size=64, shuffle=True, num_workers=8)]

            eval_dataset = torch.utils.data.Subset(cifar10_data, eval_dataset_index)
            valid_eval_dataset.append(eval_dataset)
            valid_eval_dataloader += [torch.utils.data.DataLoader(eval_dataset, batch_size=64, shuffle=True, num_workers=8)]
            
            total_valid_test["valid_eval"] += len(eval_dataset)
            total_valid_test["valid_per"] += len(per_dataset)
            logger.log(f"Validation client : {_NUM_TRAIN_CLIENTS + idx} has per: {len(per_dataset)} samples, eval: {len(eval_dataset)} samples")
                
        else:
            clients_type.update({_NUM_TRAIN_CLIENTS + idx: "Test"})
            
            # get list of value < 50000 from client_data_idx
            per_dataset_index = [i for i in client_data_idx if i < 50000]
            eval_dataset_index = [i for i in client_data_idx if i >= 50000]

            per_dataset = torch.utils.data.Subset(cifar10_data, per_dataset_index)
            test_per_dataset.append(per_dataset)
            test_per_dataloader += [torch.utils.data.DataLoader(per_dataset, batch_size=64, shuffle=True, num_workers=8)]
            
            eval_dataset = torch.utils.data.Subset(cifar10_data, eval_dataset_index)
            test_eval_dataset.append(eval_dataset)
            test_eval_dataloader += [torch.utils.data.DataLoader(eval_dataset, batch_size=64, shuffle=True, num_workers=8)]
            total_valid_test["test_per"] += len(per_dataset)
            total_valid_test["test_eval"] += len(eval_dataset)
            logger.log(f"Test client : {_NUM_TRAIN_CLIENTS + idx} has per: {len(per_dataset)} samples, eval: {len(eval_dataset)} samples")
            
    logger.log(len(valid_per_dataset), len(valid_eval_dataset), len(test_per_dataset), len(test_eval_dataset) )
    logger.log(total_valid_test)
    
    return clients_type, train_datasets, train_loaders, valid_per_dataset, valid_eval_dataset, test_per_dataset, test_eval_dataset, valid_per_dataloader, valid_eval_dataloader, test_per_dataloader, test_eval_dataloader
    
    # import IPython
    # IPython.embed()
    exit(0)
    
    
    train_sample = len(cifar10_trainset)
    test_sample = len(cifar10_testset)
    print(train_sample, test_sample)
    
    # train, validation, test = torch.utils.data.random_split(cifar10_transform, [train_sample * 0.7, train_sample * 0.1, train_sample * 0.2])
    # divide the dataset into train, validation, test with ratio 0.7, 0.1, 0.2
    train_size = int(train_sample * 0.7)
    val_size = int(train_sample * 0.1)
    test_size = train_sample - train_size - val_size

    train_indices, val_indices, test_indices = torch.utils.data.random_split(range(train_sample), [train_size, val_size, test_size])

    train_targets = np.array([cifar10_trainset.targets[i] for i in train_indices])
    val_targets = np.array([cifar10_trainset.targets[i] for i in val_indices])
    test_targets = np.array([cifar10_trainset.targets[i] for i in test_indices])


    # divide the train dataset into different clients with non.iid dataset in federated learning setup
    #   - non.iid dataset: each client has different dataset
    
    train_clients = generate_dataset_clients(train_targets, n_clients=_NUM_TRAIN_CLIENTS, partition='hetero', partition_alpha=0.5)
    valid_clients_per = generate_dataset_clients(val_targets, n_clients=_NUM_VALID_CLIENTS, partition='hetero', partition_alpha=0.5)
    test_clients_per = generate_dataset_clients(test_targets, n_clients=_NUM_TEST_CLIENTS, partition='hetero', partition_alpha=0.5)
    
    
    valid_indices_eval, test_indices_eval = torch.utils.data.random_split(range(test_sample), [int(val_size * 0.3), int(val_size * 0.7)])
    
    valid_clients_eval = generate_dataset_clients(val_targets, n_clients=_NUM_VALID_CLIENTS, partition='hetero', partition_alpha=0.5)
    test_clients_eval = generate_dataset_clients(test_targets, n_clients=_NUM_TEST_CLIENTS, partition='hetero', partition_alpha=0.5)
    
    # print(net_dataidx_map)
    # exit(0)
    
    
    
    
    
    
    # Set the root directory where the datasets will be saved
    root = "./data"

    # Transformations to apply to the datasets
    emnist_transform = transforms.Compose(
        [   transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    

    
    mnist_transform = transforms.Compose(
        [   transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    

    
    
    # Load EMNIST dataset
    # emnist_trainset = torchvision.datasets.EMNIST(root=root, split='balanced', train=True, transform=emnist_transform, download=True)
    # emnist_testset = torchvision.datasets.EMNIST(root=root, split='balanced', train=False, transform=emnist_transform, download=True)

    # # Load MNIST dataset
    # mnist_trainset = torchvision.datasets.MNIST(root=root, train=True, transform=mnist_transform, download=True)
    # mnist_testset = torchvision.datasets.MNIST(root=root, train=False, transform=mnist_transform, download=True)

    # # Load CIFAR10 dataset
    # cifar10_trainset = torchvision.datasets.CIFAR10(root=root, train=True, transform=cifar10_transform, download=True)
    # cifar10_testset = torchvision.datasets.CIFAR10(root=root, train=False, transform=cifar10_transform, download=True)

    # Create data loaders for easier data handling
    # emnist_trainloader = torch.utils.data.DataLoader(emnist_trainset, batch_size=32, shuffle=True)
    # emnist_testloader = torch.utils.data.DataLoader(emnist_testset, batch_size=32, shuffle=False)

    # mnist_trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=32, shuffle=True)
    # mnist_testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=32, shuffle=False)

    # cifar10_trainloader = torch.utils.data.DataLoader(cifar10_trainset, batch_size=32, shuffle=True)
    # cifar10_testloader = torch.utils.data.DataLoader(cifar10_testset, batch_size=32, shuffle=False)

if __name__ == "__main__":
    init_random_seed()
    split_data_emnist()