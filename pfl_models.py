from models.vgg import get_vgg_model
from models.resnet import ResNet18
from models.net import Net
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR


def get_model_and_optimizer(args):
    if args.model == "LeNet":
        model = Net(num_classes=10).to(args.device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        # model.load_state_dict(torch.load('./checkpoint/mnist_cnn.pt'))

    elif args.model in ("vgg9", "vgg11", "vgg13", "vgg16"):
        model = get_vgg_model(args.model).to(args.device)
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
        )
        scheduler = MultiStepLR(
            optimizer, milestones=[e for e in [151, 251]], gamma=0.1
        )

    elif args.model in ("ResNet18"):
        model = ResNet18().to(args.device)
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
        )
        scheduler = MultiStepLR(
            optimizer, milestones=[e for e in [151, 251]], gamma=0.1
        )

    criterion = nn.CrossEntropyLoss(reduction="sum")
    return model, optimizer, scheduler, criterion

def get_client_optimizer(client_optimizer_name, client_model, client_lr, client_weight_decay):
    if client_optimizer_name == "sgd":
        return optim.SGD(client_model.parameters(), lr=client_lr, weight_decay=client_weight_decay)
    elif client_optimizer_name == "adam":
        return optim.Adam(client_model.parameters(), lr=client_lr, weight_decay=client_weight_decay)
    elif client_optimizer_name == "adagrad":
        return optim.Adagrad( client_model.parameters(), lr=client_lr, weight_decay=client_weight_decay)
    elif client_optimizer_name == "adadelta":
        return optim.Adadelta( client_model.parameters(), lr=client_lr, weight_decay=client_weight_decay)
    elif client_optimizer_name == "rmsprop": 
        return optim.RMSprop( client_model.parameters(), lr=client_lr, weight_decay=client_weight_decay)
    else:
        raise Exception("Optimizer not supported")