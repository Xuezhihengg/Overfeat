import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.overfeat_accurate_base import OverFeat_accurate_base


#  define model parameters
NUM_EPOCHS = 90
BATCH_SIZE = 128
MOMENTUM = 0.6
LR_DECAY = 1e-5
LR_INIT = 5e-2
CROP_SIZE = 221
NUM_CLASSES = 1000
DEVICE_IDS = [0,1,2,3]

#  path to data directory
INPUT_ROOT_DIR = '/home/zhxue/projects/CNN_Impl/data/imagenet/2012' # absolute path
TRAIN_IMG_DIR = INPUT_ROOT_DIR + '/ILSVRC2012_img_train'
VAL_IMG_DIR = INPUT_ROOT_DIR + '/ILSVRC2012_img_val'
OUTPUT_DIR = 'output'
LOG_DIR = OUTPUT_DIR + "/tblogs"
CHECKPOINT_DIR = OUTPUT_DIR + '/checkpoints'

# make checkpoint path directory
os.makedirs(OUTPUT_DIR,exist_ok=True)
os.makedirs(LOG_DIR,exist_ok=True)
os.makedirs(CHECKPOINT_DIR,exist_ok=True)

device = ("cuda" if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    # seed
    seed = torch.initial_seed()
    print(f"Used seed: {seed}")

    # Tensorboard summary
    tbwriter = SummaryWriter(log_dir=LOG_DIR)
    print("Tensorborad summary writer created")

    # create model
    overfeat_accurate_base = OverFeat_accurate_base(NUM_CLASSES).to(device)
    # train on multiple GPUs
    overfeat_accurate_base = nn.parallel.DataParallel(overfeat_accurate_base,device_ids=DEVICE_IDS)
    print(overfeat_accurate_base)
    print("Model created")

    # create datasets and dataloaders
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_train = datasets.ImageFolder(TRAIN_IMG_DIR,transform=transform)
    dataset_val = datasets.ImageFolder(VAL_IMG_DIR,transform=transform)
    print("Datasets created")

    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )
    dataloader_val = DataLoader(
        dataset=dataset_val,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )
    print("Dataloaders created")

    # create optimizer
    # it can't train, the gradient disappears.
    optimizer = optim.SGD(
        params=overfeat_accurate_base.parameters(),
        lr=LR_INIT, # 5e-2
        momentum=MOMENTUM, # 0.6
        weight_decay=LR_DECAY) # 1e-5
    print("Optimizer created")

    # create lr_scheduler
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[30,50,60,70,80],gamma=0.5)

    # start training...
    overfeat_accurate_base.train()
    step = 1
    print("Start training...")
    for epoch in range(NUM_EPOCHS):
        for imgs, classes in dataloader_train:  # imgs size (BATCH_SIZE x 3 x 256 x 256); classes size (BATCH_SIZE)
            imgs, classes = imgs.to(device), classes.to(device)

            output = overfeat_accurate_base(imgs)  # output size (BATCH_SIZE,NUM_CLASSES)
            loss = F.cross_entropy(output,classes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                with torch.no_grad():
                    _, preds = torch.max(output,dim=1)  # preds size (BATCH_SIZE)
                    accuracy = torch.divide(torch.sum(preds == classes),BATCH_SIZE)

                    print(f"Epoch: {epoch+1}/{NUM_EPOCHS} \tStep: {step} \tLoss: {loss.item():.4f} \tAcc: {accuracy.item():.4f}%")
                    tbwriter.add_scalar('loss',loss.item(),step)
                    tbwriter.add_scalar('accuracy', accuracy.item(), step)

                    for name, parameter in overfeat_accurate_base.named_parameters():
                        if parameter.grad is not None:
                            avg_grad = torch.mean(parameter.grad)
                            print(f'\t{name} - grad_avg: {avg_grad}')
                        if parameter.data is not None:
                            avg_weight = torch.mean(parameter.data)
                            print(f'\t{name} - param_avg: {avg_weight}')

            if step % 1000 == 0:
                with torch.no_grad():
                    for name, parameter in overfeat_accurate_base.named_parameters():
                        if parameter.grad is not None:
                            avg_grad = torch.mean(parameter.grad)
                            print(f'\t{name} - grad_avg: {avg_grad}')
                            tbwriter.add_scalar(f'grad_avg/{name}', avg_grad.item(), step)
                            tbwriter.add_histogram(f'grad/{name}', parameter.grad.cpu().numpy(), step)
                        if parameter.data is not None:
                            avg_weight = torch.mean(parameter.data)
                            print(f'\t{name} - param_avg: {avg_weight}')
                            tbwriter.add_histogram(f'weight/{name}', parameter.data.cpu().numpy(), step)
                            tbwriter.add_scalar(f'weight_avg/{name}', avg_weight.item(), step)

                    overfeat_accurate_base.eval()
                    val_cLoss = 0
                    val_cAcc = 0
                    val_count = 0
                    for val_imgs, val_classes in dataloader_val:
                        val_imgs, val_classes = val_imgs.to(device), val_classes.to(device)

                        val_output = overfeat_accurate_base(imgs)
                        val_cLoss += F.cross_entropy(val_output,val_classes)

                        _, val_pred = torch.max(val_output, 1)
                        val_cAcc += torch.divide(torch.sum(val_pred == val_classes),BATCH_SIZE)

                        val_count += 1

                    val_loss = val_cLoss/val_count
                    val_accuracy = val_cAcc/val_count

                    print(f"Epoch: {epoch+1}/{NUM_EPOCHS} \tValidation Loss: {val_loss:.4f} \tValidation Acc: {val_accuracy:.4f}%")
                    tbwriter.add_scalar('val_loss',val_loss,step)
                    tbwriter.add_scalar('val_accuracy', val_accuracy, step)

            step += 1

        lr_scheduler.step()

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'overfeat_accurate_states_epoch{epoch + 1}.pkl')
        state = {
            'epoch':epoch,
            "step":step,
            "optimizer": optimizer.state_dict(),
            "model": overfeat_accurate_base.state_dict(),
            'seed': seed,
        }
        torch.save(state,checkpoint_path)








