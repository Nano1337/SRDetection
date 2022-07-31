# import modules from outside
from loading_data import create_dataloaders
from architecture import NoPoolASPP
from metrics import dice_score

# system libraries 
from pathlib import Path
from tqdm import tqdm
import time

# deep learning libraries
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary 


num_epochs = 100
initial_lr = 0.001
batch_size = 4

if __name__ == '__main__':

    # img_dir = Path(r"/content/GLENDA_img")
    # mask_dir = Path(r"/content/GLENDA_mask")
    img_dir = Path(r"D:\GLENDA_v1.5_no_pathology\no_pathology\GLENDA_img")
    mask_dir = Path(r"D:\GLENDA_v1.5_no_pathology\no_pathology\GLENDA_mask")

    train_loader, val_loader, test_loader = create_dataloaders(img_dir, mask_dir, batch_size)
    model = NoPoolASPP()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(device)

    summary(model, (3, 360, 640))
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    # overfit to single sample first to debug model
    batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    for epoch in tqdm(range(1, num_epochs+1)):
        start_time = time.time()
        scheduler.step()

        lr = scheduler.get_lr()[0]

        model.train()
        train_loss_total = 0.0
        num_steps = 0

        # run train loop for one epoch
        # for i, batch in enumerate(train_loader):
        img, mask = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        pred = model(img)
        loss = F.binary_cross_entropy_with_logits(pred, mask.float())
        loss.backward()
        optimizer.step()
        train_loss_total += loss.item()
        num_steps += 1

        # if i % 1 == 0:
        print("Training - Epoch: {}, Step: {}, Loss: {}, Dice Score: {}".format(epoch, num_steps, loss.item(), dice_score(pred, mask.float())))

        train_loss_total_avg = train_loss_total / num_steps

        # run validation loop for one epoch
        model.eval()
        with torch.no_grad():
            val_loss_total = 0.0
            num_steps = 0

            # for i, val_batch in enumerate(val_loader):
            img, mask = val_batch[0].to(device), val_batch[1].to(device)
            pred = model(img)
            loss = F.binary_cross_entropy_with_logits(pred, mask.float())
            val_loss_total += loss.item()
            print("Validation - Epoch: {}, Step: {}, Loss: {}, Dice Score: {}".format(epoch, num_steps, loss.item(), dice_score(pred, mask.float())))
            num_steps += 1  

        val_loss_total_avg = val_loss_total / num_steps
        print("Epoch: {}, Train Loss: {}, Val Loss: {}".format(epoch, train_loss_total_avg, val_loss_total_avg))
