# import modules from outside
from loading_data import create_dataloaders
from architecture import NoPoolASPP
from losses import dice_loss

# system libraries 
from pathlib import Path
from tqdm import tqdm
import time

# deep learning libraries
import torch
import torch.optim as optim

num_epochs = 1
initial_lr = 0.001
batch_size = 32

if __name__ == '__main__':

    img_dir = Path(r"/content/GLENDA_img")
    mask_dir = Path(r"/content/GLENDA_mask")

    train_loader, val_loader, test_loader = create_dataloaders(img_dir, mask_dir, batch_size)
    model = NoPoolASPP()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(device)

    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    for epoch in tqdm(range(1, num_epochs+1)):
        start_time = time.time()
        scheduler.step()

        lr = scheduler.get_lr()[0]

        model.train()
        train_loss_total = 0.0
        num_steps = 0
        
        # run train loop for one epoch
        for i, batch in enumerate(train_loader):
            img, mask = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            pred = model(img)
            loss = dice_loss(pred, mask)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
            num_steps += 1

            if i % 5 == 0:
                print("Epoch: {}, Step: {}, Loss: {}".format(epoch, i, -loss.item()))

        train_loss_total_avg = train_loss_total / num_steps
        # run validation loop for one epoch
        model.eval()
        val_loss_total = 0.0
        num_steps = 0

        for i, batch in enumerate(val_loader):
            img, mask = batch[0].to(device), batch[1].to(device)
            pred = model(img)
            loss = dice_loss(pred, mask)
            val_loss_total += loss.item()
            num_steps += 1

        val_loss_total_avg = val_loss_total / num_steps
        print("Epoch: {}, Train Loss: {}, Val Loss: {}".format(epoch, train_loss_total_avg, val_loss_total_avg))
