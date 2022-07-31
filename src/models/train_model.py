# import modules from outside
from loading_data import create_dataloaders
from architecture import NoPoolASPP
from metrics import dice_score

# system libraries 
from pathlib import Path
from tqdm import tqdm, trange
import time

# deep learning libraries
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary 
import optuna
from torch.cuda.amp import GradScaler, autocast

gradient_accumulations = 8 # training will be done for effective batch size of 4 * 8 = 32

class Trainer: 
    def __init__(self, 
                model: torch.nn.Module, 
                criterion: torch.nn.Module, 
                optimizer: torch.optim.Optimizer, 
                train_loader: torch.utils.data.Dataset, 
                val_loader: torch.utils.data.Dataset, 
                lr_scheduler = torch.optim.lr_scheduler, 
                epochs: int = 100,
                epoch: int = 0
                ): 
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.epoch = epoch
        self.scaler = GradScaler()

        self.train_loss = []
        self.val_loss = []
        self.learning_rate = []
        self.dice_score_train = []
        self.dice_score_val = []

    def run_trainer(self):
        progressbar = trange(self.epochs, desc="Progress")
        for i in progressbar: 
            self.epoch += 1
            self.train_loop()
            self.val_loop()
            self.lr_scheduler.step()
            return self.train_loss, self.val_loss, self.learning_rate, self.dice_score_train, self.dice_score_val
    
    def train_loop(self):
        self.model.train()
        train_losses = [] # accumulate losses from this run
        dice_scores = []
        batch_iter = tqdm(enumerate(self.train_loader), 'Training', total=len(self.train_loader), leave=False)

        for i, (x, y) in batch_iter: 
            input, target = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad() # zerograd parameters
            with autocast():
                pred = self.model(input) # forward pass
                loss = self.criterion(pred, target.float()) # calculate loss
                loss_value = loss.item() # get loss value
                dice_value = dice_score(pred, target.float()) # calculate dice score
                dice_scores.append(dice_value)
                train_losses.append(loss_value)
            self.scaler.scale(loss/gradient_accumulations).backward() # backpropagate
            if (i+1) % gradient_accumulations == 0:
                self.scaler.step(optimizer)
                self.scaler.update()

            batch_iter.set_description(f"Training: (loss {loss_value:.4f}, dice {dice_value:.4f})") # update progressbar
        
        self.train_loss.append(sum(train_losses) / len(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])
        self.dice_score_train.append(sum(dice_scores) / len(dice_scores))
        batch_iter.close()

    def val_loop(self):
        self.model.eval() # set model to evaluation mode
        valid_losses = []
        dice_scores = []
        batch_iter = tqdm(enumerate(self.val_loader), 'Validation', total=len(self.val_loader), leave=False)
        
        for i, (x, y) in batch_iter: 
            input, target = x.to(self.device), y.to(self.device)

            with torch.no_grad():
                pred = self.model(input)
                loss = self.criterion(pred, target.float())
                loss_value = loss.item()
                dice_value = dice_score(pred, target.float()) # calculate dice score
                dice_scores.append(dice_value)
                valid_losses.append(loss_value)

                batch_iter.set_description(f"Validation: (loss {loss_value:.4f}, dice {dice_value:.4f})")
        
        self.val_loss.append(sum(valid_losses) / len(valid_losses))
        self.dice_score_val.append(sum(dice_scores) / len(dice_scores))
        batch_iter.close()

if __name__ == '__main__':

    img_dir = Path(r"/content/GLENDA_img")
    mask_dir = Path(r"/content/GLENDA_mask")
    # img_dir = Path(r"D:\GLENDA_v1.5_no_pathology\no_pathology\GLENDA_img")
    # mask_dir = Path(r"D:\GLENDA_v1.5_no_pathology\no_pathology\GLENDA_mask")  

    # hyperparameters
    num_epochs = 1
    initial_lr = 0.001
    batch_size = 4
    num_workers = 2
    drop_rate = 0.4
    bn_momentum = 0.1
    base_num_filters = 64

    train_loader, val_loader, test_loader = create_dataloaders(img_dir, mask_dir, batch_size, num_workers)
    model = NoPoolASPP(drop_rate=drop_rate, bn_momentum=bn_momentum, base_num_filters=base_num_filters)
    criterion = F.binary_cross_entropy_with_logits
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs) # try using with warm restarts
    trainer = Trainer(
        model=model, 
        criterion=criterion, 
        optimizer=optimizer, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        lr_scheduler=scheduler, 
        epochs=num_epochs)

    train_loss, val_loss, learning_rate, dice_score_train, dice_score_val = trainer.run_trainer()


