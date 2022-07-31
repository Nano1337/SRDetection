# import outside functions
from loading_data import create_dataloaders

# system libraries
from pathlib import Path

# deep learning libraries
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# global constants
# img_dir = Path(r"D:\GLENDA_v1.5_no_pathology\no_pathology\GLENDA_img")
# mask_dir = Path(r"D:\GLENDA_v1.5_no_pathology\no_pathology\GLENDA_mask")
img_dir = Path(r"/content/GLENDA_img")
mask_dir = Path(r"/content/GLENDA_mask")
batch_size = 4
num_workers = 2
initial_lr = 0.001

class SRDetectModel(pl.LightningModule):
    def __init__(self, drop_rate=0.4, bn_momentum=0.1,
                 base_num_filters=64):
        super().__init__()

        self.conv1a = nn.Conv2d(3, base_num_filters, kernel_size=3, padding=1)
        self.conv1a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.conv1a_drop = nn.Dropout2d(drop_rate)
        self.conv1b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=1)
        self.conv1b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.conv1b_drop = nn.Dropout2d(drop_rate)

        self.conv2a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=2, dilation=2)
        self.conv2a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.conv2a_drop = nn.Dropout2d(drop_rate)
        self.conv2b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=2, dilation=2)
        self.conv2b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.conv2b_drop = nn.Dropout2d(drop_rate)

        # Branch 1x1 convolution
        self.branch1a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=1)
        self.branch1a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch1a_drop = nn.Dropout2d(drop_rate)
        self.branch1b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=1)
        self.branch1b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch1b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 6
        self.branch2a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=6, dilation=6)
        self.branch2a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch2a_drop = nn.Dropout2d(drop_rate)
        self.branch2b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=6, dilation=6)
        self.branch2b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch2b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 12
        self.branch3a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=12, dilation=12)
        self.branch3a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch3a_drop = nn.Dropout2d(drop_rate)
        self.branch3b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=12, dilation=12)
        self.branch3b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch3b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 18
        self.branch4a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=18, dilation=18)
        self.branch4a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch4a_drop = nn.Dropout2d(drop_rate)
        self.branch4b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=18, dilation=18)
        self.branch4b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch4b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 24
        self.branch5a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=24, dilation=24)
        self.branch5a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch5a_drop = nn.Dropout2d(drop_rate)
        self.branch5b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=24, dilation=24)
        self.branch5b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch5b_drop = nn.Dropout2d(drop_rate)

        self.concat_drop = nn.Dropout2d(drop_rate)
        self.concat_bn = nn.BatchNorm2d(6*base_num_filters, momentum=bn_momentum)

        self.amort = nn.Conv2d(6*base_num_filters, base_num_filters*2, kernel_size=1)
        self.amort_bn = nn.BatchNorm2d(base_num_filters*2, momentum=bn_momentum)
        self.amort_drop = nn.Dropout2d(drop_rate)

        self.prediction = nn.Conv2d(base_num_filters*2, 1, kernel_size=1)

    def forward(self, x):
        """Model forward pass.

        :param x: input data.
        """
        x = F.relu(self.conv1a(x))
        x = self.conv1a_bn(x)
        x = self.conv1a_drop(x)

        x = F.relu(self.conv1b(x))
        x = self.conv1b_bn(x)
        x = self.conv1b_drop(x)

        x = F.relu(self.conv2a(x))
        x = self.conv2a_bn(x)
        x = self.conv2a_drop(x)
        x = F.relu(self.conv2b(x))
        x = self.conv2b_bn(x)
        x = self.conv2b_drop(x)

        # Branch 1x1 convolution
        branch1 = F.relu(self.branch1a(x))
        branch1 = self.branch1a_bn(branch1)
        branch1 = self.branch1a_drop(branch1)
        branch1 = F.relu(self.branch1b(branch1))
        branch1 = self.branch1b_bn(branch1)
        branch1 = self.branch1b_drop(branch1)

        # Branch for 3x3 rate 6
        branch2 = F.relu(self.branch2a(x))
        branch2 = self.branch2a_bn(branch2)
        branch2 = self.branch2a_drop(branch2)
        branch2 = F.relu(self.branch2b(branch2))
        branch2 = self.branch2b_bn(branch2)
        branch2 = self.branch2b_drop(branch2)

        # Branch for 3x3 rate 6
        branch3 = F.relu(self.branch3a(x))
        branch3 = self.branch3a_bn(branch3)
        branch3 = self.branch3a_drop(branch3)
        branch3 = F.relu(self.branch3b(branch3))
        branch3 = self.branch3b_bn(branch3)
        branch3 = self.branch3b_drop(branch3)

        # Branch for 3x3 rate 18
        branch4 = F.relu(self.branch4a(x))
        branch4 = self.branch4a_bn(branch4)
        branch4 = self.branch4a_drop(branch4)
        branch4 = F.relu(self.branch4b(branch4))
        branch4 = self.branch4b_bn(branch4)
        branch4 = self.branch4b_drop(branch4)

        # Branch for 3x3 rate 24
        branch5 = F.relu(self.branch5a(x))
        branch5 = self.branch5a_bn(branch5)
        branch5 = self.branch5a_drop(branch5)
        branch5 = F.relu(self.branch5b(branch5))
        branch5 = self.branch5b_bn(branch5)
        branch5 = self.branch5b_drop(branch5)

        # Global Average Pooling
        global_pool = F.avg_pool2d(x, kernel_size=x.size()[2:])
        global_pool = global_pool.expand(x.size())

        concatenation = torch.cat([branch1,
                                   branch2,
                                   branch3,
                                   branch4,
                                   branch5,
                                   global_pool], dim=1)

        concatenation = self.concat_bn(concatenation)
        concatenation = self.concat_drop(concatenation)

        amort = F.relu(self.amort(concatenation))
        amort = self.amort_bn(amort)
        amort = self.amort_drop(amort)

        predictions = self.prediction(amort)
        predictions = F.sigmoid(predictions)

        return predictions

    def configure_optimizers(self):
        """Return optimizers and schedulers."""
        optimizer = optim.Adam(self.parameters(), lr=initial_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, verbose=True)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        """Train step.

        :param batch: input data.
        :param batch_idx: batch index.
        """
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred, y.float())
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step.

        :param batch: input data.
        :param batch_idx: batch index.
        """
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred, y.float())
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Test step.

        :param batch: input data.
        :param batch_idx: batch index.
        """
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred, y.float())
        self.log("test_loss", loss, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        """Predict step.

        :param batch: input data.
        :param batch_idx: batch index.
        """
        x, y = batch
        y_pred = self(x)
        return y_pred

if __name__ == "__main__":

    train_loader, val_loader, test_loader = create_dataloaders(img_dir, mask_dir, batch_size, num_workers, )

    model = SRDetectModel()
    trainer = pl.Trainer(accelerator="gpu", devices=1)
    trainer.fit(model, train_loader, val_loader)