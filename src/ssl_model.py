# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
import torchvision.transforms as T

from torchvision.models import vit_b_32, ViT_B_32_Weights, resnet18

from pytorch_lightning.callbacks import ModelCheckpoint

from lightly.loss import BarlowTwinsLoss
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)

from data_loaders_ssl import MultimodalDataset


class BarlowTwins(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # resnet = torchvision.models.resnet18()
        vit = vit_b_32(weights=ViT_B_32_Weights.DEFAULT, dropout=0.3)
        vit.conv_proj = nn.Conv2d(4, 768, kernel_size=(32,32), stride=(32,32))
        # self.backbone = nn.Sequential(*list(vit.children())[:-1])
        vit.heads = nn.Identity()
        self.backbone = vit
        self.projection_head = BarlowTwinsProjectionHead(768, 2048, 2048)

        # enable gather_distributed to gather features from all gpus
        # before calculating the loss
        self.criterion = BarlowTwinsLoss(gather_distributed=True)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1) = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("loss", loss.item(), on_step=True, logger=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim

if __name__ == "__main__":

    model = BarlowTwins()

    #Random grayscale and solarization not implemented for 4 channels
    transform = BYOLTransform(
        view_1_transform=BYOLView1Transform(input_size=224, channels=4, random_gray_scale=0, solarization_prob=0),
        view_2_transform=BYOLView2Transform(input_size=224, channels=4, random_gray_scale=0, solarization_prob=0),
    )

    dataset = MultimodalDataset(
        metadata_path='data/PresenceOnlyOccurrences/GLC24-PO-metadata-train-fixed.csv',
        transforms=transform, 
        img_only=True)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    checkpoint_callback = ModelCheckpoint(
            monitor='valid_F1',
            mode='max',
            filename='best_model-{epoch:02d}-{valid_F1:.2f}',
            save_top_k=1,
            save_on_train_epoch_end=True,
            save_last=True
            )

    trainer = pl.Trainer(
        max_epochs=50,
        devices="auto",
        accelerator="gpu",
        strategy="ddp",
        sync_batchnorm=True,
        use_distributed_sampler=True,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model=model, train_dataloaders=dataloader)