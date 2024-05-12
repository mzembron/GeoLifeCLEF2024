import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from torchvision.ops import focal_loss
from torchvision.models import vit_b_32, ViT_B_32_Weights
from data_loaders import MultimodalDataset, custom_collate
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from collections import OrderedDict


class CLEFdummy(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = vit_b_32(
            weights=ViT_B_32_Weights.DEFAULT,
            dropout=0.3)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        self.model.conv_proj = nn.Conv2d(4, 768, kernel_size=(32,32), stride=(32,32))
        # self.model.heads.head = nn.Linear(768, 5016, bias=True)

        self.model.heads = nn.Sequential(OrderedDict([
            ('head', nn.Linear(768, 2048, bias=True)),
            # nn.ReLU(),
            # nn.Linear(2048, 5016, bias=True)
            ('scale', nn.Sigmoid()) # make sure the range is the same for image and numerical features
        ]))
        self.classifier = nn.Sequential(
            nn.Linear(self.model.heads.head.out_features + 46, 4096),
            nn.ReLU(),
            nn.Linear(4096, 5016, bias=True)
        )
        # self.loss = nn.BCEWithLogitsLoss()
        self.focal_loss_gamma = 5 # TODO check gamma = 5
        self.focal_loss_alpha = 0.25

        self.top_k = None

    def loss(self, outputs, y):
        return focal_loss.sigmoid_focal_loss(outputs,
                                                y,
                                                gamma=self.focal_loss_gamma,
                                                alpha=self.focal_loss_alpha,
                                                reduction='sum')

    def forward(self, features):
        images, features = features

        image_out = self.model(images)
        x = torch.cat([image_out, features], dim=1)

        return self.classifier(x)

    def _prepare_input(self, batch):
        batch, species = batch
        features = torch.stack(batch['features']).nan_to_num()
        images = torch.cat((torch.stack(batch['image_rgb']), torch.stack(batch['image_nir'])), dim=1)
        # images = torch.stack(batch['image_rgb'])

        return (images, features), species

    def training_step(self, batch, batch_idx):
        x, y = self._prepare_input(batch)

        outputs = self.forward(x)

        loss = self.loss(outputs, y)

        # self.print_predictions(batch, self.global_step)
        self.log(f"train_loss", loss.item(), prog_bar=True, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._prepare_input(batch)
        outputs = self.forward(x)
        loss = self.loss(outputs, y)
        self.log(f"valid_loss", loss.item(), prog_bar=True, on_step=True, logger=True)

        F1 = self.F1_score(outputs, y)
        self.log(f"valid_F1", F1.item(), prog_bar=True, on_step=True, logger=True)

    def F1_score(self, preds, targets):
        threshold = 0.3
        preds = nn.Sigmoid()(preds)
        preds = torch.where(preds < threshold, 0, 1)
        TP = torch.logical_and(preds == targets, targets == 1).sum(dim=1)
        FP = torch.logical_and(preds != targets, preds == 1).sum(dim=1)
        FN = torch.logical_and(preds != targets, preds == 0).sum(dim=1)

        self.log(f"valid_TP", TP.sum().item(), on_step=True, logger=True)
        self.log(f"valid_FP", FP.sum().item(), on_step=True, logger=True)
        self.log(f"valid_FN", FN.sum().item(), on_step=True, logger=True)

        F1 = TP/(TP+(FP+FN)/2)
        return F1.mean()

    def test_step(self, batch, batch_idx):
        x, y = self._prepare_input(batch)
        outputs = self.forward(x)
        loss = self.loss(outputs, y)
        self.log(f"test_loss", loss.item(), on_step=True, logger=True)
        # TODO: add different metrics

    def predict_step(self, batch, batch_idx):
        x, y = self._prepare_input(batch)
        preds = self.forward(x)
        threshold = 0.3
        preds = nn.Sigmoid()(preds)
        if self.top_k:
            _, top_indices = torch.topk(preds, k=self.top_k, dim=1)
            mask = torch.zeros_like(preds)
            mask.scatter_(1, top_indices, 1)
            preds = preds * mask
        preds = torch.where(preds < threshold, 0, 1)
        return preds

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}


if __name__ == "__main__":
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_F1',
        mode='max',
        filename='best_model-{epoch:02d}-{valid_F1:.2f}',
        save_top_k=1,
        save_on_train_epoch_end=True,
        save_last=True
        )
    transform_rgb = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_nir = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485], std=[0.229])
    ])

    dataset = MultimodalDataset(transforms=[transform_rgb, transform_nir])

    trainset, validset = random_split(dataset, [0.8, 0.2])
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2, collate_fn=custom_collate)
    validloader = DataLoader(validset, batch_size=64, shuffle=False, num_workers=2, collate_fn=custom_collate)
    model = CLEFdummy()
    trainer = Trainer(max_epochs=100, accelerator='gpu', callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=validloader)

    # chckpt = 'lightning_logs/version_95/checkpoints/epoch=1-step=2782.ckpt'
    # model = CLEFdummy.load_from_checkpoint(chckpt)



