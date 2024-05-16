import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from torchvision.ops import focal_loss
from torchvision.models import vit_b_32, ViT_B_32_Weights, resnet18

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from collections import OrderedDict

from data_loaders import MultimodalDataset, custom_collate, get_transforms


class CLEFModel(LightningModule):
    def __init__(self,
                 focal_loss_gamma=5,
                 focal_loss_alpha=0.25,
                 prob_threshold=0.3,
                 lr=1e-3,
                 sched_step=10,
                 top_k=None):

        super().__init__()
        self.vit = vit_b_32(
            weights=ViT_B_32_Weights.DEFAULT,
            dropout=0.3)

        self.vit.conv_proj = nn.Conv2d(4, 768, kernel_size=(32,32), stride=(32,32))
        self.vit.heads = nn.Sequential(OrderedDict([
            ('head', nn.Linear(768, 2048, bias=True)),
            ('scale', nn.Sigmoid())
        ]))

        self.res_norm = nn.LayerNorm([4,19,12])
        self.res = resnet18(num_classes=256)
        self.res.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1, bias=False)

        self.res_landsat_norm = nn.LayerNorm([6,4,21])
        self.res_landsat = resnet18(num_classes=256)
        self.res_landsat.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1, bias=False)

        self.classifier = nn.Sequential(
            nn.Linear(self.vit.heads.head.out_features + self.res.fc.out_features + self.res_landsat.fc.out_features + 49, 4096),
            nn.ReLU(),
            nn.Linear(4096, 5016, bias=True)
        )

        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha
        self.lr = lr
        self.sched_step = sched_step

        self.prob_threshold = prob_threshold
        self.top_k = top_k

    def loss(self, outputs, y):
        return focal_loss.sigmoid_focal_loss(outputs,
                                             y,
                                             gamma=self.focal_loss_gamma,
                                             alpha=self.focal_loss_alpha,
                                             reduction='sum')

    def forward(self, features):
        images, features, biomonthly, landsat = features

        image_out = self.vit(images)

        bio_out = self.res_norm(biomonthly)
        bio_out = self.res(bio_out)

        landsat_out = self.res_landsat_norm(landsat)
        landsat_out = self.res_landsat(landsat_out)

        x = torch.cat([image_out, bio_out, landsat_out, features], dim=1)

        return self.classifier(x)

    def _prepare_input(self, batch):
        batch, species = batch

        features = torch.stack(batch['features']).nan_to_num()
        meta = torch.stack(batch['metadata']).nan_to_num()
        features = torch.cat((features, meta), dim=1)

        images = torch.cat((torch.stack(batch['image_rgb']), torch.stack(batch['image_nir'])), dim=1)

        biomonthly = torch.stack(batch['biomonthly']).nan_to_num()
        landsat = torch.stack(batch['landsat']).nan_to_num()
        # images = torch.stack(batch['image_rgb'])

        return (images, features, biomonthly, landsat), species

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
        preds = nn.Sigmoid()(preds)
        preds = torch.where(preds < self.prob_threshold, 0, 1)
        TP = torch.logical_and(preds == targets, targets == 1).sum(dim=1)
        FP = torch.logical_and(preds != targets, preds == 1).sum(dim=1)
        FN = torch.logical_and(preds != targets, preds == 0).sum(dim=1)

        self.log(f"valid_TP", TP.sum().item(), on_step=True, logger=True)
        self.log(f"valid_FP", FP.sum().item(), on_step=True, logger=True)
        self.log(f"valid_FN", FN.sum().item(), on_step=True, logger=True)

        F1 = TP/(TP+(FP+FN)/2)
        return F1.mean()

    def predict_step(self, batch, batch_idx):
        x, y = self._prepare_input(batch)
        preds = self.forward(x)
        preds = nn.Sigmoid()(preds)
        if self.top_k:
            _, top_indices = torch.topk(preds, k=self.top_k, dim=1)
            mask = torch.zeros_like(preds)
            mask.scatter_(1, top_indices, 1)
            preds = preds * mask
        preds = torch.where(preds < self.prob_threshold, 0, 1)
        return preds

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.sched_step, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}


def set_seed(seed):
    # Set seed for Python's built-in random number generator
    torch.manual_seed(seed)
    # Set seed for numpy
    np.random.seed(seed)
    # Set seed for CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Set cuDNN's random number generator seed for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def define_chkpt():
    return ModelCheckpoint(
        monitor='valid_F1',
        mode='max',
        filename='best_model-{epoch:02d}-{valid_F1:.2f}',
        save_top_k=1,
        save_on_train_epoch_end=True,
        save_last=True
        )


if __name__ == "__main__":
    set_seed(47)
    transforms = get_transforms()
    pa_dataset = MultimodalDataset(transforms=transforms)
    po_dataset = MultimodalDataset(metadata_path='data/PresenceOnlyOccurrences/GLC24-PO-metadata-train.csv',
                                   transforms=transforms)
    trainset, validset = random_split(pa_dataset, [0.8, 0.2])

    pa_trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2, collate_fn=custom_collate)
    po_trainloader = DataLoader(po_dataset, batch_size=64, shuffle=True, num_workers=2, collate_fn=custom_collate)
    validloader = DataLoader(validset, batch_size=64, shuffle=False, num_workers=2, collate_fn=custom_collate)

    model_chkpt = 'models/v_143_F1-0-32/best_model-epoch=15-valid_F1=0.28.ckpt'
    # model = CLEFModel()
    model = CLEFModel.load_from_checkpoint(model_chkpt)

    trainer = Trainer(max_epochs=50, accelerator='gpu', callbacks=[define_chkpt()])
    trainer.fit(model=model, train_dataloaders=po_trainloader, val_dataloaders=validloader)
