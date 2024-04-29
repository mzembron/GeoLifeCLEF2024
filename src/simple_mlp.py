import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from torchvision.models import vit_b_32, ViT_B_32_Weights
from data_loaders import MultimodalDataset, custom_collate
from lightning import LightningModule, Trainer


class CLEFdummy(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = vit_b_32(
            weights=ViT_B_32_Weights.DEFAULT,
            dropout=0.3)
        for param in self.model.parameters():
            param.requires_grad = False
        # self.model.conv_proj = nn.Conv2d(4, 768, kernel_size=(32,32), stride=(32,32))
        # self.model.heads.head = nn.Linear(768, 5016, bias=True)

        self.model.heads.head = nn.Sequential(
            nn.Linear(768, 2048, bias=True),
            nn.ReLU(),
            nn.Linear(2048, 5016, bias=True)
        )
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, features):
        return self.model(features)

    def _prepare_input(self, batch):
        features = torch.stack(batch['features']).nan_to_num()
        # images = torch.cat((torch.stack(batch['image_rgb']),torch.stack(batch['image_nir'])), dim=1)
        images = torch.stack(batch['image_rgb'])
        species = torch.stack(batch['species'])
        return images, species


    def training_step(self, batch, batch_idx):
        x, y = self._prepare_input(batch)

        outputs = self.forward(x)

        loss = self.loss(outputs, y)

        # self.print_predictions(batch, self.global_step)
        self.log(f"train_loss", loss.item(), prog_bar=True, on_step=True, logger=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._prepare_input(batch)
        outputs = self.forward(x)
        loss = self.loss(outputs, y)
        self.log(f"valid_loss", loss.item(), prog_bar=True, on_step=True, logger=True, batch_size=x.shape[0])
        # TODO: add different metrics

    def test_step(self, batch, batch_idx):
        x, y = self._prepare_input(batch)
        outputs = self.forward(x)
        loss = self.loss(outputs, y)
        self.log(f"test_loss", loss.item(), prog_bar=True, on_step=True, logger=True, batch_size=x.shape[0])
        # TODO: add different metrics

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer


if __name__ == "__main__":
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
    trainer = Trainer(max_epochs=100, accelerator='gpu')
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=validloader)

    # chckpt = 'lightning_logs/version_95/checkpoints/epoch=1-step=2782.ckpt'
    # model = CLEFdummy.load_from_checkpoint(chckpt)

    # model.eval()
    # for i_batch, batch in enumerate(dataloader):
    #     # print(i_batch, batch['survey_id'])
    #     inpt = torch.cat((torch.stack(batch['image_rgb']),torch.stack(batch['image_nir'])), dim=1)
    #     inpt = inpt.to(model.device)
    #     out = model(inpt)
    #     out = nn.Sigmoid()(out) > 0.5
    #     for i in range(inpt.shape[0]):
    #         pred_match = ((nn.Sigmoid()(out) > 0.5)[i].cpu() == batch['species'][i])
    #         if sum([1 for m, s in zip(pred_match, batch['species'][63]) if s==1 and m is True]) > 0:
    #             print(batch['survey_id'][i])
    #     # print(out)
    #     if i_batch > 10:
    #         break

