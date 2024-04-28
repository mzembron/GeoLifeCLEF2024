import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, random_split
from data_loaders import MultimodalDataset, custom_collate
from lightning import LightningModule, Trainer


class CLEFdummy(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                        nn.Linear(46, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 4096),
                        nn.ReLU(),
                        nn.Linear(4096, 8192),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(8192, 5016)
                        )
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, features):
        return self.model(features)
    
    def _prepare_input(self, batch):
        features = torch.stack(batch['features']).nan_to_num()
        species = torch.stack(batch['species'])
        return features, species


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
        #TODO: add different metrics

    def test_step(self, batch, batch_idx):
        x, y = self._prepare_input(batch)
        outputs = self.forward(x)
        loss = self.loss(outputs, y)
        self.log(f"test_loss", loss.item(), prog_bar=True, on_step=True, logger=True, batch_size=x.shape[0])
        #TODO: add different metrics

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    dataset = MultimodalDataset()
    trainset, validset = random_split(dataset, [0.8, 0.2])
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2, collate_fn=custom_collate)
    validloader = DataLoader(validset, batch_size=64, shuffle=False, num_workers=2, collate_fn=custom_collate)
    model = CLEFdummy()
    trainer = Trainer(max_epochs=100, accelerator='gpu')
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=validloader)
    
