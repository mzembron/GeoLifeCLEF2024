import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
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

    def training_step(self, batch, batch_idx):
        features = torch.stack(batch['features']).nan_to_num()
        species = torch.stack(batch['species'])

        outputs = self.forward(features)

        loss = self.loss(outputs, species)

        # self.print_predictions(batch, self.global_step)
        self.log(f"loss", loss.item(), prog_bar=True, on_step=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    dataset = MultimodalDataset()
    dataloader = DataLoader(dataset, batch_size=64,
                        shuffle=True, num_workers=2, collate_fn=custom_collate)
    model = CLEFdummy()
    trainer = Trainer(max_epochs=100, accelerator='gpu')
    trainer.fit(model=model, train_dataloaders=dataloader)
    
