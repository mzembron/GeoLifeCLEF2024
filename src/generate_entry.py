from clef_model import CLEFModel
from data_loaders import MultimodalDataset, custom_collate, get_transforms
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from lightning import LightningModule, Trainer
import torch
import pandas as pd
import numpy as np

if __name__ == "__main__":
    chckpt = 'lightning_logs/version_701531/model/best_model-epoch=15-valid_F1=0.25.ckpt'
    model = CLEFModel.load_from_checkpoint(chckpt)
    model.eval()

    trainset = MultimodalDataset()
    dataset = MultimodalDataset(
        metadata_path='data/PresenceAbsenceSurveys/GLC24-PA-metadata-test.csv',
        transforms=get_transforms(),
        meta_scaler_path='metadata_scaler.pkl',
        tab_scaler_path='feature_scaler.pkl')
    testloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2, collate_fn=custom_collate)

    model.top_k = 25
    trainer = Trainer(accelerator='gpu')
    predictions = trainer.predict(model=model, dataloaders=testloader)

    df = pd.DataFrame(np.vstack([tensor.numpy() for tensor in predictions]))
    df.columns = trainset.classes.astype(int)
    df.index = dataset.metadata.index

    def aggregate_columns(row):
        return ' '.join(row.index[row == 1].astype(str))

    df['predictions'] = df.apply(aggregate_columns, axis=1)
    df['predictions'].to_csv('predictions_vit+2x_rgb+nir+env+landsat_best_lr1e-3_all.csv')
    pass
