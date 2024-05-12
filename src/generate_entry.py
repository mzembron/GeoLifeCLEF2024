from vit_model import CLEFdummy
from data_loaders import MultimodalDataset, custom_collate
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from lightning import LightningModule, Trainer
import torch
import pandas as pd
import numpy as np

if __name__ == "__main__":
    chckpt = 'lightning_logs/version_136/checkpoints/best_model-epoch=07-valid_F1=0.20.ckpt'
    model = CLEFdummy.load_from_checkpoint(chckpt)
    model.eval()

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

    trainset = MultimodalDataset()
    dataset = MultimodalDataset(
        metadata_path='data/PresenceAbsenceSurveys/GLC24-PA-metadata-test.csv',
        transforms=[transform_rgb, transform_nir],
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
    df['predictions'].to_csv('predictions_vit_rgb+nir+env_best_top25.csv')
    pass
