from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import torch
from torchvision.io import read_image, ImageReadMode
import glob
from torch.utils.data import Dataset, DataLoader
import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)


class MultimodalDataset(Dataset):
    def __init__(self, 
                 root_dir='./data',
                 metadata_path='./data/PresenceAbsenceSurveys/GLC24-PA-metadata-train.csv',
                 environmental_dir='EnvironmentalRasters',
                 image_dir='SatellitePatches',
                 timeseries_dir='SatelliteTimeSeries',
                 id_column='surveyId',
                 transforms=None,
                 meta_scaler_path=None,
                 tab_scaler_path=None):

        dataset_name = '-'.join(s for i, s in enumerate(Path(metadata_path).stem.split('-')) if i in (1,3))
        csv_files = glob.glob(f'{root_dir}/{environmental_dir}/*/*{dataset_name}*.csv')

        self.meta_scaler_path = meta_scaler_path
        self.tab_scaler_path = tab_scaler_path
        self.metadata = self._read_metadata(metadata_path)
        self.tabdata = self._merge_data(csv_files, id_column)

        self.image_dirs = glob.glob(f'{root_dir}/{image_dir}/*{dataset_name[:4].upper()}{dataset_name[4:]}*/[!_]*')

        self.timeseries_dir = Path(glob.glob(f'{root_dir}/{timeseries_dir}/cubes/*{dataset_name}*[!.zip]')[0])
        self.transforms = transforms  # TODO: convert transform into separate parameters

    def __len__(self):
        return len(self.tabdata)

    def __getitem__(self, idx):
        survey = self.metadata.iloc[idx]
        survey_id = survey.name

        sample = self.tabdata.loc[survey_id]

        cd = str(survey_id)[-2:]
        ab = str(survey_id)[-4:-2]

        for d in self.image_dirs:
            if d[-3:] == 'rgb':
                image_rgb = f'{d}/{cd}/{ab}/{survey_id}.jpeg'
            else:
                image_nir = f'{d}/{cd}/{ab}/{survey_id}.jpeg'

        timeseries = f'{self.timeseries_dir}/{self.timeseries_dir.stem}_{survey_id}_cube.pt'

        sample_dict = {
            'survey_id': survey_id,
            'metadata': torch.tensor(survey.drop('speciesId', errors='ignore'), dtype=torch.float),  # lat, lon, geoUncertaintyInM
            'image_nir': self.transforms[1](read_image(image_nir, ImageReadMode.GRAY)),
            'image_rgb': self.transforms[0](read_image(image_rgb)),
            'features': torch.tensor(sample, dtype=torch.float),  # Soilgrid [0-8], HumanFootprint[9-24], Bio [25-43], Landcover[44], Elevation[45]
            'timeseries': torch.load(timeseries),
            # 'species': torch.tensor(np.isin(self.classes, np.array(survey['speciesId'])).astype(int), dtype=torch.float)
            # 'species': torch.tensor(survey['speciesId'], dtype=torch.int32), #TODO: if no species don't raise error
        }
        if 'speciesId' in survey.index:
            classes = torch.tensor(np.isin(self.classes, np.array(survey['speciesId'])).astype(int), dtype=torch.float)
        else:
            classes = None
        return sample_dict, classes

    def _read_metadata(self, path):
        df = pd.read_csv(path)
        if 'speciesId' in df.columns:
            self.classes = np.sort(np.array(df['speciesId'].unique().tolist()))
            df = (df.groupby(['surveyId', 'lat', 'lon',])
                    .agg({'geoUncertaintyInM': 'max', 'speciesId': lambda x: x.tolist(),})
                    .reset_index())
        else:
            self.classes = None
            df = (df.groupby(['surveyId', 'lat', 'lon',])
                    .agg({'geoUncertaintyInM': 'max'})
                    .reset_index())
        df = df.set_index('surveyId')
        df[df.drop(columns='speciesId', errors='ignore').columns], metadata_scaler = self._scale(df.drop(columns='speciesId', errors='ignore'), self.meta_scaler_path)
        with open('metadata_scaler.pkl', 'wb') as f:
            pickle.dump(metadata_scaler, f)
        return df

    def _merge_data(self, csv_files, id_column):
        data_frames = [pd.read_csv(file).set_index(id_column) for file in csv_files]

        merged_data = data_frames[0]
        for df in data_frames[1:]:
            merged_data = pd.merge(merged_data, df, left_index=True, right_index=True)

        merged_data, feature_scaler = self._scale(merged_data, self.tab_scaler_path)
        with open('feature_scaler.pkl', 'wb') as f:
            pickle.dump(feature_scaler, f)
        return merged_data

    def _scale(self, df, scaler_path):
        df = df.replace(np.NINF, np.nan)

        if scaler_path:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            df[df.columns] = scaler.transform(df)
        else:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            df[df.columns] = scaler.fit_transform(df)

        df = df.fillna(df.mean())
        return df, scaler


def custom_collate(batch):
    # TODO: move torch.stack here
    dicts, tensors = zip(*batch)
    col_dicts = {key: [sample[key] for sample in dicts] for key in dicts[0]}
    batch_tensor = torch.stack(tensors) if isinstance(tensors[0], torch.Tensor) else None
    return col_dicts, batch_tensor


if __name__ == "__main__":
    from torchvision.transforms import v2
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
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=custom_collate)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched)
        # print(type(sample_batched))
        break