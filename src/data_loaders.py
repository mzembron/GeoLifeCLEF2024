from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import torch
from torchvision.io import read_image
import glob
from torch.utils.data import Dataset, DataLoader

warnings.simplefilter(action='ignore', category=FutureWarning)

class MultimodalDataset(Dataset):
    def __init__(self, 
                 root_dir='./data',
                 metadata_path='./data/PresenceAbsenceSurveys/GLC24-PA-metadata-train.csv',
                 environmental_dir='EnvironmentalRasters',
                 image_dir='SatellitePatches',
                 timeseries_dir='SatelliteTimeSeries',
                 id_column='surveyId',
                 transform=None):

        dataset_name = '-'.join(s for i, s in enumerate(Path(metadata_path).stem.split('-')) if i in (1,3))
        csv_files = glob.glob(f'{root_dir}/{environmental_dir}/*/*{dataset_name}*.csv')

        self.metadata = self._read_metadata(metadata_path)
        self.tabdata = self._merge_data(csv_files, id_column)
        self.image_dirs = glob.glob(f'{root_dir}/{image_dir}/*{dataset_name[:4].upper()}{dataset_name[4:]}*/[!_]*')

        self.timeseries_dir = Path(glob.glob(f'{root_dir}/{timeseries_dir}/cubes/*{dataset_name}*[!.zip]')[0])
        self.transform = transform

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
            'metadata': torch.tensor(survey.drop('speciesId'), dtype=torch.float), # lat, lon, geoUncertaintyInM
            'image_nir': read_image(image_rgb),
            'image_rgb': read_image(image_nir),
            'features': torch.tensor(sample, dtype=torch.float),  # Soilgrid [0-8], HumanFootprint[9-24], Bio [25-43], Landcover[44], Elevation[45]
            'timeseries': torch.load(timeseries),
            'species': torch.tensor(np.isin(self.classes, np.array(survey['speciesId'])).astype(int), dtype=torch.float)
            # 'species': torch.tensor(survey['speciesId'], dtype=torch.int32), #TODO: if no species don't raise error
        }
        return sample_dict

    def _read_metadata(self, path):
        df = pd.read_csv(path)
        self.classes = np.sort(np.array(df['speciesId'].unique().tolist()))
        df = (df.groupby(['surveyId', 'lat', 'lon',])
                .agg({'geoUncertaintyInM': 'max', 'speciesId': lambda x: x.tolist(),})
                .reset_index())
        return df.set_index('surveyId')

    def _merge_data(self, csv_files, id_column):
        data_frames = [pd.read_csv(file).set_index(id_column) for file in csv_files]

        merged_data = data_frames[0]
        for df in data_frames[1:]:
            merged_data = pd.merge(merged_data, df, left_index=True, right_index=True)

        return merged_data


def custom_collate(batch):
    return {key: [sample[key] for sample in batch] for key in batch[0]}


if __name__ == "__main__":
    dataset = MultimodalDataset()
    # for i, x in enumerate(dataset):
    #     if i == 2:
    #         print(x)
    #         break
    dataloader = DataLoader(dataset, batch_size=4,
                        shuffle=True, num_workers=0, collate_fn=custom_collate)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched)
        # print(type(sample_batched))
        break