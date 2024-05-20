import pickle
import warnings
import zipfile
import glob
import torch
import numpy as np
import pandas as pd

import torchvision.transforms.functional as TF
from torchvision.io import read_image, decode_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from pathlib import Path

warnings.simplefilter(action='ignore', category=FutureWarning)


class MultimodalDataset(Dataset):
    def __init__(self,
                 root_dir='./data',
                 metadata_path='./data/PresenceAbsenceSurveys/GLC24-PA-metadata-train.csv',
                 environmental_dir='EnvironmentalRasters',
                 image_dir='SatellitePatches',
                 landsat_dir='SatelliteTimeSeries',
                 biomonthly_dir='EnvironmentalRasters/Climate/Climatic_Monthly_2000-2019_cubes',
                 id_column='surveyId',
                 transforms=None,
                 meta_scaler_path=None,
                 tab_scaler_path=None,
                 classes_path=None,
                 img_only=False,
                 from_zip=False):
        assert img_only if from_zip else True, 'from_zip=True requires img_only=True'

        dataset_name = '-'.join(s for i, s in enumerate(Path(metadata_path).stem.split('-')) if i in (1, 3))
        csv_files = glob.glob(f'{root_dir}/{environmental_dir}/*/*{dataset_name}*.csv')

        self.classes = self._load_classes(classes_path)
        self.meta_scaler_path = meta_scaler_path
        self.metadata = self._read_metadata(metadata_path)

        if not img_only:
            self.tab_scaler_path = tab_scaler_path
            self.tabdata = self._merge_data(csv_files, id_column)
            if from_zip:
                self.landsat_dir = Path(glob.glob(f'{root_dir}/{landsat_dir}/cubes/*{dataset_name}*.zip')[0])
                self.biomonthly_dir = Path(glob.glob(f'{root_dir}/{biomonthly_dir}/*{dataset_name}*.zip')[0])
            else:
                self.landsat_dir = Path(glob.glob(f'{root_dir}/{landsat_dir}/cubes/*{dataset_name}*[!.zip]')[0])
                self.biomonthly_dir = Path(glob.glob(f'{root_dir}/{biomonthly_dir}/*{dataset_name}*[!.zip]/*{dataset_name}*')[0])
        if from_zip:
            self.image_zipfile = {
                'rgb': zipfile.ZipFile(glob.glob(f'{root_dir}/{image_dir}/*{dataset_name[:4].upper()}{dataset_name[4:]}*RGB*.zip')[0]),
                'nir': zipfile.ZipFile(glob.glob(f'{root_dir}/{image_dir}/*{dataset_name[:4].upper()}{dataset_name[4:]}*NIR*.zip')[0])}
        else:
            self.image_dirs = {
                'rgb': glob.glob(f'{root_dir}/{image_dir}/*{dataset_name[:4].upper()}{dataset_name[4:]}*RGB*/[!_]*')[0],
                'nir': glob.glob(f'{root_dir}/{image_dir}/*{dataset_name[:4].upper()}{dataset_name[4:]}*NIR*/[!_]*')[0]}
        self.transforms = transforms  # TODO: convert transform into separate parameters
        self.img_only = img_only
        self.from_zip = from_zip

    def __len__(self):
        return len(self.metadata)

    def _get_all_modalities(self, survey):
        survey_id = survey.surveyId
        sample = self.tabdata.loc[survey_id]
        image_rgb, image_nir = self._get_img_paths(survey_id)

        landsat = f'{self.landsat_dir}/{self.landsat_dir.stem}_{survey_id}_cube.pt'
        biomonthly = f'{self.biomonthly_dir}/{self.biomonthly_dir.stem}_{survey_id}_cube.pt'

        sample_dict = {
            'survey_id': survey_id,
            'metadata': torch.tensor(survey.drop(['speciesId', 'surveyId'], errors='ignore'), dtype=torch.float),  # lat, lon, geoUncertaintyInM
            'image_nir': self.transforms[1](read_image(image_nir, ImageReadMode.GRAY)),
            'image_rgb': self.transforms[0](read_image(image_rgb)),
            'features': torch.tensor(sample, dtype=torch.float),  # Soilgrid [0-8], HumanFootprint[9-24], Bio [25-43], Landcover[44], Elevation[45]
            'landsat': torch.load(landsat),
            'biomonthly': torch.load(biomonthly)
        }
        return sample_dict

    def _get_img_only(self, survey_id):
        image_rgb, image_nir = self._get_img_paths(survey_id)

        if self.from_zip:
            f = self.image_zipfile['rgb']
            image_rgb = f.read(image_rgb)
            # try:
            #     image_rgb = f.read(image_rgb)
            # except Exception as e:
            #     print(image_rgb)
            #     raise Exception(e)
            image_rgb = decode_image(torch.frombuffer(image_rgb, dtype=torch.uint8))

            f = self.image_zipfile['nir']
            image_nir = f.read(image_nir)
            image_nir = decode_image(torch.frombuffer(image_nir, dtype=torch.uint8), ImageReadMode.GRAY)
        else:
            image_nir = read_image(image_nir, ImageReadMode.GRAY)
            image_rgb = read_image(image_rgb)

        image = torch.cat((image_rgb, image_nir))
        image = TF.to_pil_image(image)

        images = self.transforms(image)
        return images

    def _get_img_paths(self, survey_id):
        cd = str(survey_id)[-2:]
        ab = str(survey_id)[-4:-2]

        if self.from_zip:
            if not hasattr(self, 'rgb_dir'):
                f = self.image_zipfile['rgb']
                self.rgb_dir = f.infolist()[0].filename
            image_rgb = f'{self.rgb_dir}{cd}/{ab}/{survey_id}.jpeg'
            if not hasattr(self, 'nir_dir'):
                f = self.image_zipfile['nir']
                self.nir_dir = f.infolist()[0].filename
            image_nir = f'{self.nir_dir}{cd}/{ab}/{survey_id}.jpeg'
        else:
            image_rgb = f'{self.image_dirs["rgb"]}/{cd}/{ab}/{survey_id}.jpeg'
            image_nir = f'{self.image_dirs["nir"]}/{cd}/{ab}/{survey_id}.jpeg'
        return image_rgb, image_nir

    def __getitem__(self, idx):
        survey = self.metadata.iloc[idx]
        survey_id = survey.surveyId

        sample = self._get_img_only(survey_id) if self.img_only else self._get_all_modalities(survey)

        if 'speciesId' in survey.index:
            classes = torch.tensor(np.isin(self.classes, np.array(survey['speciesId'])).astype(int), dtype=torch.float)
        else:
            classes = None
        return sample, classes

    def _read_metadata(self, path):
        df = pd.read_csv(path)
        if 'speciesId' in df.columns:
            if self.classes is None:
                self.classes = np.sort(np.array(df['speciesId'].unique().tolist()))
            df = (df.groupby(['surveyId', 'lat', 'lon',])
                    .agg({'geoUncertaintyInM': 'max', 'speciesId': list, })
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
        return df.reset_index()

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

    def _load_classes(self, path):
        if path:
            with open(path, 'rb') as f:
                classes = pickle.load(f)
            return classes


def custom_collate(batch):
    dicts, tensors = zip(*batch)
    col_dicts = {key: [sample[key] for sample in dicts] for key in dicts[0]}
    batch_tensor = torch.stack(tensors) if isinstance(tensors[0], torch.Tensor) else None
    return col_dicts, batch_tensor


def get_transforms():
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
    return [transform_rgb, transform_nir]


if __name__ == "__main__":
    # dataset = MultimodalDataset(metadata_path='data/PresenceOnlyOccurrences/GLC24-PO-metadata-train-fixed.csv',
    #                                transforms=get_transforms(),
    #                                classes_path='classes.pkl')
    # dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, collate_fn=custom_collate)

    # for i_batch, sample_batched in tqdm(enumerate(dataloader)):
    #     break
    from lightly.transforms.byol_transform import BYOLTransform
    from custom_byol_transforms import BYOLView1Transform, BYOLView2Transform
    transform = BYOLTransform(
        view_1_transform=BYOLView1Transform(input_size=224, channels=4, random_gray_scale=0, solarization_prob=0),
        view_2_transform=BYOLView2Transform(input_size=224, channels=4, random_gray_scale=0, solarization_prob=0),
    )

    dataset = MultimodalDataset(transforms=transform, from_zip=False, img_only=True)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True,
        num_workers=1,
    )

    for i_batch, sample_batched in enumerate(dataloader):
        sample_batched, classes = sample_batched
        print(i_batch, len(sample_batched))
        print(sample_batched[0].shape)
        print(sample_batched[1].shape)
        # print(type(sample_batched))
        break
