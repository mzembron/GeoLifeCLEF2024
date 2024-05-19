import glob
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

root_dir = './data'
metadata_path = './data/PresenceOnlyOccurrences/GLC24-PO-metadata-train.csv'
environmental_dir = 'EnvironmentalRasters'
image_dir = 'SatellitePatches'
landsat_dir = 'SatelliteTimeSeries'
biomonthly_dir = 'EnvironmentalRasters/Climate/Climatic_Monthly_2000-2019_cubes'
id_column = 'surveyId'

po = pd.read_csv(metadata_path)
po_classes = np.sort(np.array(po['speciesId'].unique().tolist()))
# df = (po.groupby(['surveyId', 'lat', 'lon',])
#       .agg({'geoUncertaintyInM': 'max', 'speciesId': lambda x: x.tolist(),})
#       .reset_index())

dataset_name = '-'.join(s for i, s in enumerate(Path(metadata_path).stem.split('-')) if i in (1,3))
image_dirs = glob.glob(f'{root_dir}/{image_dir}/*{dataset_name[:4].upper()}{dataset_name[4:]}*/[!_]*')
landsat_dir = Path(glob.glob(f'{root_dir}/{landsat_dir}/cubes/*{dataset_name}*[!.zip]')[0])
biomonthly_dir = Path(glob.glob(f'{root_dir}/{biomonthly_dir}/*{dataset_name}*[!.zip]/*{dataset_name}*')[0])


missing_data = []
for survey_id in tqdm(po.surveyId.unique()):
    cd = str(survey_id)[-2:]
    ab = str(survey_id)[-4:-2]

    for d in image_dirs:
        if d[-3:] == 'rgb':
            image_rgb = Path(f'{d}/{cd}/{ab}/{survey_id}.jpeg')
        else:
            image_nir = Path(f'{d}/{cd}/{ab}/{survey_id}.jpeg')

    landsat = Path(f'{landsat_dir}/{landsat_dir.stem}_{survey_id}_cube.pt')
    biomonthly = Path(f'{biomonthly_dir}/{biomonthly_dir.stem}_{survey_id}_cube.pt')

    if not (image_rgb.exists() and image_nir.exists() and landsat.exists() and biomonthly.exists()):
        missing_data.append(survey_id)

df = po.loc[~po.surveyId.isin(missing_data)]
df.to_csv('./data/PresenceOnlyOccurrences/GLC24-PO-metadata-train-fixed.csv', index=False)