import pickle
import pandas as pd
import numpy as np

po = pd.read_csv('data/PresenceOnlyOccurrences/GLC24-PO-metadata-train.csv')
pa = pd.read_csv('data/PresenceAbsenceSurveys/GLC24-PA-metadata-train.csv')

po_classes = np.sort(np.array(po['speciesId'].unique().tolist()))
pa_classes = np.sort(np.array(pa['speciesId'].unique().tolist()))

all_classes = set(pa_classes) | set(po_classes)
all_classes = sorted(list(all_classes))

with open('classes.pkl', 'wb') as f:
    pickle.dump(all_classes, f)
