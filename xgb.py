from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

import xgboost as xgb

from rdkit import Chem
from rdkit.Chem import Descriptors

# creating dataframe
df = pd.read_csv("aqsoldb.csv")

# creating list of molecules in dataset using RDKit and SMILES
mol_list = []

for i in df.SMILES:
    molecule = Chem.MolFromSmiles(i)
    mol_list.append(molecule)


# creating descriptors of all molecules
complete_mol_desc = []

for molecule in mol_list:
    mol_desc = {}

    for name, function in Descriptors._descList: # Descriptors._descList provides list of all descriptors in RDKit Library
        # try-catch in case the descriptor fails to produce a value
        try:
            desc_value = function(molecule)
        
        except:
            # print exception
            import traceback
            traceback.print_exc()

            desc_value = None

        mol_desc[name] = desc_value
    
    complete_mol_desc.append(mol_desc)


# creating dataframe of descriptors
df_desc = pd.DataFrame(complete_mol_desc)
df_desc = df_desc.assign(Solubility = df.Solubility)

inf_locations = np.where(df_desc.values >= np.finfo(np.float32).max) # locating infinite values in dataframe
for i in inf_locations[0]: # replacing infinite values with None
    for j in inf_locations[1]:
        df_desc.iat[i, j] = None

df_desc = df_desc.dropna()


# creating training and testing data
x = df_desc.drop(['Solubility'], axis=1)
y = df_desc['Solubility']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# creating model
model = xgb.XGBRegressor(
    eta=0.04, max_depth=6, n_estimators=200, gamma=0.5, min_child_weight=1, subsample=1, colsample_bytree=0.6
)

model = model.fit(x_train, y_train)

# evaluation metrics
predictions = model.predict(x_test)

r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(r2)
print(rmse)


'''
# defining hyperparameters and distributions for randomised search
param_distributions = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [ 3, 4, 5, 6, 8, 10, 12, 15],
    'min_child_weight': [ 1, 3, 5, 7 ],
    'eta': uniform(0, 1),
    'gamma': uniform(0, 0.5),
    'colsample_bytree': uniform(0.3, 1)
}

# creating model
model = xgb.XGBRegressor()

# performing randomised search
random_search = RandomizedSearchCV(model, param_distributions, cv=3, n_iter=200, scoring='r2', random_state=42)

random_search.fit(x_train, y_train)
print(random_search.best_params_)
'''
