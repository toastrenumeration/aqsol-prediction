from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

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
model = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=20), learning_rate=0.6, loss='exponential', n_estimators=17, random_state=42
)

model = model.fit(x_train, y_train)

# evaluation metrics
predictions = model.predict(x_test)

r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(r2)
print(rmse)


'''
# using grid search to tune hyperparameters
model = AdaBoostRegressor()

param_grid = [
    {
        'loss': ['linear'], 'estimator': [DecisionTreeRegressor(max_depth=5), DecisionTreeRegressor(max_depth=10), DecisionTreeRegressor(max_depth=20)],
        'learning_rate': [0.2, 0.4, 0.6, 0.8, 1.0], 'n_estimators': [15, 16, 17, 18, 19, 20]
    },
    {
        'loss': ['exponential'], 'estimator': [DecisionTreeRegressor(max_depth=5), DecisionTreeRegressor(max_depth=10), DecisionTreeRegressor(max_depth=20)],
        'learning_rate': [0.2, 0.4, 0.6, 0.8, 1.0], 'n_estimators': [15, 16, 17, 18, 19, 20]
     }
]

grid_search = GridSearchCV(model, param_grid, scoring='neg_root_mean_squared_error', cv=2)

grid_search = grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
'''