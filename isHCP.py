
import pandas as pd
from sklearn.model_selection import train_test_split ,GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import LabelEncoder

import hashlib



data = pd.read_csv('./Doceree-HCP_Train.csv',encoding='latin1')
testdata = pd.read_csv('./Doceree-HCP_Test.csv',encoding='latin1')



data.drop(['ID', 'CHANNELTYPE','USERAGENT'], axis=1, inplace=True)

testdata.drop(['ID', 'CHANNELTYPE','USERAGENT'], axis=1, inplace=True)



label_encoder = LabelEncoder()

# Fit and transform the 'DEVICETYPE' column
data['DEVICETYPE'] = label_encoder.fit_transform(data['DEVICETYPE'])
data['USERCITY'] = label_encoder.fit_transform(data['USERCITY'])
data['PLATFORMTYPE'] = label_encoder.fit_transform(data['PLATFORMTYPE'])
data['TAXONOMY'] = label_encoder.fit_transform(data['TAXONOMY'])
data['URL'] = label_encoder.fit_transform(data['URL'])
data['KEYWORDS'] = label_encoder.fit_transform(data['KEYWORDS'])
data['BIDREQUESTIP'] = label_encoder.fit_transform(data['BIDREQUESTIP'])
data['USERPLATFORMUID'] = label_encoder.fit_transform(data['USERPLATFORMUID'])

data['USERZIPCODE'].fillna(-1, inplace=True)
data['IS_HCP'].fillna(0, inplace=True)




# Fit and transform the 'DEVICETYPE' column
testdata['DEVICETYPE'] = label_encoder.fit_transform(testdata['DEVICETYPE'])
testdata['USERCITY'] = label_encoder.fit_transform(testdata['USERCITY'])
testdata['PLATFORMTYPE'] = label_encoder.fit_transform(testdata['PLATFORMTYPE'])
# testdata['TAXONOMY'] = label_encoder.fit_transform(testdata['TAXONOMY'])
testdata['URL'] = label_encoder.fit_transform(testdata['URL'])
testdata['KEYWORDS'] = label_encoder.fit_transform(testdata['KEYWORDS'])
testdata['BIDREQUESTIP'] = label_encoder.fit_transform(testdata['BIDREQUESTIP'])
testdata['USERPLATFORMUID'] = label_encoder.fit_transform(testdata['USERPLATFORMUID'])

testdata['USERZIPCODE'].fillna(-1, inplace=True)
# testdata['IS_HCP'].fillna(0, inplace=True)



features = ['DEVICETYPE', 'PLATFORM_ID', 'BIDREQUESTIP','USERPLATFORMUID', 'USERCITY', 'USERZIPCODE', 'PLATFORMTYPE', 'URL', 'TAXONOMY','KEYWORDS']

X = data[features]
y = data['IS_HCP']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



param_grid = {
    'n_estimators': [300],
    'max_depth': [None],
    'min_samples_split': [2]
}

# Create the Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Model Accuracy: {accuracy:.4f}")
