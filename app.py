import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify

app = Flask(__name__)

dataset = pd.read_csv("train.csv")
X = dataset.filter(['Pclass','Sex','Age','SibSp','Parch','Embarked'],axis = 1).values
Y = dataset.filter(['Survived'],axis = 1).values



le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])



ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [-1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean' )
imputer.fit(X[:,0:8])
X[:,0:8] = imputer.transform(X[:,0:8])
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit(X_train)


model = tf.keras.models.load_model('model')


@app.route('/', methods=['POST'])
def get_chances():
    if request.method == 'POST':
        new_data = request.json
        name = new_data.get('name')
        pclass = int(new_data.get('pclass'))
        sex = int(new_data.get('sex'))
        age = int(new_data.get('age'))
        sibsp = int(new_data.get('sibsp'))
        parch = int(new_data.get('parch'))
        embark = new_data.get('embark')
        inp = [[pclass,sex,age,sibsp,parch,embark]]
        z = np.array(ct.transform(inp))
        result = round(model.predict(sc.transform(z))[0][0]*100,2)
        return jsonify({'name':name,'chances':result})






if __name__ == '__main__':
    app.run()


# {"name" : "Hisham" , "pclass" : "3" , "sex" : "0" , "age" : "21" , "sibsp" : "0" , "parch" : "0" , "embark" : "C"    }