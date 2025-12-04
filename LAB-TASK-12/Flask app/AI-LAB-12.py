from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

data = pd.read_csv("titanic_big.csv")

encoder = LabelEncoder()
data['sex'] = encoder.fit_transform(data['sex'])
data['embarked'] = encoder.fit_transform(data['embarked'])

X = data[['pclass', 'sex', 'age', 'fare', 'embarked']]
y = data['survived']


model = LogisticRegression()
model.fit(X, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""

    if request.method == 'POST':
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        embarked = int(request.form['embarked'])

        result = model.predict([[pclass, sex, age, fare, embarked]])

        if result[0] == 1:
            prediction = "Passenger Survived "
        else:
            prediction = "Passenger Did Not Survive "

    return render_template('index.html', output=prediction)

if __name__ == "__main__":
    app.run(debug=True)
