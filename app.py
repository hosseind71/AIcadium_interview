# here the trained model will be deployed using flask library

from flask import Flask
from flask import jsonify
import requests
from flask import request
import joblib
import pandas as pd

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
#@app.route('/predict')
def predict():
     data=request.json
     data = pd.DataFrame([data])
     x_input=process(data)
     classifier = joblib.load('classifier.pkl')
    

     prediction = classifier.predict(x_input)
     if prediction==1:
        return jsonify({'prediction': 'Customer WILL buy the product'})
     else:
            return jsonify({'prediction': 'Customer WILL NOT buy the product'})



def process(input_data):
    import sklearn
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import PowerTransformer
    from sklearn.preprocessing import MinMaxScaler
    
    cat_col=['Month','OperatingSystems','Browser','Region','TrafficType','VisitorType']
    num_col=input_data.columns[0:10]
    input_data[['Weekend']]=input_data[['Weekend']].astype('int64')
    input_data[cat_col]=input_data[cat_col].astype("category")
    
    preproceesing = joblib.load('preprocessing.pkl')
    x=pd.DataFrame(preproceesing.transform(input_data))
    
    x.columns=['Administrative', 'Administrative_Duration', 'Informational','Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
       'BounceRates', 'ExitRates', 'PageValues', 'Month_Aug',
       'Month_Dec', 'Month_Feb', 'Month_Jul', 'Month_June', 'Month_Mar',
       'Month_May', 'Month_Nov', 'Month_Oct', 'Month_Sep',
       'OperatingSystems_1', 'OperatingSystems_2', 'OperatingSystems_3',
       'OperatingSystems_4', 'OperatingSystems_5', 'OperatingSystems_6',
       'OperatingSystems_7', 'OperatingSystems_8', 'Browser_1', 'Browser_2',
       'Browser_3', 'Browser_4', 'Browser_5', 'Browser_6', 'Browser_7',
       'Browser_8', 'Browser_9', 'Browser_10', 'Browser_11', 'Browser_12',
       'Browser_13', 'Region_1', 'Region_2', 'Region_3', 'Region_4',
       'Region_5', 'Region_6', 'Region_7', 'Region_8', 'Region_9',
       'TrafficType_1', 'TrafficType_2', 'TrafficType_3', 'TrafficType_4',
       'TrafficType_5', 'TrafficType_6', 'TrafficType_7', 'TrafficType_8',
       'TrafficType_9', 'TrafficType_10', 'TrafficType_11', 'TrafficType_12',
       'TrafficType_13', 'TrafficType_14', 'TrafficType_15', 'TrafficType_16',
       'TrafficType_17', 'TrafficType_18', 'TrafficType_19', 'TrafficType_20',
       'VisitorType_New_Visitor', 'VisitorType_Other',
       'VisitorType_Returning_Visitor', 'SpecialDay', 'Weekend']
    
    return x

if __name__ == '__main__':
     app.run(port=8030)
        

    
