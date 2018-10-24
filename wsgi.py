#Load Packages
import os, csv, string
import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import tree 
import pickle
from flask import Flask, jsonify, request, render_template , json
import csv
import io
import shutil

 

application = Flask(__name__,template_folder='templates')


@app.route('/')
def form():
    return """
        <html>
            <body>
                <h1>Carbon monoxide (CO) emissions Predictions</h1>

                <form action="/predict" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" />
                    <input type="submit" />
					
                </form>
            </body>
        </html>
    """




@app.route('/predict', methods=['POST'])
def apicall():
	"""API Call
	
	Pandas dataframe from API Call
	"""
	try:
		file = request.files['data_file']
		if not file:
			return "No file"
		print(file)
			
		new_data = pd.read_csv(file,encoding='latin-1')		
			
		#data cleaning
		new_Remove_variables = ["Car_id","Description"] 
		new_char_variables = ["Manufacturer","Model","Transmission","Fuel_Type","Fuel_Cost","Electricity_cost","Total_Cost","Engine_Capacity","Metric_ExtraUrban","Noise_Level"]
		new_X  = new_data.drop(new_Remove_variables,axis = 1)
		new_X_char = new_X[new_char_variables]
		new_X_num = new_X.drop(new_char_variables,axis=1)
		new_X_num = new_X_num.fillna(0) 
		for col in list(new_X_char):
		   new_X_char.loc[:,col] = new_X_char.loc[:,col].astype(str)
		#Label Encoding                                  
		dict = {}
		le = preprocessing.LabelEncoder()
		for column in new_X_char:                                        
			le.fit(new_X_char.loc[:,column])
			dict[column] = le.classes_               
			new_X_char.loc[:,column]= le.transform(new_X_char.loc[:,column])
		
		new_pred_data = pd.concat([new_X_char,new_X_num],axis = 1)     
	
	except Exception as e:
		raise e
	
	
	
	if new_pred_data.empty:
		return(bad_request())
	else:
		#Load the saved model
		print("Loading the model...")   
		
		loaded_model = None
		with open('model_v1.pkl','rb') as f:
			loaded_model = pickle.load(f,encoding='latin-1')
		print("The model has been loaded...doing predictions now...")
		#Predict CO Emission
		predictions = loaded_model.predict(new_pred_data)  
		
		df = pd.DataFrame(predictions)
		df.columns = ['pred_emission_co']
		def fun2 (x):
			if x.pred_emission_co == 1:
				y="Low"
			elif x.pred_emission_co == 2:
				y="Medium"
			else:
				y="High"
			return y
		new_data['CO_Emission_flag'] = df.apply(fun2,axis=1)
		
		new_pred_data = new_data[['Car_id','CO_Emission_flag']]
		new_pred_data = pd.DataFrame(new_pred_data)
		#new_pred = new_pred_data.to_dict() 
			
			
		"""We can be as creative in sending the responses.
		   But we need to send the response codes as well.
		"""
		#os.remove("/templates/test.html")
		current_path="test.html"
		target_folder="/templates/"
		
		new_pred_data.to_html('test.html')
		#shutil.move(current_path, target_folder)
		
		return render_template('test.html')
		# responses = jsonify(predictions=new_pred_data.to_json(orient="records"))
		# #responses = jsonify(predictions=new_pred)
		# print("completed")
		# return (responses)
	
		
	

@app.errorhandler(400)
def bad_request(error=None):
	message = {
			'status': 400,
			'message': 'Bad Request: ' + request.url + '--> Please check your data ...',
	}
	resp = jsonify(message)
	resp.status_code = 400

	return resp

if __name__ == '__main__':
    application.run(host= '0.0.0.0',port=5000,debug=True)
