# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 09:43:00 2021

@author: Deepnil Vasava
"""
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__) 
                                                                                   
    

loaded_model = pickle.load(open("finalized_model.sav", "rb"))

@app.route('/')

def symptom():
    
   return render_template('text_pred.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form['Data']
      l=[]
      l.append(result)
      result_pred = loaded_model.best_estimator_.predict(np.array(l))
      return render_template("text_pred.html",result = result_pred)

if __name__ == '__main__':
   app.run(host="localhost", port=8000, debug=True)
