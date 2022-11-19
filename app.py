from flask import Flask,render_template,request
import pickle

app=Flask(__name__)

#load the model
with open("boston.pkl","rb") as f:
    load_model=pickle.load(f)

@app.route('/')
def home():
    result=" "
    return render_template('index.html',**locals())

@app.route('/predict',methods=['POST'])
def predict():
    CRIM=float(request.form['CRIM'])
    ZN=float(request.form['ZN'])
    INDUS=float(request.form['INDUS'])
    CHAS=float(request.form['CHAS'])
    NOX=float(request.form['NOX'])
    RM=float(request.form['RM'])
    AGE=float(request.form['AGE'])
    DIS=float(request.form['DIS'])
    RAD=float(request.form['RAD'])
    TAX=float(request.form['TAX'])
    PTRATIO=float(request.form['PTRATIO'])
    B=float(request.form['B'])
    LSTAT=float(request.form['LSTAT'])
    result=load_model.predict([[CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT]])
    return render_template('index.html',**locals())

if __name__=="__main__":
    app.run(host='0.0.0.0',port=8080,debug=True)