from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_children = pd.read_csv("data/Statewise Cases Reported of Crimes Committed Against Children 1994-2016.csv", header=None)
children_states=[i for i in data_children[0][1:].unique()]
children_crimes=[i for i in data_children[1][1:].unique()]
children_years=[2021,2022,2023,2024,2025]
data_women = pd.read_csv("data/statewise_crime_against_women_2001_15.csv")
women_states=[i for i in data_women['STATE/UT'].unique()]
women_crimes=[i for i in data_women][2:]
women_years=[2021,2022,2023,2024,2025]

def children_prediction(state,year,crime):
	global data_children
	data=data_children
	year=int(year)
	X=data.iloc[0:1,2:].values[0]
	X_train=np.array([int(i) for i in X])
	data =data[data [0]==state]
	data =data[data [1]==crime]
	print(data.iloc[:,2:])
	y=data.iloc[:,2:].values[0]
	y_train=np.array([int(i) for i in y])
	linear_regression=LinearRegression()
	linear_regression.fit(X_train.reshape(-1,1),y_train)
	print(len(X_train),len(y_train))
	score=linear_regression.score(X_train.reshape(-1,1),y_train)
	b=np.array([])
	if score < 0.60:
		b=np.array([str(i) for i in range(1994,2017)])
		y = list(y)
		years = list(b)
		year = 2016
		output = "Can't predict further"
	else:
		for j in range(2017,year+1):
			prediction = linear_regression.predict(np.array([[j]]))
			if(prediction < 0):
				prediction = 0
			y = np.append(y,prediction)
		b=np.array([str(i) for i in range(1994,year+1)])
		y = list(y)
		years = list(b)
		output = ""
	if output:
		print(output)
		print(score)
	else:
		print(y)
	print(b)
	return (y,years,output)

def women_prediction(state,year,crime):	
	global data_women
	data=data_women
	print(year)
	year=int(year)
	data =data[data['STATE/UT']==state]
	X=[i for i in data['Year']]
	X_train=np.array([int(i) for i in X])
	y=[i for i in data[crime]]
	y_train=np.array([int(i) for i in y])
	linear_regression=LinearRegression()
	linear_regression.fit(X_train.reshape(-1,1),y_train)
	print(len(X_train),len(y_train))
	score=linear_regression.score(X_train.reshape(-1,1),y_train)
	b=np.array([])
	if score < 0.60:
		b=np.array([str(i) for i in range(2001,2016)])
		y = list(y)
		years = list(b)
		year = 2015
		output = "Can't predict"
	else:
		for j in range(2016,year+1):
			prediction = linear_regression.predict(np.array([[j]]))
			if(prediction < 0):
				prediction = 0
			y = np.append(y,prediction)
		b=np.array([str(i) for i in range(2001,year+1)])
		y = list(y)
		years = list(b)
		output = ""
	if output:
		print(output)
	else:
		print(y)
	print(b)

	return (y,years,output)

def data_pred(file):
	import numpy as np 
	import matplotlib.pyplot as plt 
	from sklearn.ensemble import RandomForestClassifier
	import pandas as pd
	import numpy as np
	from sklearn.metrics import confusion_matrix,accuracy_score
	from sklearn import tree

	np.random.seed(0)

	data_set = pd.read_csv('data\\'+file)

	data_set['is_train'] = np.random.uniform(0, 1, len(data_set)) <= .75

	data_set['Region'] = pd.factorize(data_set['Region'])[0]
	data_set['States/UTs'] = pd.factorize(data_set['States/UTs'])[0]
	data_set['Type'] = pd.factorize(data_set['Type'])[0]

	X_train, X_test = data_set[data_set['is_train'] == True], data_set[data_set['is_train'] == False]

	features = data_set.columns[1:4]

	y_train = X_train['Region']
	y_test = X_test['Region']
	X_train = X_train[features]
	X_test = X_test[features]

	clf = RandomForestClassifier(n_jobs=5, random_state=1000)
	clf.fit(X_train, y_train)
	clf_tree = clf.fit(X_train, y_train)
	
	preds = clf.predict(X_test)
	y_pred=clf.predict(X_test)
	score=accuracy_score(y_test, preds)
	
	A=['Arabian Sea','Bay of Bengal','Central','Eastern','Northeastern','Northern','Southern','Western']
	import seaborn as sns
	cm=confusion_matrix(y_test, preds)
	sns.heatmap(cm,fmt='d',annot=True,xticklabels=A, yticklabels=A)
	plt.title('Ploting Graph')
	plt.xlabel('Actual')
	plt.ylabel('Predicted')
	plt.savefig('static/images/predicted_heatmap_plot.png')

	df_pred=pd.DataFrame(y_test.values,columns=['Actual'])
	df_pred['Predicted']=y_pred

	X=['Arabian Sea', 'Bay of Bengal', 'Central', 'Eastern','Northeastern', 'Northern', 'Southern', 'Western']
	plt.figure(figsize=(15,10))
	plt.bar(df_pred.Actual - 0.2, df_pred.index.values, 0.4, label = 'Actual')
	plt.bar(df_pred.Predicted + 0.2, df_pred.index.values, 0.4, label = 'Predicted')

	a=df_pred.Actual.unique()
	plt.xticks(a, X)

	plt.xlabel("Region")
	plt.ylabel("")
	plt.title("Actual vs Predicted")
	plt.legend()
	plt.savefig('static/images/compare_act_pred_bar.png')

	return score


def pred_crime_plot(state,crime,x,y):
    plt.figure(figsize=(10,10)) 
    plt.grid(True)
    plt.xticks(fontsize=8)
    plt.plot(y,x)
    plt.xlabel('Years')
    plt.ylabel('No. of '+crime+' Cases in '+state)
    plt.title(crime)
    plt.savefig('static/images/plot.png')