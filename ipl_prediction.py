import pandas as pd
from sklearn.preprocessing import LabelEncoder

matches = pd.read_csv("input/matches.csv")

#Getting the info of the dataset
print(matches.info())
print("-------------------------------------------------------")

#Check for columns with missing values
for i in matches.columns:
    print("Checking null values for ", i, " column")
    print(matches[pd.isnull(matches[i])])
    print("-------------------------------------------------------")

#Replacing missing values in winners column with draw
matches['winner'].fillna('Draw', inplace=True)

#Replacing missing values in cities column with Dubai
matches['city'].fillna('Dubai',inplace=True)

#Encoding of the team names
matches.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Rising Pune Supergiant','Kochi Tuskers Kerala','Pune Warriors']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','RPS','KTK','PW'],inplace=True)
encode = {'team1': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'team2': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'toss_winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13,'Draw':14}}
matches.replace(encode, inplace=True)

#Descending order of teams winning the tosses and Descending order of wins by the teams
dicVal = encode['winner']
matches = matches[['team1','team2','city','toss_decision','toss_winner','venue','winner']]
temp1 = matches['toss_winner'].value_counts(sort=True)
temp2 = matches['winner'].value_counts(sort=True)
print(matches.head())
print("-------------------------------------------------------")
print('No of toss winners by each team')
for idx, val in temp1.iteritems():
   print('{} -> {}'.format(list(dicVal.keys())[list(dicVal.values()).index(idx)],val))
print("-------------------------------------------------------")
print('No of match winners by each team')
for idx, val in temp2.iteritems():
   print('{} -> {}'.format(list(dicVal.keys())[list(dicVal.values()).index(idx)],val))

#encoding of the categorical variables
var_mod = ['city','toss_decision','venue']
le = LabelEncoder()
for i in var_mod:
    matches[i] = le.fit_transform(matches[i])
print(matches.dtypes)