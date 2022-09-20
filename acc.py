import pandas as pd
from sklearn.metrics import accuracy_score

xls = pd.ExcelFile('Task5.xlsx')
df1 = pd.read_excel(xls, 'Day1')
df2 = pd.read_excel(xls, 'Day2')
df3 = pd.read_excel(xls, 'Day3')
df4 = pd.read_excel(xls, 'Day4')
df5 = pd.read_excel(xls, 'Day5')
df6 = pd.read_excel(xls, 'Day6')
df7 = pd.read_excel(xls, 'Day7')
df8 = pd.read_excel(xls, 'Day8')

label = df1['Label']
predict = df1['Predict']

print(predict)

acc = accuracy_score(label, predict)
print(acc)