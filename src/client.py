import pandas as pd
import requests

url = 'http://localhost:3000/getTableDataFromFiles'
files = {'1page0.jpg': ('1page0.jpg', open('demo/1page0.jpg', 'rb'), 'application/pdf', {'Expires': '0'})}
#{'1.pdf': ('1.pdf', open('demo/1.pdf', 'rb'), 'application/pdf', {'Expires': '0'})}
#       {'1page3.jpg': ('1page3.jpg', open('demo/1page3.jpg', 'rb'), 'application/pdf', {'Expires': '0'})}
        #{'1.pdf': ('1.pdf', open('demo/1.pdf', 'rb'), 'application/pdf', {'Expires': '0'})}
        #{'1page10.jpg': ('1page10.jpg', open('demo/1page10.jpg', 'rb'), 'application/pdf', {'Expires': '0'})}
         #'2.pdf': ('2.pdf', open('demo/2.pdf', 'rb'), 'application/pdf', {'Expires': '0'}),
         #'3.pdf': ('3.pdf', open('demo/3.pdf', 'rb'), 'application/pdf', {'Expires': '0'}),
         #'4.pdf': ('4.pdf', open('demo/4.pdf', 'rb'), 'application/pdf', {'Expires': '0'})}

response = requests.post(
    url,
    files=files
)
if response.status_code == 200:
    print("Response 200")
    data = response.json()
    for d in data:
        dataframe = pd.DataFrame.from_dict(data.get(d))
        pwd = "/Users/kamil/PycharmProjects/pythonProject/HW6/" + d + ".csv"
        dataframe.to_csv(pwd)
        print("Save " + d + ".csv")
else:
    print("Server error")