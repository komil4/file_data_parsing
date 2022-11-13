import pandas as pd
import os
import requests

def get_file_struct(filename):
    return {filename: (filename, open((os.getcwd() + '/test/' + filename).replace('/src', ''), 'rb'), 'application/pdf', {'Expires': '0'})}


url = 'http://localhost:3001/getJpegImagesFromFiles?threshold=0&color=1&autorotate=1'
files = get_file_struct('5.pdf')
# 1.pdf
# 1page3.jpg
# 2.pdf
# 3.pdf

response = requests.post(
    url,
    files=files
)
if response.status_code == 200:
    print("Response 200")
    data = response.json()
    for d in data:
        dataframe = pd.DataFrame.from_dict(data.get(d))
        pwd = (os.getcwd() + '/csvs/' + d + ".csv").replace('src/', '')
        dataframe.to_csv(pwd)
        print("Save " + d + ".csv")
else:
    print("Server error")