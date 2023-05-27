import requests
import pandas as pd

def download_data(month_start, month_end, limit=100, offset=0):
    url = "https://data.gov.sg/api/action/datastore_search?resource_id=f1765b54-a209-4718-8d38-a39237f502b3&limit="


    # print(data_json)
    all_data = []
    month_range = (
        pd.date_range(month_start, month_end, freq="MS").strftime("%Y-%m").tolist()
    )
    for month in month_range:
        url_extend = url + \
            str(limit) + \
            "&offset=" + \
            str(offset) + \
            "&q=%7B%22month%22%3A%20%22" + \
            month + \
            "%22%7D"
        connection = requests.get(url_extend)
        data_json = connection.json()
        data_json = data_json["result"]["records"]
        all_data.extend(data_json)

    df = pd.DataFrame.from_records(all_data)
    df.to_csv('./data/tmp.csv')

if __name__ == "__main__":
    download_data("2021-01", "2021-03", limit=5)
