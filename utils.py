import pandas as pd

def trim_time(data):
    if "good_objects" not in data.columns:
        raise KeyError("Column 'good_objects' not found in the dataset.")
    if "datetime" not in data.columns:
        raise KeyError("Column 'datetime' not found in the dataset.")

    start_time = data.loc[data[data["good_objects"] > 0].index[0], "datetime"]
    end_time = data.loc[data["good_objects"].idxmax(), "datetime"]
    return data[(data["datetime"] >= start_time) & (data["datetime"] <= end_time)].reset_index(drop=True)

def process_csv(file):
    data = pd.read_csv(file)

    data['datetime'] = pd.to_datetime(data['datetime'])
    #data.set_index('datetime', inplace=True)

    data["code"] = data.code.convert_dtypes("float", "int")
    data["error_code"] = data.error_code.convert_dtypes("float", "int")
    data["good_objects"] = data.good_objects.convert_dtypes("float", "int")
    data["total_time"] = data.total_time.convert_dtypes("float", "int")
    data["inactive_time"] = data.inactive_time.convert_dtypes("float", "int")

    data.loc[:, ["buffer_level", "good_objects", "total_time", "inactive_time"]] = \
        data[["buffer_level", "good_objects", "total_time", "inactive_time"]].ffill()

    #data = data[data.index > pd.Timestamp("2024-04-17 08:00")]
    data = data.query("error_code != 1705")

    return trim_time(data)
