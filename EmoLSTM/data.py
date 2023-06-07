from datetime import datetime
import os
import numpy as np
import pickle


def load_data(path):
    # Unzip the training data
    data2021 = {}
    for fname in os.listdir(os.path.join(path, "data")):
        if not fname.endswith(".xml"):
            continue
        file_path = os.path.join(os.path.join(path, "data"), fname)
        with open(file_path, "r") as file:
            _id = fname[:-4]
            data2021[_id] = {"time": [], "text": []}
            for line in [x.strip() for x in file.read().split("\n")]:
                if line.startswith("<TEXT>"):
                    data2021[_id]["text"].append(line[6:-7])
                elif line.startswith("<DATE>"):
                    data2021[_id]["time"].append(
                        datetime.strptime(line[6:-7], "%Y-%m-%d %H:%M:%S")
                    )
    arr = np.loadtxt(
        os.path.join(path, "risk_golden_truth.txt"), delimiter=" ", dtype=object
    )
    for row in arr:
        _id, isSad = row
        if not _id in data2021:
            print(_id)
        else:
            data2021[_id]["isSad"] = int(isSad)
    for key in data2021:
        if not "isSad" in data2021[key]:
            print(key)
    for key in data2021:
        assert len(data2021[key]["text"]) == len(data2021[key]["time"])
    data2022 = {}
    for fname in os.listdir(os.path.join(path, "data")):
        if not fname.endswith(".xml"):
            continue
        file_path = os.path.join(os.path.join(path, "data"), fname)
        with open(file_path, "r") as file:
            _id = fname[:-4]
            data2022[_id] = {"time": [], "text": []}
            for line in [x.strip() for x in file.read().split("\n")]:
                if line.startswith("<TEXT>"):
                    data2022[_id]["text"].append(line[6:-7])
                elif line.startswith("<DATE>"):
                    data2022[_id]["time"].append(
                        datetime.strptime(line[6:-7], "%Y-%m-%d %H:%M:%S")
                    )
    arr = np.loadtxt(
        os.path.join(path, "risk_golden_truth.txt"), delimiter=" ", dtype=object
    )
    for row in arr:
        _id, isSad = row
        if not _id in data2022:
            print(_id)
        else:
            data2022[_id]["isSad"] = int(isSad)
    for key in data2022:
        if not "isSad" in data2022[key]:
            print(key)
    for key in data2022:
        assert len(data2022[key]["text"]) == len(data2022[key]["time"])
    with open(os.path.join(path, "data2021.pkl"), "wb") as file:
        pickle.dump(data2021, file)
    with open(os.path.join(path, "data2022.pkl"), "wb") as file:
        pickle.dump(data2022, file)
    return data2021, data2022