import pickle
with open("settings.pkl", "rb") as file:
    print(pickle.load(file))