import os

def test_dataset_setup():
    data_dir = "data"
    files = os.listdir(data_dir)
    print("Files in data folder:", files)

if __name__ == "__main__":
    test_dataset_setup()

