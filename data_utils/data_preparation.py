import zipfile

def unzip_dataset(inputpath: str, outputpath: str):
    with zipfile.ZipFile(inputpath, "r") as zipped_file:
        zipped_file.extractall(outputpath)

if __name__ == "__main__":
    print("Extraction started")
    unzip_dataset("data/cmnist.zip", "data")
    print("Extraction ended")