import zipfile
import argparse


def unzip_dataset(inputpath: str, outputpath: str):
    with zipfile.ZipFile(inputpath, "r") as zipped_file:
        zipped_file.extractall(outputpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="data/cifar10c.zip", help='path to csv')
    args = parser.parse_args()

    print("Extraction started")
    unzip_dataset(args.dataset, "data")
    print("Extraction ended")