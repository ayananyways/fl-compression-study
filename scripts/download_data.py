import os
import sys
import urllib.request
import zipfile

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_IMAGENET_DIR = "./data/tiny-imagenet-200"
TINY_IMAGENET_ZIP = "./data/tiny-imagenet-200.zip"

ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
ETTH1_DIR = "./data/etth1"
ETTH1_PATH = "./data/etth1/ETTh1.csv"


def _report_progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        sys.stdout.write(f"\r  {pct:.1f}% ({downloaded // 1024 // 1024} MB)")
        sys.stdout.flush()


def download_tiny_imagenet() -> None:
    if os.path.isdir(TINY_IMAGENET_DIR):
        print("Tiny ImageNet already downloaded, skipping.")
        return

    os.makedirs("./data", exist_ok=True)
    print(f"Downloading Tiny ImageNet from {TINY_IMAGENET_URL} ...")
    urllib.request.urlretrieve(TINY_IMAGENET_URL, TINY_IMAGENET_ZIP, reporthook=_report_progress)
    print()

    print("Extracting Tiny ImageNet ...")
    with zipfile.ZipFile(TINY_IMAGENET_ZIP, "r") as zf:
        zf.extractall("./data/")

    os.remove(TINY_IMAGENET_ZIP)
    print(f"Done. Dataset at {TINY_IMAGENET_DIR}")


def download_etth1() -> None:
    if os.path.isfile(ETTH1_PATH):
        print("ETTh1 already downloaded, skipping.")
        return

    os.makedirs(ETTH1_DIR, exist_ok=True)
    print(f"Downloading ETTh1 from {ETTH1_URL} ...")
    urllib.request.urlretrieve(ETTH1_URL, ETTH1_PATH, reporthook=_report_progress)
    print()
    print(f"Done. Dataset at {ETTH1_PATH}")


def main() -> None:
    download_tiny_imagenet()
    download_etth1()


if __name__ == "__main__":
    main()
