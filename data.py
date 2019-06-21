import os
import subprocess
from glob import glob


def docs_generator(files_dir):
    files_path = glob(os.path.join(files_dir, "*.gz"))

    for fp in files_path:
        file_name = fp.split("/")[-1].split(".")[0] # filename without extension
        ofp = os.path.join(files_dir, file_name)
        subprocess.run(["gunzip",'-k', fp])
        with open(ofp) as fobj:
            yield fobj

        subprocess.run(["rm", ofp])


def text_line_generator(docs):
    for doc in docs:
        for line in doc:
            yield " ".join(line.split()[:5])