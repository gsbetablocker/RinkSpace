"""Adopt an intact copy of the official pvanet.pt from a local download."""
from argparse import ArgumentParser
from pathlib import Path
from shutil import copyfile
from zipfile import ZipFile

from experiments.skate_video_v1.inputs import atomic_json, digest, emit

URL="https://drive.google.com/file/d/1dffyj3cdH1Myw2-tzgKTXryaO72Gbys1/view"
SIZE=3_569_259_862


def verify(path):
    if path.stat().st_size!=SIZE:
        raise ValueError("File size does not match the author-posted checkpoint")
    with ZipFile(path) as z:
        names=z.namelist()
        if not any(s.endswith("/data.pkl") for s in names):
            raise ValueError("Expected a PyTorch serialization archive")
        if bad:=z.testzip():
            raise ValueError(f"Archive member CRC mismatch: {bad}")


def adopt(source, folder):
    folder.mkdir(exist_ok=True,parents=True)
    target=folder/"pvanet.pt"
    if target.is_file():
        if source.resolve()!=target.resolve() and digest(source)!=digest(target):
            raise FileExistsError(target)
        verify(target)
    else:
        if source.stat().st_size!=SIZE:
            raise ValueError("Source is still incomplete or a different file")
        temp=folder/"user_download_verified.pending"
        copyfile(source,temp)
        verify(temp)
        temp.replace(target)
    sha=digest(target)
    atomic_json(folder/"weight_receipt.json",dict(path=str(target),url=URL,bytes=SIZE,
                sha256=sha,crc_verified=True,downloaded_from_path=str(source)))
    emit("pva_weight_ready",checkpoint=str(target),bytes=SIZE,sha256=sha)


if __name__=="__main__":
    cli=ArgumentParser(description=__doc__)
    cli.add_argument("--download",type=Path,required=True)
    cli.add_argument("--cache",type=Path,required=True)
    args=cli.parse_args()
    adopt(args.download,args.cache)
