import argparse
from pathlib import Path
import os

import torch
from einops import rearrange

from .emb import g2p, qnt
from .utils import to_device


def main():
    parser = argparse.ArgumentParser("VALL-E TTS")
    parser.add_argument("text")
    parser.add_argument("reference", type=Path)
    parser.add_argument("--ar-ckpt", type=Path, default="zoo/ar.pt")
    parser.add_argument("--nar-ckpt", type=Path, default="zoo/nar.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ar = torch.load(args.ar_ckpt).to(args.device)
    nar = torch.load(args.nar_ckpt).to(args.device)

    symmap = ar.phone_symmap

    proms = qnt.encode_from_file(args.reference)
    proms = rearrange(proms, "1 l t -> t l")

    phns = torch.tensor([symmap[p] for p in g2p.encode(args.text)])

    proms = to_device(proms, args.device)
    phns = to_device(phns, args.device)

    resp_list = ar(text_list=[phns], proms_list=[proms])
    resps_list = [r.unsqueeze(-1) for r in resp_list]

    resps_list = nar(text_list=[phns], proms_list=[proms], resps_list=resps_list)

    if not os.path.exists("out_audios"):
        os.mkdir("out_audios")

    fpath_without_ext = os.path.splitext(str(args.reference))[0]
    speaker_name = os.path.normpath(fpath_without_ext).split(os.sep)[-1]

    qnt.decode_to_file(resps=resps_list[0], path=os.path.join("out_audios", f"{speaker_name}_syn.wav"))
    print(f"{speaker_name}_syn.wav", "saved.")


if __name__ == "__main__":
    main()
