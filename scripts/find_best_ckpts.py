import sys
from pathlib import Path

ckpt_dir=sys.argv[1]
num_ckpts=sys.argv[2]
min_or_max=sys.argv[3]

ckpt_dir = Path(ckpt_dir)
num_ckpts = int(num_ckpts)

sorted_ckpt_names = sorted(ckpt_dir.glob("checkpoint.best*"))
if min_or_max == "max":
    sorted_ckpt_names = sorted_ckpt_names[::-1]

with open(ckpt_dir / f"best_{num_ckpts}.txt", "w") as f:
    for i, ckpt_name in enumerate(sorted_ckpt_names[:num_ckpts]):
        f.write(str(ckpt_name))
        if i != num_ckpts-1:
            f.write(" ")
