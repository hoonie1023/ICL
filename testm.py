from datasets import load_from_disk
ds = load_from_disk("./m2iv_distill_dataset")
print(ds[0])
print(len(ds))