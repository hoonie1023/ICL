from datasets import load_from_disk
# 把下面的路径换成你服务器上的实际路径
dataset = load_from_disk("./mini_mixed_dataset") 
print(dataset)

# --- 追加在 test.py 的末尾 ---

# 1. 抽查第一条数据 (属于 A-OKVQA)
print("\n" + "="*40)
print("🔍 抽查样本 1: 多模态 VQA")
sample_vqa = dataset[1] # 取第 1 条
print(f"来源: {sample_vqa['dataset_source']}")
print(f"Query: {sample_vqa['query']}")
print(f"Label: {sample_vqa['label']}")
print("正在尝试打开真实图片...")
sample_vqa['image'].show()  # 这会调用你电脑自带的看图软件弹出一张真实的图片

# 2. 抽查第501条数据 (属于 CSQA 的第一条)
print("\n" + "="*40)
print("🔍 抽查样本 2: 纯文本 CSQA")
sample_csqa = dataset[501] # 取第 501 条
print(f"来源: {sample_csqa['dataset_source']}")
print(f"Query: {sample_csqa['query']}")
print(f"Label: {sample_csqa['label']}")
print("正在尝试打开占位图片...")
sample_csqa['image'].show()  # 这应该会弹出一张 224x224 的全黑正方形图片