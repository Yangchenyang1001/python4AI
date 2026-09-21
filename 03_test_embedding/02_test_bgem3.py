"""
  @Author:桌角是小黑
  @Time:2026/9/17
  @Desc:
"""
# pip install FlagEmbedding==1.3.5
# pip install -U FlagEmbedding
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel(model_name_or_path=r"..\assets\models\bge-m3")
res = model.encode(
    ["标量字段通常用来存储一些元数据，并可以在搜索时通过元数据进行过滤"],
    return_sparse=True,
    return_dense=True
)
print('encode 结果为：',res,end='\n\n')
"""
encode 结果为： {'dense_vecs': array([[-0.02839014, -0.05221429, -0.08404333, ...,  0.01279993,
         0.00544309, -0.03865913]], shape=(1, 1024), dtype=float32), 'lexical_weights': [defaultdict(<class 'int'>, {'6': np.float32(0.06473796), '23204': np.float32(0.26163626), '3272': np.float32(0.2870151), '7234': np.float32(0.18502101), '12002': np.float32(0.26074505), '17072': np.float32(0.17660604), '140278': np.float32(0.17971733), '165497': np.float32(0.20325246), '4321': np.float32(0.061180353), '2954': np.float32(0.16814002), '12833': np.float32(0.19002178), '2672': np.float32(0.048890904), '54093': np.float32(0.07285758), '63449': np.float32(0.21043669), '4511': np.float32(0.057723396), '3327': np.float32(0.05733729), '3074': np.float32(0.051461495), '193956': np.float32(0.22671731)})], 'colbert_vecs': None}
"""
# 1、打印稀疏向量
print('稀疏向量为：',res["lexical_weights"],end='\n\n')
# 2、将稀疏向量当中的id转换为token，并打印
sparse_vecs = model.convert_id_to_token(res["lexical_weights"])
print('稀疏向量转换为 token 后的结果为：',sparse_vecs,end='\n\n')
# 3、打印稠密向量
print('稠密向量为：',res["dense_vecs"],end='\n\n')