"""
  @Author:桌角是小黑
  @Time:2026/9/17
  @Desc: 创建milvus集合
"""
# pip install pymilvus
from pymilvus import MilvusClient, CollectionSchema


# 创建客户端
def get_milvus_client():
    client = MilvusClient(
        uri="http://localhost:19530",
        token="",
    )
    res = client.list_collections()
    print(res)
    return client


# 创建集合schema结构
def build_schema(client: MilvusClient) -> CollectionSchema:
    from pymilvus import DataType
    # 自动为 id 字段赋值
    return (client.create_schema(auto_id=True)
            .add_field(field_name="id", datatype=DataType.INT64, is_primary=True)  # 添加 id 字段，类型为整数，设置为主键
            .add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR,
                       dim=1024)  # 添加 稠密向量  dense_vector字段，类型为浮点数向量，维度为 1024
            .add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1500)  # 添加 text 字段，类型为字符串，最大长度为 1500
            .add_field(field_name="metadata", datatype=DataType.JSON)  # 添加 metadata 字段，类型为 JSON
            .add_field(field_name="sparse_vector",
                       datatype=DataType.SPARSE_FLOAT_VECTOR))  # 添加稀疏向量 sparse_vector 字段 数据类型SPARSE_FLOAT_VECTOR


def build_index(client: MilvusClient):
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",  # 建立索引的字段
        index_type="HNSW",  # 索引类型  分层导航小世界
        metric_type="L2",  # 向量相似度度量方式  欧几里得距离    也可以选择余弦相似度 COSIN
    )
    index_params.add_index(
        field_name="sparse_vector",  # 字段名称
        index_type="SPARSE_INVERTED_INDEX",  # 倒排索引
        metric_type="IP",  # 内积 -无穷 ~正无穷   ====》 -1 ~ +1
    )
    return index_params


def create_collection(client: MilvusClient, collection_name: str = "demo_collection"):
    from pprint import pprint
    client.drop_collection(collection_name=collection_name)
    if not client.has_collection(collection_name=collection_name):
        print("collection demo_collection not exists, create it")
        client.create_collection(
            collection_name=collection_name,  # collection 名称
            schema=build_schema(client),  # collection 的 schema
            index_params=build_index(client),  # collection 的 index
        )
        # 查看 collection
        print(client.list_collections())
        # 查看 collection 描述
        print(client.describe_collection(collection_name=collection_name))


if __name__ == '__main__':
    create_collection(get_milvus_client())


""" output
[]
collection demo_collection not exists, create it
['demo_collection']
{'collection_name': 'demo_collection', 'auto_id': True, 'num_shards': 1, 'description': '', 'fields': [{'field_id': 100, 'name': 'id', 'description': '', 'type': <DataType.INT64: 5>, 'params': {}, 'auto_id': True, 'is_primary': True}, {'field_id': 101, 'name': 'dense_vector', 'description': '', 'type': <DataType.FLOAT_VECTOR: 101>, 'params': {'dim': 1024}}, {'field_id': 102, 'name': 'text', 'description': '', 'type': <DataType.VARCHAR: 21>, 'params': {'max_length': 1500}}, {'field_id': 103, 'name': 'metadata', 'description': '', 'type': <DataType.JSON: 23>, 'params': {}}, {'field_id': 104, 'name': 'sparse_vector', 'description': '', 'type': <DataType.SPARSE_FLOAT_VECTOR: 104>, 'params': {}}], 'functions': [], 'aliases': [], 'collection_id': 469142657057960212, 'consistency_level': 2, 'properties': {}, 'num_partitions': 1, 'enable_dynamic_field': False, 'enable_namespace': False, 'created_timestamp': 469143312287399939, 'update_timestamp': 469143312287399939}

"""