"""
  @Author:桌角是小黑
  @Time:2026/9/17
  @Desc:
"""
import warnings
from pymilvus import MilvusClient, DataType
from typing import List

warnings.filterwarnings("ignore", message=".*langchain-community.*")  # 抑制警告
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredWordDocumentLoader


def get_client():
    return MilvusClient(uri="http://localhost:19530", token="", )


def insert_data(client: MilvusClient, collection_name: str):
    # 构建数据，并插入到 collection 中
    # 1、加载一个文件:此处以 assets/sample.docx 文件为例

    loader = UnstructuredWordDocumentLoader("../assets/sample.docx",
                                            mode="single")
    doc_list: List[Document] = loader.load()
    # 2、切分文件
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=20,
                                                   separators=["\n\n", "\n", "。","？","!"])
    splitted_doc_list = text_splitter.split_documents(doc_list)
    splitted_doc_list = splitted_doc_list[0:50]

    # 查看当前文本列表当中最大的文本长度
    max_len = max([len(bytes(doc.page_content.encode("utf-8"))) for doc in splitted_doc_list])
    print('当前最大长度是：', max_len)

    # 3、构建向量：稠密向量，稀疏向量
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel("../assets/models/bge-m3")  # 需要安装 带 cuda 的 torch
    all_vectors = model.encode([doc.page_content for doc in splitted_doc_list],
                               return_dense=True, return_sparse=True)
    dense_vectors = all_vectors["dense_vecs"]
    sparse_vectors = all_vectors['lexical_weights']

    # 4、准备数据：组装成 List[Dict]
    insert_data_list = []
    for doc, dense_vector, sparse_vector in zip(splitted_doc_list, dense_vectors, sparse_vectors):
        insert_data_list.append({
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector,
            "metadata": doc.metadata,  # 原信息  JSON类型
            "text": doc.page_content  # 切片内容
        })

    # 5、调用 client.insert()方法，插入数据
    res = client.insert(
        collection_name=collection_name,
        data=insert_data_list
    )

    # 有多少条数据插入成功
    print(res)
    print(f"数据保存条数 {res}")


def delete_demo(client):
    res = client.delete(
        collection_name="demo_collection",
        # 过滤条件，仅删除 id 在指定范围内的实体,也可以传递其他字段的过滤条件
        filter="id in [469142657058971668, 469142657058971670]",
    )
    print(res)


if __name__ == '__main__':
    insert_data(get_client(), "demo_collection")
    # delete_demo(get_client())

"""
当前最大长度是： 1462
You're using a XLMRobertaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
{'insert_count': 20, 'ids': [469142657058971668, 469142657058971669, 469142657058971670, 469142657058971671, 469142657058971672, 469142657058971673, 469142657058971674, 469142657058971675, 469142657058971676, 469142657058971677, 469142657058971678, 469142657058971679, 469142657058971680, 469142657058971681, 469142657058971682, 469142657058971683, 469142657058971684, 469142657058971685, 469142657058971686, 469142657058971687]}
数据保存条数 {'insert_count': 20, 'ids': [469142657058971668, 469142657058971669, 469142657058971670, 469142657058971671, 469142657058971672, 469142657058971673, 469142657058971674, 469142657058971675, 469142657058971676, 469142657058971677, 469142657058971678, 469142657058971679, 469142657058971680, 469142657058971681, 469142657058971682, 469142657058971683, 469142657058971684, 469142657058971685, 469142657058971686, 469142657058971687]}
"""
"""
{'delete_count': 2}
"""