"""
  @Author:桌角是小黑
  @Time:2026/9/15
  @Desc:
"""
# pip install unstructured[docx]

import warnings
from typing import List

warnings.filterwarnings("ignore", message=".*langchain-community.*")  # 抑制警告

def word_loader_demo():
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader
    docs = UnstructuredWordDocumentLoader(
        # 文件路径
        file_path="../assets/sample.docx",
        # 加载模式:
        # single 返回单个 Document 对象
        # elements 按标题等元素切分文档
        mode="elements",
    ).load()
    for doc in docs[0:19]: # 从文档中间选取 30 个文档查看结构
        print(doc.page_content)
        print(doc.metadata,end="\n============\n")
if __name__ == "__main__":
	word_loader_demo()

