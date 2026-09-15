"""
  @Author:桌角是小黑
  @Time:2026/9/15
  @Desc:
"""
# pip install markdown langchain_community unstructured[md]


import warnings
from typing import List

warnings.filterwarnings("ignore", message=".*langchain-community.*")  # 抑制警告

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document


def markdown_loader_demo():
    # mode="single"（默认） 把整个 Markdown 文件当作一个文档 返回 List[Document]，长度为 1
    # mode="elements" 按语义元素拆分成多个文档（标题、段落、列表、表格等各自独立） 返回 List[Document]，每个元素一个 Document
    loader = UnstructuredMarkdownLoader(
        file_path="../assets/sample.md",
        encodings="utf-8",
        mode="elements"
    )
    docs: List[Document] = loader.load()
    print(f"成功加载了 {len(docs)} 个文档片段！")
    # for doc in docs:
    # 	print(doc.page_content)
    # 	print("=" * 50)
    return docs


def enrich_document_info(document_list: list[Document]):
    # 栈， 用来存放历史的标题 [(category_depth,title)]
    stack = []
    category_depth = 0
    enriched_list = []  # 用来存储处理后的文档
    for doc in document_list:
        if doc.metadata['category'] == "Title":
            category_depth = doc.metadata['category_depth']
            while stack and stack[-1][0] >= category_depth:
                stack.pop()
            stack.append((category_depth, doc.page_content))
        else:
            title_level_info = " > ".join([title for _, title in stack])
            doc_dict = {
                "meta_data": doc.metadata,
                "content": title_level_info + "\n" + doc.page_content
            }
            enriched_list.append(doc_dict)
    return enriched_list


if __name__ == "__main__":
    document = markdown_loader_demo()
    print(document)
    enrich_doc_list = enrich_document_info(document)
    for doc_dict in enrich_doc_list:
        print(doc_dict['content'])
        print("=" * 50)
