"""
  @Author:桌角是小黑
  @Time:2026/9/20
  @Desc:
"""
from typing import Tuple, Dict, List

from langchain_ollama import ChatOllama
from pymilvus import MilvusClient

"""
RAG完整链路测试
"""


def get_milvus_client(uri: str = "http://localhost:19530", token: str = ""):
    from pymilvus import MilvusClient
    return MilvusClient(uri=uri, token=token)


def get_bge_m3_model():
    from FlagEmbedding import BGEM3FlagModel
    return BGEM3FlagModel(model_name_or_path="../assets/models/bge-m3")


def encode_query(model, query: str) -> Tuple[List[float], Dict[int, float]]:
    all_embeddings = model.encode([query], return_dense=True, return_sparse=True)
    dense_vec = all_embeddings["dense_vecs"][0]
    sparse_raw = all_embeddings["lexical_weights"][0]
    return dense_vec, sparse_raw


def print_hits(title: str, hits: List[dict]):
    print("\n" + "=" * 20)
    print(title)
    print("=" * 20)
    for i, hit in enumerate(hits, start=1):
        entity = hit.get("entity", {})
        print(
            {
                "rank": i,
                "id": entity.get("id"),
                "distance": hit.get("distance"),
                "text": entity.get("text"),
                "metadata": entity.get("metadata"),
            }
        )


def hybrid_vector_search_example_rrf(client, query: str, limit: int = 5, collection_name: str = 'demo_collection'):
    from pymilvus import AnnSearchRequest, RRFRanker
    model = get_bge_m3_model()
    dense_vec, sparse_vec = encode_query(model, query)
    dense_req = AnnSearchRequest(
        data=[dense_vec],
        anns_field="dense_vector",
        param={"metric_type": "L2"},
        limit=limit,
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_vec],
        anns_field="sparse_vector",
        param={"metric_type": "IP"},
        limit=limit,
    )
    # 混合检索
    results = client.hybrid_search(
        collection_name=collection_name,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(k=60),
        limit=limit,
        output_fields=["id", "text", "metadata"],
    )
    print_hits("混合向量检索（RRF 融合稠密+稀疏）", results[0])
    """
        ====================
        混合向量检索（RRF 融合稠密+稀疏）
        ====================
        {'rank': 1, 'id': 469142657058971271, 'distance': 0.016393441706895828, 'text': '第三十三条\u3000具有完全民事行为能力的成年人，可以与其近亲属、其他愿意担任监护人的个人或者组织事先协商，以书面形式确定自己的监护人，在自己丧失或者部分丧失民事行为能力时，由该监护人履行监护职责。\n\n第三十四条\u3000监护人的职责是代理被监护人实施民事法律行为，保护被监护人的人身权利、财产权利以及其他合法权益等。\n\n监护人依法履行监护职责产生的权利，受法律保护。\n\n监护人不履行监护职责或者侵害被监护人合法权益的，应当承担法律责任。\n\n因发生突发事件等紧急情况，监护人暂时无法履行监护职责，被监护人的生活处于无人照料状态的，被监护人住所地的居民委员会、村民委员会或者民政部门应当为被监护人安排必要的临时生活照料措施。\n\n第三十五条\u3000监护人应当按照最有利于被监护人的原则履行监护职责。监护人除为维护被监护人利益外，不得处分被监护人的财产。\n\n未成年人的监护人履行监护职责，在作出与被监护人利益有关的决定时，应当根据被监护人的年龄和智力状况，尊重被监护人的真实意愿。', 'metadata': {'source': '../assets/sample.docx'}}
        {'rank': 2, 'id': 469142657058971280, 'distance': 0.016393441706895828, 'text': '第二十五条\u3000自然人以户籍登记或者其他有效身份登记记载的居所为住所；经常居所与住所不一致的，经常居所视为住所。\n\n第二节\u3000监护\n\n第二十六条\u3000父母对未成年子女负有抚养、教育和保护的义务。\n\n成年子女对父母负有赡养、扶助和保护的义务。\n\n第二十七条\u3000父母是未成年子女的监护人。\n\n未成年人的父母已经死亡或者没有监护能力的，由下列有监护能力的人按顺序担任监护人：\n\n（一）祖父母、外祖父母；\n\n（二）兄、姐；\n\n（三）其他愿意担任监护人的个人或者组织，但是须经未成年人住所地的居民委员会、村民委员会或者民政部门同意。\n\n第二十八条\u3000无民事行为能力或者限制民事行为能力的成年人，由下列有监护能力的人按顺序担任监护人：\n\n（一）配偶；\n\n（二）父母、子女；\n\n（三）其他近亲属；\n\n（四）其他愿意担任监护人的个人或者组织，但是须经被监护人住所地的居民委员会、村民委员会或者民政部门同意。\n\n第二十九条\u3000被监护人的父母担任监护人的，可以通过遗嘱指定监护人。\n\n第三十条\u3000依法具有监护资格的人之间可以协议确定监护人。协议确定监护人应当尊重被监护人的真实意愿。', 'metadata': {'source': '../assets/sample.docx'}}
        {'rank': 3, 'id': 469142657058971269, 'distance': 0.016129031777381897, 'text': '第二十五条\u3000自然人以户籍登记或者其他有效身份登记记载的居所为住所；经常居所与住所不一致的，经常居所视为住所。\n\n第二节\u3000监护\n\n第二十六条\u3000父母对未成年子女负有抚养、教育和保护的义务。\n\n成年子女对父母负有赡养、扶助和保护的义务。\n\n第二十七条\u3000父母是未成年子女的监护人。\n\n未成年人的父母已经死亡或者没有监护能力的，由下列有监护能力的人按顺序担任监护人：\n\n（一）祖父母、外祖父母；\n\n（二）兄、姐；\n\n（三）其他愿意担任监护人的个人或者组织，但是须经未成年人住所地的居民委员会、村民委员会或者民政部门同意。\n\n第二十八条\u3000无民事行为能力或者限制民事行为能力的成年人，由下列有监护能力的人按顺序担任监护人：\n\n（一）配偶；\n\n（二）父母、子女；\n\n（三）其他近亲属；\n\n（四）其他愿意担任监护人的个人或者组织，但是须经被监护人住所地的居民委员会、村民委员会或者民政部门同意。\n\n第二十九条\u3000被监护人的父母担任监护人的，可以通过遗嘱指定监护人。\n\n第三十条\u3000依法具有监护资格的人之间可以协议确定监护人。协议确定监护人应当尊重被监护人的真实意愿。', 'metadata': {'source': '../assets/sample.docx'}}
        {'rank': 4, 'id': 469142657058971282, 'distance': 0.016129031777381897, 'text': '第三十三条\u3000具有完全民事行为能力的成年人，可以与其近亲属、其他愿意担任监护人的个人或者组织事先协商，以书面形式确定自己的监护人，在自己丧失或者部分丧失民事行为能力时，由该监护人履行监护职责。\n\n第三十四条\u3000监护人的职责是代理被监护人实施民事法律行为，保护被监护人的人身权利、财产权利以及其他合法权益等。\n\n监护人依法履行监护职责产生的权利，受法律保护。\n\n监护人不履行监护职责或者侵害被监护人合法权益的，应当承担法律责任。\n\n因发生突发事件等紧急情况，监护人暂时无法履行监护职责，被监护人的生活处于无人照料状态的，被监护人住所地的居民委员会、村民委员会或者民政部门应当为被监护人安排必要的临时生活照料措施。\n\n第三十五条\u3000监护人应当按照最有利于被监护人的原则履行监护职责。监护人除为维护被监护人利益外，不得处分被监护人的财产。\n\n未成年人的监护人履行监护职责，在作出与被监护人利益有关的决定时，应当根据被监护人的年龄和智力状况，尊重被监护人的真实意愿。', 'metadata': {'source': '../assets/sample.docx'}}
        {'rank': 5, 'id': 469142657058971250, 'distance': 0.01587301678955555, 'text': '第二十五条\u3000自然人以户籍登记或者其他有效身份登记记载的居所为住所；经常居所与住所不一致的，经常居所视为住所。\n\n第二节\u3000监护\n\n第二十六条\u3000父母对未成年子女负有抚养、教育和保护的义务。\n\n成年子女对父母负有赡养、扶助和保护的义务。\n\n第二十七条\u3000父母是未成年子女的监护人。\n\n未成年人的父母已经死亡或者没有监护能力的，由下列有监护能力的人按顺序担任监护人：\n\n（一）祖父母、外祖父母；\n\n（二）兄、姐；\n\n（三）其他愿意担任监护人的个人或者组织，但是须经未成年人住所地的居民委员会、村民委员会或者民政部门同意。\n\n第二十八条\u3000无民事行为能力或者限制民事行为能力的成年人，由下列有监护能力的人按顺序担任监护人：\n\n（一）配偶；\n\n（二）父母、子女；\n\n（三）其他近亲属；\n\n（四）其他愿意担任监护人的个人或者组织，但是须经被监护人住所地的居民委员会、村民委员会或者民政部门同意。\n\n第二十九条\u3000被监护人的父母担任监护人的，可以通过遗嘱指定监护人。\n\n第三十条\u3000依法具有监护资格的人之间可以协议确定监护人。协议确定监护人应当尊重被监护人的真实意愿。', 'metadata': {'source': '../assets/sample.docx'}}
    """
    return results


def rag_demo(client: MilvusClient, query):
    from langchain_openai import ChatOpenAI

    # 加载模型
    llm = ChatOllama(
        model="qwen2.5:7b", base_url="http://localhost:11434",
    )
    print("大模型客户端输出", llm)
    # llm = ChatOpenAI(model_name="gpt-4o-mini")
    retrieval_res = hybrid_vector_search_example_rrf(client=client, query=query)
    print("召回与精排", retrieval_res)
    # 构建上下文
    context = "\n".join([hit["entity"]["text"] for hit in retrieval_res[0]])
    message_list = [
        {"role": "system",
         "content": "你是一个专业的法律问答机器人，请根据上下文回答问题，当上下文无法回答问题时，请回答“根据上下文无法回答该问题"},
        {"role": "user", "content": f"根据以下上下文回答问题：{context}\\n问题：{query}"}
    ]

    # 生成文本
    res = llm.invoke(message_list)
    print("输出结果", res)


if __name__ == "__main__":
    client = get_milvus_client()
    rag_demo(client, '未成年人的监护人履行什么职责？ ')
