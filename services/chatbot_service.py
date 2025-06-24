import re
import jaconv
from sentence_transformers import SentenceTransformer, util
from data.code_map import company_to_code

# BERTモデルのロード（初回は時間かかる）
model = SentenceTransformer('sonoisa/sentence-bert-base-ja-mean-tokens')

# 企業名リストとベクトル化（起動時に一度だけ）
companies = list(company_to_code.keys())
company_embeddings = model.encode(companies, convert_to_tensor=True)

def normalize_text(text: str) -> str:
    text = text.lower()
    text = jaconv.z2h(text, kana=False, ascii=True, digit=True)
    text = jaconv.hira2kata(text)
    text = re.sub(r'\s+', '', text)
    return text

def get_similar_companies_bert(query: str, top_k=3, threshold=0.5):
    """
    BERT埋め込みで類似企業名を検索。
    threshold以下の類似度は除外。
    """
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, company_embeddings)[0]
    
    # 類似度順にソート
    top_results = cos_scores.topk(k=top_k)
    
    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        if score >= threshold:
            results.append((companies[idx], float(score)))
    return results

def get_stock_code_from_text(user_text: str) -> str:
    matched_companies = get_similar_companies_bert(user_text)
    
    if not matched_companies:
        return "申し訳ありませんが、その企業名は登録されていません。"
    
    if len(matched_companies) == 1:
        name = matched_companies[0][0]
        code = company_to_code[name]
        return f"{name} の銘柄コードは {code} です。"
    
    response = "類似する企業名が複数あります:\n"
    for name, score in matched_companies:
        code = company_to_code[name]
        response += f"- {name}（銘柄コード: {code}） 類似度: {score:.2f}\n"
    response += "企業名をもう少し詳しく教えてください。"
    return response
