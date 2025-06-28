import re
import jaconv
from sentence_transformers import SentenceTransformer, util
from data.code_map import company_to_code
from rapidfuzz import process

# BERTモデルのロード（初回は時間がかかる）
model = SentenceTransformer('sonoisa/sentence-bert-base-ja-mean-tokens')

# 企業名リストとベクトル化（起動時に一度だけ）
companies = list(company_to_code.keys())
company_embeddings = model.encode(companies, convert_to_tensor=True)

# ---------------------------
# 正規化ユーティリティ関数
# ---------------------------
def normalize_text(text: str) -> str:
    text = text.lower()
    text = jaconv.z2h(text, kana=True, ascii=True, digit=True)  # 全角→半角
    text = jaconv.hira2kata(text)  # ひらがな→カタカナ
    text = re.sub(r'\s+', '', text)
    return text

# ------------------------------------
# 類似企業検索（BERTベース＋閾値あり）
# ------------------------------------
def get_similar_companies_bert(query: str, top_k=3, threshold=0.5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, company_embeddings)[0]
    
    top_results = cos_scores.topk(k=top_k)
    results = []

    for score, idx in zip(top_results.values, top_results.indices):
        if score >= threshold:
            results.append((companies[idx], float(score)))

    return results

# --------------------------------------
# rapidfuzzによるバックアップマッチ処理
# --------------------------------------
def fuzzy_match_company(user_text: str, score_threshold=80):
    normalized_input = normalize_text(user_text)
    normalized_dict = {
        normalize_text(name): code
        for name, code in company_to_code.items()
    }
    match, score, _ = process.extractOne(normalized_input, normalized_dict.keys())
    if score >= score_threshold:
        return match, normalized_dict[match]
    return None, None

# ------------------------------------------
# ユーザー入力に基づく銘柄コードの抽出
# ------------------------------------------
def get_stock_code_from_text(user_text: str) -> str:
    # ステップ1: BERTで検索
    matched_companies = get_similar_companies_bert(user_text)

    if matched_companies:
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
    
    # ステップ2: fuzzy match fallback
    fuzzy_name, fuzzy_code = fuzzy_match_company(user_text)
    if fuzzy_code:
        return f"{fuzzy_name} の銘柄コードは {fuzzy_code} です。"
    
    # どちらにもヒットしない場合
    return "申し訳ありませんが、その企業名は登録されていません。"
