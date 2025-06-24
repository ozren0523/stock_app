import re
import jaconv
import difflib
from data.code_map import company_to_code

def normalize_text(text: str) -> str:
    text = text.lower()
    text = jaconv.z2h(text, kana=False, ascii=True, digit=True)
    text = jaconv.hira2kata(text)
    text = re.sub(r'\s+', '', text)
    return text

def get_similar_companies(query: str, companies: list, n=3, cutoff=0.6) -> list:
    query_norm = normalize_text(query)
    companies_norm = {name: normalize_text(name) for name in companies}
    norm_names = list(companies_norm.values())
    matches_norm = difflib.get_close_matches(query_norm, norm_names, n=n, cutoff=cutoff)
    matches = [name for name, norm in companies_norm.items() if norm in matches_norm]
    return matches

def get_stock_code_from_text(user_text: str) -> str:
    companies = list(company_to_code.keys())
    matched_companies = get_similar_companies(user_text, companies)

    if not matched_companies:
        return "申し訳ありませんが、その企業名は登録されていません。"

    if len(matched_companies) == 1:
        name = matched_companies[0]
        code = company_to_code[name]
        return f"{name} の銘柄コードは {code} です。"

    response = "類似する企業名が複数あります:\n"
    for name in matched_companies:
        code = company_to_code[name]
        response += f"- {name}（銘柄コード: {code}）\n"
    response += "企業名をもう少し詳しく教えてください。"
    return response
