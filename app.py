"""
📰 뉴스 검색 비교 & 저장 앱 (URL 검증 추가 버전)

LLM 기반 검색(Google + Gemini)과 키워드 기반 검색(네이버 뉴스 API)을 
동시에 활용해 검색 결과를 비교하고, Supabase DB에 자동 저장합니다.

v2 변경사항:
- Gemini가 환각하는 가짜 URL 문제 해결
- 모든 URL을 HTTP 요청으로 실시간 검증 후 살아있는 링크만 표시
- 프롬프트에 "URL 추측 금지" 지시 강화
"""

import streamlit as st
import pandas as pd
import json
import re
import html
import requests
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types
from supabase import create_client, Client


# ====================================================================
# 1. 페이지 기본 설정
# ====================================================================
st.set_page_config(
    page_title="뉴스 검색 비교 & 저장 앱",
    page_icon="📰",
    layout="wide",
)


# ====================================================================
# 2. 비밀 키(Secrets) 불러오기
# ====================================================================
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
NAVER_CLIENT_ID = st.secrets["NAVER_CLIENT_ID"]
NAVER_CLIENT_SECRET = st.secrets["NAVER_CLIENT_SECRET"]


# ====================================================================
# 3. 외부 서비스 클라이언트 초기화
# ====================================================================
@st.cache_resource
def init_supabase() -> Client:
    """Supabase 클라이언트 생성 (앱 전체에서 1개만 유지)"""
    return create_client(SUPABASE_URL, SUPABASE_KEY)


supabase = init_supabase()
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# ====================================================================
# 4. 헬퍼 함수
# ====================================================================
def clean_html(text: str) -> str:
    """네이버 API 응답의 HTML 태그(<b>, </b> 등)와 엔티티(&quot; &amp; 등)를 제거"""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return text.strip()


def validate_url(url: str, timeout: float = 8.0) -> bool:
    """
    URL이 실제로 존재하고 접근 가능한지 HTTP 요청으로 확인.
    - 200~399 응답: 유효
    - 404, 500 등: 무효
    - 타임아웃, 연결 거부: 무효
    
    LLM이 환각으로 만들어낸 가짜 URL을 걸러내기 위함.
    """
    if not url or not isinstance(url, str):
        return False
    if not url.startswith(("http://", "https://")):
        return False
    
    # 일부 사이트는 봇으로 인식되면 차단하므로 브라우저인 척 헤더 추가
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    
    try:
        # 1차: HEAD 요청 (본문은 안 받아서 빠름)
        r = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        # 일부 서버는 HEAD를 거부 → GET으로 재시도
        if r.status_code in (403, 405, 501):
            r = requests.get(
                url, headers=headers, timeout=timeout,
                allow_redirects=True, stream=True
            )
            r.close()
        return 200 <= r.status_code < 400
    except (requests.RequestException, OSError):
        return False


def filter_existing_urls(news_list: list, max_workers: int = 8) -> tuple:
    """
    병렬로 URL 검증 후 살아있는 뉴스만 반환.
    
    Returns:
        (valid_news_list, original_total_count)
    """
    if not news_list:
        return [], 0
    
    total = len(news_list)
    urls = [n.get("url", "") for n in news_list]
    
    # 5건을 한꺼번에 검증 → 가장 느린 한 건 시간만 소요
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        is_valid_list = list(executor.map(validate_url, urls))
    
    valid = [n for n, ok in zip(news_list, is_valid_list) if ok]
    return valid, total


def search_google_news(keyword: str) -> tuple:
    """
    Gemini API + Google Search Grounding으로 뉴스 검색·요약 + URL 검증.
    
    주의: Search Grounding과 강제 JSON 포맷은 동시 사용 불가.
         프롬프트로 JSON만 출력하도록 강하게 지시한 뒤 정규표현식으로 추출.
    
    Returns:
        (validated_news_list, total_count_from_llm)
        - validated_news_list: URL 검증을 통과한 뉴스만
        - total_count_from_llm: LLM이 처음 반환한 개수 (검증 전)
    """
    # 환각 방지 강화 프롬프트
    prompt = f"""
다음 키워드에 대한 최근 뉴스를 검색해주세요: '{keyword}'

[중요 규칙 - 반드시 준수]
1. URL은 반드시 Google Search 결과에 실제로 노출된 기사의 정확한 URL이어야 합니다.
2. 절대로 URL을 추측하거나 도메인 패턴만 보고 만들어내지 마세요. 본 적 없는 URL을 적으면 안 됩니다.
3. 5건을 억지로 채우려고 확실하지 않은 기사를 추가하지 마세요.
   검증 가능한 기사가 3건뿐이라면 3건만, 1건뿐이라면 1건만 반환하세요.
4. 의심스러운 URL이거나 출처 도메인이 모호한 경우 해당 항목은 제외하세요.

[출력 형식]
- 응답은 JSON 배열로만 (인사말, 마크다운, ```json 등 일체 금지)
- 각 항목: title, source, news_date(YYYY-MM-DD), url, summary(3~4문장)

[
  {{
    "title": "뉴스 제목",
    "source": "언론사 이름",
    "news_date": "2026-05-06",
    "url": "https://...",
    "summary": "3~4문장 요약"
  }}
]
"""
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[{"google_search": {}}],
            temperature=0.2,
        ),
    )

    raw_text = response.text or ""
    match = re.search(r"\[\s*\{.*?\}\s*\]", raw_text, re.DOTALL)
    if not match:
        return [], 0
    try:
        raw_results = json.loads(match.group(0))
    except json.JSONDecodeError:
        return [], 0
    
    # ⭐ 핵심: LLM 응답을 그대로 믿지 않고 URL 실재 여부 검증
    return filter_existing_urls(raw_results)


def search_naver_news(keyword: str) -> list:
    """네이버 검색 API(news.json)로 뉴스 5건 검색. URL 환각 없음(공식 API)."""
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {
        "query": keyword,
        "display": 5,
        "sort": "date",
    }
    response = requests.get(url, headers=headers, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    items = []
    for item in data.get("items", []):
        items.append({
            "title": clean_html(item.get("title", "")),
            "description": clean_html(item.get("description", "")),
            "link": item.get("link", ""),
            "original_link": item.get("originallink", ""),
            "pub_date": item.get("pubDate", ""),
        })
    return items


def save_google_news(keyword: str, news_list: list) -> int:
    """구글 검색 결과를 news_history 테이블에 저장. URL 중복 시 자동 제외."""
    saved_count = 0
    for news in news_list:
        record = {
            "keyword": keyword,
            "title": news.get("title"),
            "source": news.get("source"),
            "news_date": news.get("news_date"),
            "url": news.get("url"),
            "summary": news.get("summary"),
        }
        try:
            supabase.table("news_history").insert(record).execute()
            saved_count += 1
        except Exception:
            pass
    return saved_count


def save_naver_news(keyword: str, news_list: list) -> int:
    """네이버 검색 결과를 naver_news_history 테이블에 저장. link 중복 시 자동 제외."""
    saved_count = 0
    for news in news_list:
        record = {
            "keyword": keyword,
            "title": news.get("title"),
            "description": news.get("description"),
            "link": news.get("link"),
            "original_link": news.get("original_link"),
            "pub_date": news.get("pub_date"),
        }
        try:
            supabase.table("naver_news_history").insert(record).execute()
            saved_count += 1
        except Exception:
            pass
    return saved_count


# ====================================================================
# 5. Session State 초기화
# ====================================================================
for key in ["google_results", "naver_results", "compare_google", "compare_naver"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ====================================================================
# 6. 헤더 및 탭 구성
# ====================================================================
st.title("📰 뉴스 검색 비교 & 저장 앱")
st.info(
    "💡 **API 한도** — Gemini: 분당 15회 / 일 1,500회 · 네이버: 일 25,000회.  \n"
    "🛡️ **URL 검증** — 구글 검색 결과는 LLM 환각 방지를 위해 실제 접속 가능한 URL만 표시됩니다."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 구글 검색",
    "🟢 네이버 검색",
    "⚖️ 비교 검색",
    "💾 저장된 뉴스",
    "📊 통계 분석",
])


# ====================================================================
# 탭 1: 구글 검색
# ====================================================================
with tab1:
    st.subheader("🔍 구글 검색 — Gemini + Search Grounding + URL 검증")
    st.caption("LLM이 Google Search로 최신 정보를 가져와 요약합니다. 모든 URL은 실시간 검증됩니다.")

    g_keyword = st.text_input("키워드 입력 (예: AI, 전기차, 부동산)", key="g_kw")

    if st.button("구글 뉴스 검색", type="primary", key="g_btn"):
        if not g_keyword.strip():
            st.warning("키워드를 입력해주세요!")
        else:
            with st.spinner(f"'{g_keyword}' 검색 + URL 검증 중... (10~20초)"):
                try:
                    results, total = search_google_news(g_keyword)
                    if results:
                        saved = save_google_news(g_keyword, results)
                        st.session_state.google_results = (g_keyword, results, saved, total)
                    else:
                        if total > 0:
                            st.error(
                                f"⚠️ Gemini가 {total}건을 반환했지만 모든 URL이 검증에 실패했습니다. "
                                "다시 시도하거나 다른 키워드로 검색해보세요."
                            )
                        else:
                            st.error("결과를 받지 못했습니다. 다시 시도해주세요.")
                except Exception as e:
                    st.error(f"검색 중 오류 발생: {e}")

    if st.session_state.google_results:
        kw, results, saved, total = st.session_state.google_results
        
        st.success(f"✅ '{kw}' — **{len(results)}건 표시** (DB 신규 저장: {saved}건)")
        if total > len(results):
            st.caption(
                f"🛡️ Gemini가 반환한 {total}건 중 {total - len(results)}건은 "
                f"URL이 존재하지 않아(404 등) 제외했습니다."
            )

        for idx, news in enumerate(results, 1):
            with st.container(border=True):
                st.markdown(f"### {idx}. {news.get('title', '제목 없음')}")
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"📰 출처: {news.get('source', '-')}")
                with col2:
                    st.caption(f"📅 날짜: {news.get('news_date', '-')}")
                st.write(news.get("summary", ""))
                st.markdown(f"[🔗 원본 기사 읽기]({news.get('url', '#')})")


# ====================================================================
# 탭 2: 네이버 검색
# ====================================================================
with tab2:
    st.subheader("🟢 네이버 검색 — Naver News API")
    st.caption("네이버에 등록된 한국 언론사 뉴스를 발행일 순으로 가져옵니다. (URL 검증 불필요 - 공식 API)")

    n_keyword = st.text_input("키워드 입력 (예: 삼성전자, 야구)", key="n_kw")

    if st.button("네이버 뉴스 검색", type="primary", key="n_btn"):
        if not n_keyword.strip():
            st.warning("키워드를 입력해주세요!")
        else:
            with st.spinner(f"'{n_keyword}' 관련 뉴스 검색 중..."):
                try:
                    results = search_naver_news(n_keyword)
                    if results:
                        saved = save_naver_news(n_keyword, results)
                        st.session_state.naver_results = (n_keyword, results, saved)
                    else:
                        st.error("결과를 받지 못했습니다.")
                except Exception as e:
                    st.error(f"검색 중 오류 발생: {e}")

    if st.session_state.naver_results:
        kw, results, saved = st.session_state.naver_results
        st.success(f"✅ '{kw}' — 총 {len(results)}건 (DB 신규 저장: {saved}건)")

        for idx, news in enumerate(results, 1):
            with st.container(border=True):
                st.markdown(f"### {idx}. {news.get('title', '제목 없음')}")
                st.caption(f"📅 발행: {news.get('pub_date', '-')}")
                st.write(news.get("description", ""))
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"[🟢 네이버 뉴스]({news.get('link', '#')})")
                with col2:
                    if news.get("original_link"):
                        st.markdown(f"[🔗 원문 보기]({news['original_link']})")


# ====================================================================
# 탭 3: 비교 검색
# ====================================================================
with tab3:
    st.subheader("⚖️ 구글 vs 네이버 비교 검색")
    st.caption("같은 키워드로 두 검색엔진을 동시에 호출해 결과를 나란히 비교합니다.")

    c_keyword = st.text_input("비교할 키워드 입력", key="c_kw")

    if st.button("비교 검색 실행", type="primary", key="c_btn"):
        if not c_keyword.strip():
            st.warning("키워드를 입력해주세요!")
        else:
            with st.spinner("두 검색엔진 동시 호출 + 구글 URL 검증 중..."):
                try:
                    g_results, g_total = search_google_news(c_keyword)
                    n_results = search_naver_news(c_keyword)
                    g_saved = save_google_news(c_keyword, g_results) if g_results else 0
                    n_saved = save_naver_news(c_keyword, n_results) if n_results else 0
                    st.session_state.compare_google = (c_keyword, g_results, g_saved, g_total)
                    st.session_state.compare_naver = (c_keyword, n_results, n_saved)
                except Exception as e:
                    st.error(f"검색 중 오류 발생: {e}")

    if st.session_state.compare_google and st.session_state.compare_naver:
        col_g, col_n = st.columns(2)

        # 구글 결과 (왼쪽)
        with col_g:
            st.markdown("### 🔍 구글 (Gemini)")
            kw, results, saved, total = st.session_state.compare_google
            invalid_count = total - len(results)
            caption = f"{len(results)}건 표시 (저장: {saved})"
            if invalid_count > 0:
                caption += f" · 검증 실패 {invalid_count}건 제외"
            st.caption(caption)
            for idx, news in enumerate(results, 1):
                with st.container(border=True):
                    st.markdown(f"**{idx}. {news.get('title', '')}**")
                    st.caption(f"{news.get('source', '-')} · {news.get('news_date', '-')}")
                    summary = news.get("summary", "")
                    if len(summary) > 120:
                        summary = summary[:120] + "..."
                    st.write(summary)
                    st.markdown(f"[원본]({news.get('url', '#')})")

        # 네이버 결과 (오른쪽)
        with col_n:
            st.markdown("### 🟢 네이버")
            kw, results, saved = st.session_state.compare_naver
            st.caption(f"{len(results)}건 (저장: {saved})")
            for idx, news in enumerate(results, 1):
                with st.container(border=True):
                    st.markdown(f"**{idx}. {news.get('title', '')}**")
                    st.caption(f"{news.get('pub_date', '-')}")
                    desc = news.get("description", "")
                    if len(desc) > 120:
                        desc = desc[:120] + "..."
                    st.write(desc)
                    st.markdown(f"[원본]({news.get('original_link') or news.get('link', '#')})")

        # 비교 인사이트
        st.markdown("---")
        st.markdown("### 📝 검색 결과 비교 인사이트")

        _, g_list, _, g_total = st.session_state.compare_google
        _, n_list, _ = st.session_state.compare_naver

        g_urls = {n.get("url", "") for n in g_list if n.get("url")}
        n_urls = {n.get("link", "") for n in n_list if n.get("link")}
        n_orig = {n.get("original_link", "") for n in n_list if n.get("original_link")}
        overlap_count = len(g_urls & (n_urls | n_orig))

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("구글 결과", f"{len(g_list)}건")
        with m2:
            st.metric("구글 환각", f"{g_total - len(g_list)}건")
        with m3:
            st.metric("네이버 결과", f"{len(n_list)}건")
        with m4:
            st.metric("URL 중복", f"{overlap_count}건")

        st.info(
            "**구글 (Gemini + Search Grounding)** — 의미·맥락 기반 검색. LLM이 자동 요약을 생성하며 "
            "다양한 도메인(블로그·영문 매체 포함)이 결과에 포함됩니다. "
            "단, LLM 특성상 URL을 환각하는 경우가 있어 본 앱에서는 모든 URL을 HTTP 요청으로 검증 후 "
            "실제 접속 가능한 것만 표시합니다.\n\n"
            "**네이버 (News API)** — 키워드 매칭 기반. 네이버 뉴스에 등록된 한국 언론사 위주의 "
            "결과를 발행일 순으로 반환합니다. 공식 API라 URL 환각이 없으며 본문 일부(description)만 제공됩니다.\n\n"
            "**중복이 적은 이유** — 두 엔진의 인덱싱 범위와 정렬 기준이 달라 같은 키워드여도 "
            "노출되는 기사 풀이 거의 겹치지 않습니다."
        )


# ====================================================================
# 탭 4: 저장된 뉴스 보기
# ====================================================================
with tab4:
    st.subheader("💾 데이터베이스에 저장된 뉴스 조회")

    source = st.radio(
        "소스 선택",
        ["구글 (Gemini)", "네이버"],
        horizontal=True,
        key="source_radio",
    )
    filter_kw = st.text_input("키워드 필터 (비워두면 전체)", key="filter_kw")

    table_name = "news_history" if source.startswith("구글") else "naver_news_history"

    try:
        query = supabase.table(table_name).select("*").order("created_at", desc=True)
        if filter_kw.strip():
            query = query.eq("keyword", filter_kw.strip())
        response = query.execute()
        data = response.data

        if data:
            df = pd.DataFrame(data)
            st.caption(f"총 {len(df)}건")
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 현재 결과 CSV 다운로드",
                data=csv,
                file_name=f"{table_name}_{filter_kw or 'all'}.csv",
                mime="text/csv",
            )
        else:
            st.info("저장된 데이터가 없습니다. 다른 탭에서 먼저 검색해주세요.")
    except Exception as e:
        st.error(f"조회 오류: {e}")


# ====================================================================
# 탭 5: 통계 분석
# ====================================================================
with tab5:
    st.subheader("📊 검색 통계 대시보드")

    try:
        g_data = supabase.table("news_history").select("*").execute().data
        n_data = supabase.table("naver_news_history").select("*").execute().data

        if not g_data and not n_data:
            st.info("저장된 데이터가 없습니다. 다른 탭에서 먼저 검색해주세요.")
        else:
            df_g = (
                pd.DataFrame(g_data)[["keyword", "created_at"]]
                if g_data else pd.DataFrame(columns=["keyword", "created_at"])
            )
            df_n = (
                pd.DataFrame(n_data)[["keyword", "created_at"]]
                if n_data else pd.DataFrame(columns=["keyword", "created_at"])
            )
            df_g["source"] = "구글"
            df_n["source"] = "네이버"

            df_all = pd.concat([df_g, df_n], ignore_index=True)
            df_all["created_at"] = pd.to_datetime(df_all["created_at"])
            df_all["date"] = df_all["created_at"].dt.date

            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.metric("구글 누적", len(df_g))
            with mc2:
                st.metric("네이버 누적", len(df_n))
            with mc3:
                st.metric("키워드 종류", df_all["keyword"].nunique())
            with mc4:
                st.metric("총 저장 건수", len(df_all))

            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🌐 소스별 누적 건수**")
                source_counts = df_all["source"].value_counts()
                st.bar_chart(source_counts)

            with col2:
                st.markdown("**🔑 키워드별 누적 검색 건수 (Top 10)**")
                kw_counts = df_all["keyword"].value_counts().head(10)
                st.bar_chart(kw_counts)

            st.markdown("**📅 일자별 저장 건수 (소스별)**")
            daily = df_all.groupby(["date", "source"]).size().unstack(fill_value=0)
            st.bar_chart(daily)
    except Exception as e:
        st.error(f"통계 조회 오류: {e}")
