"""
📰 뉴스 검색 비교 & 저장 앱

LLM 기반 검색(Google + Gemini)과 키워드 기반 검색(네이버 뉴스 API)을 
동시에 활용해 검색 결과를 비교하고, Supabase DB에 자동 저장합니다.
"""

import streamlit as st
import pandas as pd
import json
import re
import html
import requests
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
# Streamlit Community Cloud의 Secrets 또는 .streamlit/secrets.toml에서 읽음
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
NAVER_CLIENT_ID = st.secrets["NAVER_CLIENT_ID"]
NAVER_CLIENT_SECRET = st.secrets["NAVER_CLIENT_SECRET"]


# ====================================================================
# 3. 외부 서비스 클라이언트 초기화
# ====================================================================
@st.cache_resource  # 매번 재연결하지 않도록 캐싱
def init_supabase() -> Client:
    """Supabase 클라이언트 생성 (앱 전체에서 1개만 유지)"""
    return create_client(SUPABASE_URL, SUPABASE_KEY)


supabase = init_supabase()
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# ====================================================================
# 4. 헬퍼 함수
# ====================================================================
def clean_html(text: str) -> str:
    """네이버 API 응답의 HTML 태그(<b>, </b> 등)와 HTML 엔티티(&quot; &amp; 등)를 제거"""
    if not text:
        return ""
    # <태그> 형태 제거
    text = re.sub(r"<[^>]+>", "", text)
    # &quot; &amp; 같은 엔티티를 일반 문자로 변환
    text = html.unescape(text)
    return text.strip()


def search_google_news(keyword: str) -> list:
    """
    Gemini API + Google Search Grounding으로 최신 뉴스 5건 검색·요약.
    
    주의: Search Grounding과 강제 JSON 포맷은 동시 사용 불가.
         프롬프트로 JSON만 출력하도록 강하게 지시한 뒤 정규표현식으로 추출.
    """
    prompt = f"""
다음 키워드에 대한 가장 최신 뉴스 5건을 검색하고 요약해주세요: '{keyword}'

[요구사항]
1. Google Search를 사용해 최신 정보를 가져오세요.
2. 각 뉴스별로 제목(title), 출처(source), 날짜(news_date), 원본 URL(url),
   3~4문장의 요약(summary)을 작성하세요.
3. 응답은 반드시 아래 형태의 JSON 배열로만 출력해야 합니다.
   인사말이나 마크다운(```json 등)은 절대 포함하지 마세요.

[
  {{
    "title": "뉴스 제목",
    "source": "언론사 이름",
    "news_date": "YYYY-MM-DD",
    "url": "https://...",
    "summary": "3~4문장의 요약 내용"
  }}
]
"""
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[{"google_search": {}}],  # Google Search Grounding 활성화
            temperature=0.2,  # 일관된 JSON 출력을 위해 낮춤
        ),
    )

    raw_text = response.text or ""
    # 응답 텍스트에서 JSON 배열 부분만 추출
    match = re.search(r"\[\s*\{.*?\}\s*\]", raw_text, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []


def search_naver_news(keyword: str) -> list:
    """네이버 검색 API(news.json)로 뉴스 5건 검색"""
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {
        "query": keyword,
        "display": 5,
        "sort": "date",  # 최신순. 'sim'으로 바꾸면 정확도 순.
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
            # url UNIQUE 제약 위반(중복) 등은 조용히 무시
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
# 5. Session State 초기화 (다운로드 버튼 눌러도 결과 유지)
# ====================================================================
for key in ["google_results", "naver_results", "compare_google", "compare_naver"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ====================================================================
# 6. 헤더 및 탭 구성
# ====================================================================
st.title("📰 뉴스 검색 비교 & 저장 앱")
st.info(
    "💡 **API 한도 안내** — Gemini: 분당 15회 / 일 1,500회 · "
    "네이버: 일 25,000회. 검색 결과는 Supabase에 자동 저장됩니다."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 구글 검색",
    "🟢 네이버 검색",
    "⚖️ 비교 검색",
    "💾 저장된 뉴스",
    "📊 통계 분석",
])


# ====================================================================
# 탭 1: 구글 검색 (Gemini + Search Grounding)
# ====================================================================
with tab1:
    st.subheader("🔍 구글 검색 — Gemini + Search Grounding")
    st.caption("LLM이 Google Search로 최신 정보를 가져와 요약합니다.")

    g_keyword = st.text_input("키워드 입력 (예: AI, 전기차, 부동산)", key="g_kw")

    if st.button("구글 뉴스 검색", type="primary", key="g_btn"):
        if not g_keyword.strip():
            st.warning("키워드를 입력해주세요!")
        else:
            with st.spinner(f"'{g_keyword}' 관련 뉴스 검색 중... (5~15초)"):
                try:
                    results = search_google_news(g_keyword)
                    if results:
                        saved = save_google_news(g_keyword, results)
                        st.session_state.google_results = (g_keyword, results, saved)
                    else:
                        st.error("결과를 받지 못했습니다. 다시 시도해주세요.")
                except Exception as e:
                    st.error(f"검색 중 오류 발생: {e}")

    if st.session_state.google_results:
        kw, results, saved = st.session_state.google_results
        st.success(f"✅ '{kw}' — 총 {len(results)}건 (DB 신규 저장: {saved}건)")

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
# 탭 2: 네이버 검색 (Naver News API)
# ====================================================================
with tab2:
    st.subheader("🟢 네이버 검색 — Naver News API")
    st.caption("네이버에 등록된 한국 언론사 뉴스를 발행일 순으로 가져옵니다.")

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
# 탭 3: 비교 검색 (구글 vs 네이버)
# ====================================================================
with tab3:
    st.subheader("⚖️ 구글 vs 네이버 비교 검색")
    st.caption("같은 키워드로 두 검색엔진을 동시에 호출해 결과를 나란히 비교합니다.")

    c_keyword = st.text_input("비교할 키워드 입력", key="c_kw")

    if st.button("비교 검색 실행", type="primary", key="c_btn"):
        if not c_keyword.strip():
            st.warning("키워드를 입력해주세요!")
        else:
            with st.spinner("두 검색엔진 동시 호출 중..."):
                try:
                    g_results = search_google_news(c_keyword)
                    n_results = search_naver_news(c_keyword)
                    g_saved = save_google_news(c_keyword, g_results) if g_results else 0
                    n_saved = save_naver_news(c_keyword, n_results) if n_results else 0
                    st.session_state.compare_google = (c_keyword, g_results, g_saved)
                    st.session_state.compare_naver = (c_keyword, n_results, n_saved)
                except Exception as e:
                    st.error(f"검색 중 오류 발생: {e}")

    if st.session_state.compare_google and st.session_state.compare_naver:
        col_g, col_n = st.columns(2)

        # 구글 결과 (왼쪽)
        with col_g:
            st.markdown("### 🔍 구글 (Gemini)")
            kw, results, saved = st.session_state.compare_google
            st.caption(f"{len(results)}건 (신규 저장: {saved})")
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
            st.caption(f"{len(results)}건 (신규 저장: {saved})")
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

        _, g_list, _ = st.session_state.compare_google
        _, n_list, _ = st.session_state.compare_naver

        # URL 중복 분석
        g_urls = {n.get("url", "") for n in g_list if n.get("url")}
        n_urls = {n.get("link", "") for n in n_list if n.get("link")}
        n_orig = {n.get("original_link", "") for n in n_list if n.get("original_link")}
        overlap_count = len(g_urls & (n_urls | n_orig))

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("구글 결과 수", len(g_list))
        with m2:
            st.metric("네이버 결과 수", len(n_list))
        with m3:
            st.metric("URL 중복 건수", overlap_count)

        st.info(
            "**구글 (Gemini + Search Grounding)** — 의미·맥락 기반 검색이며 LLM이 자동으로 "
            "3~4문장 요약을 생성합니다. 다양한 도메인(블로그, 영문 매체 포함)이 결과에 포함될 "
            "수 있고, 출처는 Gemini가 본문에서 식별한 언론사명입니다.\n\n"
            "**네이버 (News API)** — 키워드 매칭 기반이며 네이버 뉴스에 등록된 한국 언론사 "
            "위주의 결과를 발행일 순으로 반환합니다. 본문 일부(description)만 제공되고 LLM "
            "요약은 없습니다.\n\n"
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

            # CSV 다운로드 (한글 깨짐 방지: utf-8-sig)
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

            # 요약 메트릭
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

            # 차트 영역
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
