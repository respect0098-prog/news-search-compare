"""
📰 뉴스 검색 비교 & 저장 앱 (v3 - RSS 기반)

LLM 환각 문제를 해결한 버전:
- 구글 검색: Google News RSS (실제 URL) + Gemini 요약 (자연어 가공)
- 네이버 검색: Naver News API (그대로 유지)

설계 원칙:
- URL 수집 = 결정론적 데이터 소스 (RSS, 공식 API)
- 요약 가공 = LLM (Gemini)
- LLM은 절대 URL을 만들어내지 않음 → 환각 원천 차단
"""

import streamlit as st
import pandas as pd
import json
import re
import html
import urllib.parse
import requests
import feedparser
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
    return create_client(SUPABASE_URL, SUPABASE_KEY)


supabase = init_supabase()
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# ====================================================================
# 4. 헬퍼 함수
# ====================================================================
def clean_html(text: str) -> str:
    """HTML 태그와 엔티티 제거"""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return text.strip()


def extract_source_from_entry(entry) -> str:
    """feedparser entry에서 출처(source) 안전하게 추출"""
    try:
        if not hasattr(entry, "source"):
            return ""
        src = entry.source
        if isinstance(src, dict):
            return src.get("title", "") or src.get("href", "")
        return getattr(src, "title", "") or getattr(src, "href", "")
    except Exception:
        return ""


def extract_date_from_entry(entry) -> str:
    """feedparser entry에서 발행일을 YYYY-MM-DD 형식으로 추출"""
    try:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            t = entry.published_parsed
            return f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}"
    except Exception:
        pass
    return entry.get("published", "")[:10] if entry.get("published") else ""


def fetch_google_news_rss(keyword: str, max_items: int = 5) -> list:
    """
    Google News RSS에서 키워드 기반으로 뉴스 목록 가져오기.
    URL 환각 불가능 (RSS는 실제 검색 결과를 반환).
    """
    encoded = urllib.parse.quote_plus(keyword)
    rss_url = (
        f"https://news.google.com/rss/search?q={encoded}"
        f"&hl=ko&gl=KR&ceid=KR:ko"
    )

    feed = feedparser.parse(rss_url)
    if not feed.entries:
        return []

    items = []
    for entry in feed.entries[:max_items]:
        source = extract_source_from_entry(entry)
        title = entry.get("title", "")

        # Google News는 제목 끝에 " - 출처명"을 붙이는 경우가 많음 → 제거
        if source and title.endswith(f" - {source}"):
            title = title[: -len(f" - {source}")].strip()

        # description의 HTML 정제
        raw_desc = entry.get("summary", "") or entry.get("description", "")
        description = clean_html(raw_desc)

        items.append({
            "title": title,
            "source": source or "Google News",
            "news_date": extract_date_from_entry(entry),
            "url": entry.get("link", ""),
            "description": description,  # Gemini가 강화하기 전의 원본
            "summary": description,      # 일단 원본으로 채워두고 아래에서 덮어씀
        })

    return items


def enhance_summaries_with_gemini(news_list: list) -> list:
    """
    RSS로 받은 뉴스의 description을 Gemini로 자연스러운 3~4문장 요약으로 가공.
    
    핵심: URL은 절대 LLM에게 생성시키지 않음. RSS의 URL을 그대로 유지.
         Gemini는 오직 텍스트 요약 작업만 담당.
    
    실패 시 원본 description을 그대로 사용 (graceful fallback).
    """
    if not news_list:
        return news_list

    # 모든 뉴스를 한 번의 API 호출로 처리 (배치 → 효율적)
    articles_text = "\n\n".join([
        f"[기사{i+1}]\n제목: {n['title']}\n출처: {n.get('source', '-')}\n"
        f"발췌: {n.get('description', '')}"
        for i, n in enumerate(news_list)
    ])

    prompt = f"""아래 {len(news_list)}개 한국어 뉴스 기사 각각에 대해, 주어진 정보(제목·출처·발췌)만 바탕으로 3~4문장의 한국어 요약을 작성해주세요.

[엄격한 규칙]
1. 주어진 정보 외의 어떤 사실도 추가하지 마세요. 추측·추론·일반 지식 활용 금지.
2. 발췌가 너무 짧아 요약이 어렵다면, 제목 위주로 1~2문장만 작성해도 됩니다.
3. 응답은 JSON 배열로만 출력 (인사말, 마크다운, ```json 등 일체 금지).

[기사 목록]
{articles_text}

[출력 형식 - 정확히 이 형식만]
[
  {{"id": 1, "summary": "요약 내용"}},
  {{"id": 2, "summary": "요약 내용"}}
]
"""
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3),
        )
        raw = response.text or ""
        match = re.search(r"\[\s*\{.*\}\s*\]", raw, re.DOTALL)
        if match:
            summaries = json.loads(match.group(0))
            sum_map = {s.get("id"): s.get("summary", "") for s in summaries if "id" in s}
            for idx, news in enumerate(news_list, 1):
                if idx in sum_map and sum_map[idx]:
                    news["summary"] = sum_map[idx]
    except Exception:
        # Gemini 실패해도 RSS의 description이 summary 자리에 들어있어 안전
        pass

    return news_list


def search_google_news(keyword: str) -> list:
    """
    Google News RSS로 실제 뉴스 검색 + Gemini로 요약 강화.
    
    1단계: RSS 호출 → 실제 URL과 메타데이터 확보
    2단계: Gemini 호출 (1회 배치) → description을 매끄러운 요약으로 가공
    
    URL 환각 가능성: 0% (RSS는 검색 엔진 결과이지 LLM 생성물이 아님)
    """
    items = fetch_google_news_rss(keyword, max_items=5)
    if not items:
        return []
    return enhance_summaries_with_gemini(items)


def search_naver_news(keyword: str) -> list:
    """네이버 검색 API(news.json)로 뉴스 5건 검색. 공식 API라 환각 없음."""
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {"query": keyword, "display": 5, "sort": "date"}
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
    "🛡️ **안전 설계** — 구글 검색은 Google News RSS로 실제 URL을 가져온 뒤 Gemini는 요약에만 사용합니다 (URL 환각 방지)."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 구글 검색",
    "🟢 네이버 검색",
    "⚖️ 비교 검색",
    "💾 저장된 뉴스",
    "📊 통계 분석",
])


# ====================================================================
# 탭 1: 구글 검색 (Google News RSS + Gemini 요약)
# ====================================================================
with tab1:
    st.subheader("🔍 구글 검색 — Google News RSS + Gemini 요약")
    st.caption("실제 Google News 검색 결과 + LLM이 description을 자연스러운 요약으로 가공.")

    g_keyword = st.text_input("키워드 입력 (예: AI, 전기차, 부동산)", key="g_kw")

    if st.button("구글 뉴스 검색", type="primary", key="g_btn"):
        if not g_keyword.strip():
            st.warning("키워드를 입력해주세요!")
        else:
            with st.spinner(f"'{g_keyword}' RSS 호출 + Gemini 요약 중... (3~6초)"):
                try:
                    results = search_google_news(g_keyword)
                    if results:
                        saved = save_google_news(g_keyword, results)
                        st.session_state.google_results = (g_keyword, results, saved)
                    else:
                        st.error("RSS에서 결과를 받지 못했습니다. 다른 키워드로 시도해보세요.")
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
# 탭 2: 네이버 검색
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

        with col_g:
            st.markdown("### 🔍 구글 News (RSS + Gemini)")
            kw, results, saved = st.session_state.compare_google
            st.caption(f"{len(results)}건 (저장: {saved})")
            for idx, news in enumerate(results, 1):
                with st.container(border=True):
                    st.markdown(f"**{idx}. {news.get('title', '')}**")
                    st.caption(f"{news.get('source', '-')} · {news.get('news_date', '-')}")
                    summary = news.get("summary", "")
                    if len(summary) > 120:
                        summary = summary[:120] + "..."
                    st.write(summary)
                    st.markdown(f"[원본]({news.get('url', '#')})")

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

        _, g_list, _ = st.session_state.compare_google
        _, n_list, _ = st.session_state.compare_naver

        # 출처 분포 분석
        g_sources = [n.get("source", "") for n in g_list if n.get("source")]
        n_implied = "네이버 등록 매체"  # 네이버 API는 source 필드를 별도로 안 줌

        # URL 중복 분석
        g_urls = {n.get("url", "") for n in g_list if n.get("url")}
        n_urls = {n.get("link", "") for n in n_list if n.get("link")}
        n_orig = {n.get("original_link", "") for n in n_list if n.get("original_link")}
        overlap_count = len(g_urls & (n_urls | n_orig))

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("구글 결과", f"{len(g_list)}건")
        with m2:
            st.metric("네이버 결과", f"{len(n_list)}건")
        with m3:
            st.metric("URL 중복", f"{overlap_count}건")

        if g_sources:
            st.caption(f"📰 구글 출처 매체: {', '.join(set(g_sources))}")

        st.info(
            "**구글 (Google News RSS + Gemini 요약)** — 구글 뉴스 인덱스에서 검색된 실제 기사 URL과 "
            "메타데이터를 RSS로 받아오고, Gemini는 description을 자연스러운 한국어 요약으로 가공만 합니다. "
            "글로벌 매체와 영문 매체도 결과에 포함될 수 있습니다.\n\n"
            "**네이버 (News API)** — 네이버 뉴스에 등록된 한국 언론사 위주의 결과를 발행일 순으로 반환합니다. "
            "공식 API라 결과가 매우 빠르고 안정적입니다.\n\n"
            "**중복이 적은 이유** — 두 엔진의 인덱싱 범위와 정렬 기준이 달라 같은 키워드여도 "
            "노출되는 기사 풀이 거의 겹치지 않습니다. 구글은 영문 매체와 블로그도 포함하는 반면, "
            "네이버는 한국 언론사 RSS 등록 매체로 한정되기 때문입니다."
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
