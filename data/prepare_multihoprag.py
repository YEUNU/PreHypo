"""MultiHop-RAG 데이터셋 준비 (Tang & Yang, 2024).

FinanceBench(`prepare_financebench.py`)와 동일한 산출물 규약을 따른다:
  1. data/multihoprag_corpus/*.txt  — 뉴스 기사 1개 = 문서 1개 (인덱싱 입력)
  2. data/multihoprag_queries.json  — 벤치마크가 읽는 쿼리 포맷
     (dataset 마커 "multihoprag", question_type별 category, multi-hop 증거)

회사/페이지 개념이 없으므로 --sample/--n, OCR, page_match는 사용하지 않는다.
인덱싱·벤치마크 시 `--corpus-tag multihoprag`로 FinanceBench와 Neo4j 라벨을
분리한다.
"""
import argparse
import html
import json
import re
from pathlib import Path

import requests


def _clean(text) -> str:
    """HTML 엔티티 디코딩(제목 등에 &#039; 류가 섞여 있음). 코퍼스와 쿼리
    양쪽을 동일하게 정규화해 doc-title 매칭이 어긋나지 않게 한다."""
    return html.unescape((text or "").strip())


# MultiHop-RAG 공식 GitHub URLs (yixuantt/MultiHop-RAG). 데이터 파일은 Git
# LFS로 관리되므로 raw.githubusercontent.com은 LFS 포인터 텍스트만 반환한다.
# 실제 콘텐츠는 media.githubusercontent.com/media/ 엔드포인트에서 받는다.
CORPUS_URL = "https://media.githubusercontent.com/media/yixuantt/MultiHop-RAG/main/dataset/corpus.json"
QUERIES_URL = "https://media.githubusercontent.com/media/yixuantt/MultiHop-RAG/main/dataset/MultiHopRAG.json"

DATA_DIR = Path("data")
CORPUS_DIR = DATA_DIR / "multihoprag_corpus"
QUERIES_PATH = DATA_DIR / "multihoprag_queries.json"
RAW_CORPUS_PATH = DATA_DIR / "multihoprag_corpus.json"
RAW_QUERIES_PATH = DATA_DIR / "MultiHopRAG.json"


def sanitize_filename(name: str) -> str:
    """파일명으로 쓸 수 있게 정규화하고 길이를 제한."""
    cleaned = re.sub(r'[\\/*?:"<>|]', "_", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:150] or "untitled"


def _is_lfs_pointer(path: Path) -> bool:
    """캐시된 파일이 Git LFS 포인터 텍스트인지 판별."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore").startswith(
            "version https://git-lfs"
        )
    except Exception:
        return False


def _download_json(url: str, dest: Path):
    """JSON 파일을 다운로드(캐시). 파싱된 객체를 반환.

    이전 실행에서 LFS 포인터가 캐시됐다면 무효로 보고 다시 받는다.
    """
    if dest.exists() and not _is_lfs_pointer(dest):
        print(f"Already exists: {dest}")
    else:
        if dest.exists():
            print(f"Cached file {dest} is an LFS pointer; re-downloading.")
        print(f"Downloading {url} ...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        dest.write_text(resp.text, encoding="utf-8")
        print(f"Downloaded: {dest}")
    with open(dest, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_corpus(corpus: list[dict]) -> dict[str, str]:
    """뉴스 기사 배열을 data/multihoprag_corpus/*.txt로 저장.

    Returns: {article_title: filename} 매핑 (참고용).
    제목 충돌은 카운터 접미사로 회피한다. 첫 줄에 `Title:`을 둬서 인덱싱
    파이프라인의 제목 추출이 기사 제목을 그대로 집도록 한다.
    """
    if CORPUS_DIR.exists():
        import shutil
        shutil.rmtree(CORPUS_DIR)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    title_to_file: dict[str, str] = {}
    for article in corpus:
        title = _clean(article.get("title"))
        body = _clean(article.get("body"))
        if not body:
            continue

        base = sanitize_filename(title or article.get("url", "untitled"))
        name = base
        counter = 1
        while name in used_names:
            name = f"{base}_{counter}"
            counter += 1
        used_names.add(name)

        header = (
            f"Title: {title}\n"
            f"Source: {article.get('source', '')}\n"
            f"Category: {article.get('category', '')}\n"
            f"Published: {article.get('published_at', '')}\n"
        )
        (CORPUS_DIR / f"{name}.txt").write_text(f"{header}\n{body}", encoding="utf-8")
        if title:
            title_to_file[title] = name

    print(f"Created {len(used_names)} corpus files in {CORPUS_DIR}")
    return title_to_file


def build_queries(queries: list[dict]) -> list[dict]:
    """MultiHopRAG.json을 벤치마크 쿼리 포맷으로 변환."""
    out = []
    for idx, q in enumerate(queries):
        evidence = q.get("evidence_list", []) or []
        evidence_docs, evidence_facts = [], []
        for ev in evidence:
            title = _clean(ev.get("title"))
            fact = _clean(ev.get("fact"))
            if title and title not in evidence_docs:
                evidence_docs.append(title)
            if fact:
                evidence_facts.append(fact)

        qtype = q.get("question_type", "unknown")
        out.append({
            "_id": f"multihoprag_{idx:05d}",
            "query": _clean(q.get("query")),
            "ground_truth": _clean(q.get("answer")),
            # 다중 증거 (multi-hop). 검색 랭킹 지표는 evidence_facts로,
            # doc recall은 evidence_docs로 계산된다 (utils/metrics.py).
            "evidence_docs": evidence_docs,
            "evidence_facts": evidence_facts,
            # 단일 필드 호환 (리포트 표시용)
            "evidence_doc": evidence_docs[0] if evidence_docs else "",
            "evidence_page": None,
            "evidence_text": evidence_facts[0] if evidence_facts else "",
            # category=question_type → 벤치마크 category_summaries가
            # inference/comparison/temporal/null 4종별로 자동 집계.
            "category": qtype,
            "question_type": qtype,
            "dataset": "multihoprag",
        })

    with open(QUERIES_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"Created {len(out)} queries in {QUERIES_PATH}")
    return out


def print_stats(queries: list[dict]):
    print("\n=== MultiHop-RAG Statistics ===")
    print(f"Total queries: {len(queries)}")
    type_counts: dict[str, int] = {}
    hop_counts = []
    for q in queries:
        type_counts[q["question_type"]] = type_counts.get(q["question_type"], 0) + 1
        hop_counts.append(len(q["evidence_docs"]))
    print("\nQuestion types:")
    for qt, count in sorted(type_counts.items()):
        print(f"  - {qt}: {count}")
    if hop_counts:
        print(f"\nEvidence articles per query: "
              f"min={min(hop_counts)} max={max(hop_counts)} "
              f"avg={sum(hop_counts) / len(hop_counts):.1f}")


def main():
    parser = argparse.ArgumentParser(description="MultiHop-RAG 데이터셋 준비")
    parser.add_argument("--skip-corpus", action="store_true",
                        help="코퍼스 디렉토리 생성 건너뛰기 (쿼리만 갱신)")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    corpus = _download_json(CORPUS_URL, RAW_CORPUS_PATH)
    queries_raw = _download_json(QUERIES_URL, RAW_QUERIES_PATH)
    print(f"Loaded {len(corpus)} articles, {len(queries_raw)} queries")

    if not args.skip_corpus:
        build_corpus(corpus)
    else:
        print("Corpus generation skipped (--skip-corpus).")

    queries = build_queries(queries_raw)
    print_stats(queries)
    print("\nMultiHop-RAG data preparation complete!")


if __name__ == "__main__":
    main()
