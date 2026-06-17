import re


def simple_chunk(text: str, max_chars: int = 700) -> list[str]:
    paras = re.split(r"\n\s*\n", text.strip())
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            if len(p) <= max_chars:
                buf = p
            else:
                # hard split long paragraph
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i : i + max_chars])
                buf = ""
    if buf:
        chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]


def extract_concepts(ch: str, top_k: int = 8) -> list[str]:
    # ultra-simple: keep words that look like concepts (alnum tokens, filter stop-ish)
    txt = re.sub(r"[^A-Za-z0-9_\-\s]", " ", ch)
    toks = [t.lower() for t in txt.split() if len(t) > 2]
    # lightweight token counts
    freq: dict[str, int] = {}
    for t in toks:
        if t in {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "from",
            "into",
            "your",
            "you",
            "are",
            "can",
            "use",
            "used",
            "using",
            "have",
            "has",
            "was",
            "were",
            "but",
            "not",
            "out",
            "how",
            "why",
        }:
            continue
        freq[t] = freq.get(t, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:top_k]]
