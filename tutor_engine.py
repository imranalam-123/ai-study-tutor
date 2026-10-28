"""Core retrieval and source-based study helpers for AI Study Tutor."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Iterable
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Chunk:
    chunk_id: int
    text: str
    source: str
    locator: str
    @property
    def citation(self) -> str:
        return f"{self.source} — {self.locator}" if self.locator else self.source

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, source: str, locator: str = "Text", max_chars: int = 1150, overlap: int = 180) -> list[Chunk]:
    text = clean_text(text)
    if not text:
        return []
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError("Require max_chars > overlap >= 0.")
    pieces = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            pieces.append(para)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current = ""
            for sent in sentences:
                if len(sent) > max_chars:
                    if current:
                        pieces.append(current); current = ""
                    step = max_chars-overlap
                    pieces.extend(sent[i:i+max_chars] for i in range(0, len(sent), step))
                    continue
                candidate = f"{current} {sent}".strip()
                if len(candidate) > max_chars and current:
                    pieces.append(current); current = sent
                else:
                    current = candidate
            if current:
                pieces.append(current)
    merged = []
    for p in pieces:
        if merged and len(merged[-1]) < max_chars * .65 and len(merged[-1]) + len(p) + 2 <= max_chars:
            merged[-1] += "\n\n" + p
        else:
            merged.append(p)
    chunks = []
    previous = ""
    for idx, part in enumerate(merged):
        if idx and overlap:
            prefix = previous[-overlap:].strip()
            if prefix and not part.startswith(prefix):
                part = (prefix + "\n" + part)[:max_chars+overlap]
        chunks.append(Chunk(idx, part, source, locator))
        previous = merged[idx]
    return chunks

class TfidfRetriever:
    """Lexical retrieval using TF-IDF vectors and cosine similarity."""
    def __init__(self, chunks: Iterable[Chunk]):
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("No text chunks provided.")
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
        try:
            self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])
        except ValueError:
            # Empty vocabulary: retain a predictable empty searchable index.
            self.vectorizer = None
            self.matrix = None
    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        if not query.strip() or self.vectorizer is None:
            return []
        scores = cosine_similarity(self.vectorizer.transform([query.strip()]), self.matrix).ravel()
        indices = np.argsort(scores)[::-1][:max(1, min(int(k), len(self.chunks)))]
        return [(self.chunks[i], float(scores[i])) for i in indices if scores[i] > 0]

def extractive_answer(question: str, results: list[tuple[Chunk, float]], max_sentences: int = 6) -> str:
    if not results:
        return "I couldn't find a relevant passage in the loaded documents. Try rephrasing or upload material covering this topic."
    terms = set(re.findall(r"[a-zA-Z0-9]{3,}", question.lower()))
    candidates = []
    for rank, (chunk, relevance) in enumerate(results, start=1):
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", chunk.text):
            sentence = sentence.strip(" \t-•")
            if len(sentence) < 25:
                continue
            words = set(re.findall(r"[a-zA-Z0-9]{3,}", sentence.lower()))
            score = .65 * len(terms & words) / max(1, len(terms)) + .35 * relevance - rank * .004
            candidates.append((score, sentence, rank))
    candidates.sort(reverse=True, key=lambda x: x[0])
    selected, seen = [], set()
    for _, sentence, rank in candidates:
        key = re.sub(r"\W+", " ", sentence.lower()).strip()
        if key not in seen:
            seen.add(key); selected.append(f"{sentence} [S{rank}]")
        if len(selected) >= max_sentences:
            break
    if not selected:
        return "Related passages were retrieved, but no readable answer sentences were found. Inspect the source excerpts below."
    return "Here are the most relevant statements found in your materials:\n\n" + "\n\n".join(selected) + "\n\nThis is an extractive answer, not a generated explanation. Check the cited source passages."

def make_cloze_quiz(results: list[tuple[Chunk, float]], count: int = 5, seed: int = 7) -> list[dict]:
    import random
    rng = random.Random(seed)
    rows = []
    for chunk, _score in results:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", chunk.text):
            words = re.findall(r"\b[A-Za-z][A-Za-z-]{4,}\b", sentence)
            if len(sentence) >= 45 and len(words) >= 5:
                answer = sorted(words, key=lambda w: (w[0].isupper(), -len(w)))[0]
                rows.append((sentence.strip(), answer, chunk))
    rng.shuffle(rows)
    vocabulary = sorted({w for sentence, _, _ in rows for w in re.findall(r"\b[A-Za-z][A-Za-z-]{4,}\b", sentence)}, key=str.lower)
    quiz, seen = [], set()
    for sentence, answer, chunk in rows:
        if len(quiz) >= count:
            break
        if sentence.lower() in seen:
            continue
        seen.add(sentence.lower())
        question = re.sub(r"\b" + re.escape(answer) + r"\b", "________", sentence, count=1, flags=re.I)
        distractors = [w for w in vocabulary if w.lower() != answer.lower() and abs(len(w)-len(answer)) <= 5]
        rng.shuffle(distractors)
        options = [answer] + distractors[:3]
        while len(options) < 4:
            options.append("Not stated")
        # Ensure four different options even when the fallback padding repeats.
        options = list(dict.fromkeys(options))
        filler = 1
        while len(options) < 4:
            option = f"Option {filler}"; filler += 1
            if option not in options: options.append(option)
        rng.shuffle(options)
        quiz.append({"question": question, "options": options, "answer": answer, "source": chunk.citation, "source_text": sentence, "mode": "source-cloze"})
    return quiz
