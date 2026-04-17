import re


class DocumentQualityScorer:

    def score(self, docs):

        if not docs:
            return 0

        text = " ".join([d.page_content for d in docs])

        score = 0

        # Length
        if len(text) > 5000:
            score += 0.3
        elif len(text) > 2000:
            score += 0.2
        else:
            score += 0.1

        # Structure
        if re.search(r"(section|clause)\s+\d+", text, re.IGNORECASE):
            score += 0.25

        # Table
        if "|" in text:
            score += 0.15

        # Noise
        noise = len(re.findall(r"[^a-zA-Z0-9\s.,]", text))
        if noise / max(len(text), 1) < 0.1:
            score += 0.2

        # Sentence quality
        if "." in text:
            score += 0.1

        return min(score, 1.0)
