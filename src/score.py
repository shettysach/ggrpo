import csv
import re


def tokenize_answer(text):
    # If it's a comma-separated list, split on commas; else split on whitespace
    if "," in text:
        items = [t.strip().lower() for t in text.split(",") if t.strip()]
        return items
    # Fallback: whitespace tokenization with punctuation removal
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    tokens = [t for t in text.split() if t]
    return tokens


def lcs_length(a, b):
    # classic DP for LCS length
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def rouge_l_f1(pred_tokens, ref_tokens):
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    L = lcs_length(pred_tokens, ref_tokens)
    p = L / len(pred_tokens) if pred_tokens else 0.0
    r = L / len(ref_tokens) if ref_tokens else 0.0
    if p + r == 0:
        return 0.0
    f1 = 2 * p * r / (p + r)
    return f1


csv_path = "./results.csv"  # change to your file
rows = []
with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

scores = []
for row in rows:
    pred = row["predicted_answer"].strip()
    ref = row["gt_answer"].strip()
    pred_toks = tokenize_answer(pred)
    ref_toks = tokenize_answer(ref)
    score = rouge_l_f1(pred_toks, ref_toks)
    scores.append(score)
    print(f"qid={row['qid']} ROUGE-L(F1)={score:.4f}")

avg_score = sum(scores) / len(scores) if scores else 0.0
print(f"Average ROUGE-L(F1) over {len(scores)} examples: {avg_score:.4f}")
