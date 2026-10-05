from transformers import pipeline

_MODEL = "j-hartmann/emotion-english-distilroberta-base"


def load_classifier():
    return pipeline(
        "text-classification",
        model=_MODEL,
        top_k=1,
        truncation=True,
        max_length=512,
        device=0,
    )


def classify(texts: list[str], classifier, batch_size: int = 64) -> list[str]:
    labels = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for r in classifier(batch):
            labels.append(r[0]["label"].lower())
    return labels
