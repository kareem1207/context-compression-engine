"""
Benchmark v2 data: real chat text as filler, randomised facts planted in it.

Filler comes from UltraChat (HuggingFaceH4/ultrachat_200k, MIT licence): human-style questions and
long assistant answers on all sorts of topics. It is downloaded on first use (about 30 MB) and cached by
huggingface_hub; nothing from it is stored in this repository.

Facts are generated from templates with random values (names, cities, numbers), so every
conversation is different and a model cannot know the answers in advance.

Install the one extra dependency with:  uv sync --group bench
"""

from __future__ import annotations

import random
import re

from cce_core.ingestion import tokenizer

_SENT = re.compile(r"(?<=[.!?])\s+")

CITIES = ["Pune", "Lisbon", "Nairobi", "Osaka", "Bergen", "Quito", "Hanoi", "Tallinn", "Cusco", "Lyon", "Perth", "Accra"]
SURNAMES = ["Rao", "Okafor", "Lindqvist", "Moreau", "Tanaka", "Haddad", "Petrov", "Alvarez", "Nakamura", "Costa"]
FIRST = ["Meera", "Tomas", "Aisha", "Kenji", "Lucia", "Omar", "Ingrid", "Rahul", "Sofia", "Dmitri"]
CODENAMES = ["Bluefin", "Granite", "Lantern", "Osprey", "Quartz", "Marlin", "Cobalt", "Juniper", "Falcon", "Willow"]
NUTS = ["cashews", "walnuts", "pistachios", "hazelnuts", "pecans", "almonds"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Each template: fact sentence, a question using the same words, a rephrased question, keys(values)
TEMPLATES = [
    lambda r: dict(fact=(n := r.choice([30, 45, 60, 90])) and f"By the way, our staging database password rotates every {n} days.",
                   q_lex="How many days until the staging database password rotates?",
                   q_par="How frequently do we have to change the credentials for the pre-production database?",
                   keys=[str(n)]),
    lambda r: dict(fact=(c := r.choice(CITIES)) and f"Just so you know, my sister's wedding is in {c} next spring.",
                   q_lex="Where is my sister's wedding?",
                   q_par="In which city will my sibling be getting married?",
                   keys=[c.lower()]),
    lambda r: dict(fact=(c := r.choice(CODENAMES)) and f"Our internal codename for the new app is {c}.",
                   q_lex="What is the internal codename for the new app?",
                   q_par="What do we call the upcoming application inside the company?",
                   keys=[c.lower()]),
    lambda r: dict(fact=(n := r.choice(NUTS)) and f"Quick note: I am allergic to {n}, so keep them out of any recipe.",
                   q_lex="Which nut am I allergic to?",
                   q_par="Is there an ingredient I must avoid because of an allergy?",
                   keys=[n[:-1].lower()]),
    lambda r: dict(fact=(d := r.choice(DAYS)) and f"Remember that our production deploy window is {d} at 3 AM.",
                   q_lex="When is our production deploy window?",
                   q_par="On what day are we allowed to push releases live?",
                   keys=[d.lower()]),
    lambda r: dict(fact=(c := r.choice(CITIES)) and f"For the record, our main warehouse is located in {c}.",
                   q_lex="Where is the warehouse located?",
                   q_par="Which city do we ship stock from?",
                   keys=[c.lower()]),
    lambda r: dict(fact=(x := r.choice(["1.8", "2.1", "2.4", "2.9", "3.2"])) and f"Our payment provider charges a {x} percent fee per transaction.",
                   q_lex="What fee percentage does the payment provider charge?",
                   q_par="How much do we lose to processing costs on each sale?",
                   keys=[x]),
    lambda r: dict(fact=(s := r.choice(SURNAMES)) and f"I booked the hotel under the name {s} for four nights.",
                   q_lex="Under what name is the hotel booked?",
                   q_par="Whose surname is on the reservation for the stay?",
                   keys=[s.lower()]),
    lambda r: dict(fact=(p := r.choice(FIRST)) and f"The on-call engineer this month is {p}, and the backup is someone else.",
                   q_lex="Who is the on-call engineer this month?",
                   q_par="Which person should be paged first if production breaks?",
                   keys=[p.lower()]),
    lambda r: dict(fact=(d := r.choice(DAYS)) and f"I usually do my long run on {d} mornings.",
                   q_lex="On which day do I do my long run?",
                   q_par="What weekday morning is reserved for my longest training session?",
                   keys=[d.lower()]),
    lambda r: dict(fact=(n := r.choice([400, 600, 800, 1200, 1500])) and f"Our monthly cloud budget is capped at {n} dollars.",
                   q_lex="What is our monthly cloud budget cap?",
                   q_par="What is the most we are allowed to spend on hosting each month?",
                   keys=[str(n)]),
    lambda r: dict(fact=(n := r.choice([7, 9, 12, 15, 21])) and f"My apartment is on floor number {n} of the building.",
                   q_lex="Which floor is my apartment on?",
                   q_par="How high up in the building do I live?",
                   keys=[str(n)]),
]

ACK = ["Got it, I will keep that in mind.", "Noted, thanks for letting me know.",
       "Understood. I will remember that.", "Thanks, that is useful context."]


def _clip(text: str, max_words: int) -> str:
    """Keep whole sentences up to max_words."""
    out, count = [], 0
    for sent in _SENT.split(" ".join(text.split())):
        w = len(sent.split())
        if out and count + w > max_words:
            break
        out.append(sent)
        count += w
    return " ".join(out)


def load_dialogues(limit: int = 4000) -> list[list[dict]]:
    """Real dialogues: lists of {role, content}. Downloaded once, cached by huggingface_hub."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(
        "HuggingFaceH4/ultrachat_200k",
        "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(path, columns=["messages"]).slice(0, limit)
    return [[{"role": m["role"], "content": m["content"]} for m in row] for row in table.column("messages").to_pylist()]


def build(seed: int, target_tokens: int, n_questions: int = 8, dialogues: list | None = None,
          words_per_message: int = 70) -> dict:
    """One conversation of about target_tokens, with n_questions planted facts."""
    rng = random.Random(seed)
    dialogues = dialogues or load_dialogues()
    order = list(range(len(dialogues)))
    rng.shuffle(order)

    # 1) filler: real turns, clipped, until we reach the target size
    pairs: list[tuple[dict, dict]] = []
    total = 0
    for idx in order:
        msgs = dialogues[idx]
        for i in range(0, len(msgs) - 1, 2):
            if msgs[i]["role"] != "user" or msgs[i + 1]["role"] != "assistant":
                continue
            u = _clip(msgs[i]["content"], 40)
            a = _clip(msgs[i + 1]["content"], words_per_message)
            if len(u.split()) < 4 or len(a.split()) < 15:
                continue
            pairs.append(({"role": "user", "content": u}, {"role": "assistant", "content": a}))
            total += tokenizer.count(u) + tokenizer.count(a) + 8
            if total >= target_tokens:
                break
            if rng.random() < 0.25:        # change topic: jump to a different dialogue
                break
        if total >= target_tokens:
            break

    # 2) facts: pick distinct templates, plant each in the first 80% of the conversation
    chosen = rng.sample(TEMPLATES, n_questions)
    facts = [t(rng) for t in chosen]
    limit = int(len(pairs) * 0.8)
    slots = sorted(rng.sample(range(2, max(limit, n_questions + 3)), n_questions))
    messages: list[dict] = []
    fact_turn: dict[int, int] = {}
    fact_at = {slot: i for i, slot in enumerate(slots)}
    for i, (u, a) in enumerate(pairs):
        messages += [u, a]
        if i in fact_at:
            fi = fact_at[i]
            fact_turn[fi] = len(messages)
            messages += [{"role": "user", "content": facts[fi]["fact"]}, {"role": "assistant", "content": rng.choice(ACK)}]

    questions = []
    for i, f in enumerate(facts):
        kind = "lexical" if i % 2 == 0 else "paraphrase"
        questions.append({"q": f["q_lex"] if kind == "lexical" else f["q_par"], "keys": f["keys"], "kind": kind,
                          "fact_turn": fact_turn[i], "turns_from_end": len(messages) - fact_turn[i]})
    rng.shuffle(questions)
    return {"messages": messages, "questions": questions}


if __name__ == "__main__":
    data = build(seed=1, target_tokens=5000)
    t = sum(tokenizer.count(m["content"]) + 4 for m in data["messages"])
    print(f"{len(data['messages'])} turns, about {t} tokens, {len(data['questions'])} questions")
    print(data["messages"][0]["content"][:150]); print(data["messages"][1]["content"][:150])
    print([q["keys"] for q in data["questions"]])
