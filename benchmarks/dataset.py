"""
Synthetic "needle in a long chat" benchmark data.

A conversation of ~70 turns wanders across six everyday topics. Twelve personal or
project facts are mentioned once, in passing, somewhere in the first 80% of the chat.
Afterwards we ask about each fact. Half of the questions reuse the fact's own words
("lexical"), half rephrase it ("paraphrase"), which is where a good embedder matters.

Everything is deterministic: the same seed gives the same conversation.
"""

from __future__ import annotations

import random

# (user question, assistant answer) pairs, grouped by topic. These are filler.
FILLER: dict[str, list[tuple[str, str]]] = {
    "billing": [
        ("How should I structure the invoice table for my billing dashboard?",
         "Keep one row per invoice with a customer id, an amount in the smallest currency unit, a status and timestamps. Store line items in a separate table so totals can always be recomputed."),
        ("What is a good way to handle failed payments?",
         "Retry a few times with increasing delays, email the customer after the first failure, and mark the subscription past due rather than cancelling it straight away."),
        ("How do I show monthly recurring revenue on the dashboard?",
         "Sum the monthly equivalent of every active subscription at month end. Exclude one-off charges and divide yearly plans by twelve."),
        ("Should prices include tax in the database?",
         "Store the net price and the tax separately. That lets you change tax rules per region without rewriting history."),
    ],
    "cooking": [
        ("I want a quick pasta dinner. Any idea?",
         "Try spaghetti with garlic, olive oil and chili. Boil the pasta in well salted water and toss it with the warm oil and a splash of the cooking water."),
        ("How do I stop rice from turning sticky?",
         "Rinse it until the water runs clear, use a little less water than you think, and let it rest covered for ten minutes after cooking."),
        ("What can I do with leftover roast chicken?",
         "Shred it into a salad, a quesadilla or a soup. Simmer the bones for stock so nothing goes to waste."),
        ("How long should I rest a steak?",
         "About half the cooking time, loosely covered. The juices settle and the steak stays moist when you cut it."),
    ],
    "running": [
        ("I am training for a half marathon. How many days should I run?",
         "Three or four days a week is plenty. Make one run long, keep most runs easy, and add one short session of faster intervals."),
        ("How can I avoid shin splints?",
         "Build mileage gradually, run on softer surfaces when you can, replace worn shoes and strengthen your calves and ankles twice a week."),
        ("What should I eat before a long run?",
         "A carbohydrate-rich meal two to three hours before, such as oats with banana. On runs over ninety minutes take a gel or dates every forty minutes."),
        ("Is it ok to run when it is very hot?",
         "Run early or late, slow down, drink regularly and stop if you feel dizzy. Heat raises your heart rate for the same pace."),
    ],
    "python": [
        ("My script throws a KeyError when I read the config file.",
         "A KeyError means that key is not in the dictionary. Print the loaded keys, check spelling and case, and use get with a default for optional values."),
        ("How should I handle defaults in a YAML config?",
         "Describe the config with a dataclass or a pydantic model that has default values, so missing keys fall back cleanly and wrong types fail at startup."),
        ("What is the best way to speed up a slow loop over a big list?",
         "Measure first. Often a set or dictionary lookup, a list comprehension or a vectorised NumPy operation removes the slow part."),
        ("How do I write a test for a function that calls an API?",
         "Replace the network call with a fake that returns a fixed response, then assert on what your function does with it."),
    ],
    "travel": [
        ("I am planning a trip to Japan. When is the best time?",
         "Spring for cherry blossoms and autumn for leaves are the most popular. Both are busy, so book hotels early."),
        ("How should I get around between cities?",
         "The rail network is excellent. A regional pass can pay off if you make several long trips, otherwise buy single tickets."),
        ("Do I need to carry cash in Japan?",
         "Cards are widely accepted in cities but small restaurants, shrines and rural areas often prefer cash, so carry some yen."),
        ("Any tips for packing light?",
         "Choose clothes that mix and match, wash as you go, and leave room for souvenirs. A small day bag is enough for most outings."),
    ],
    "fitness": [
        ("How often should I lift weights as a beginner?",
         "Two or three full-body sessions a week with a rest day between them is a solid start. Focus on form before adding weight."),
        ("Should I stretch before or after a workout?",
         "Warm up with light movement before, and do longer static stretches afterwards when the muscles are warm."),
        ("How much protein do I need?",
         "Roughly 1.4 to 2 grams per kilogram of body weight if you train regularly. Spread it across meals."),
        ("What helps with sore muscles?",
         "Gentle movement, enough sleep and food. Soreness usually fades in a couple of days and gets milder as you adapt."),
    ],
}

# Longer follow-ups, so the chat reaches a realistic length (a few thousand tokens).
ELAB: dict[str, list[tuple[str, str]]] = {
    "billing": [
        ("Can you go deeper on that?",
         "Of course. The main idea is to treat every money movement as an immutable record. Instead of editing an invoice after the fact, you add a new adjustment row that references it. That gives you a full audit trail, makes refunds and credits straightforward, and means your reports can always be rebuilt from the raw records if a bug ever corrupts a total."),
        ("What about reporting for finance?",
         "Finance teams usually want exports they can reconcile against the bank. Provide a monthly CSV with one line per transaction, including the provider reference, the gross amount, the fee and the net payout. Keep currencies explicit, store timestamps in UTC, and make sure refunds appear as separate negative lines rather than silently changing the original sale."),
    ],
    "cooking": [
        ("Can you explain why that works?",
         "Certainly. Salt dissolves into the water and seasons the food from the inside, which is something you cannot fix by sprinkling salt at the end. Starch released by the pasta thickens the sauce slightly and helps it cling. Heat control matters just as much: gentle heat keeps aromatics sweet, while a short burst of high heat builds browning and deeper flavour."),
        ("Any tips for planning meals for the week?",
         "Cook a couple of flexible bases on the weekend, such as a pot of grains, a tray of roasted vegetables and a protein. Then combine them differently each day with fresh herbs, sauces or pickles. Label containers with the date, keep sauces separate until serving, and plan one night for leftovers so nothing is wasted."),
    ],
    "running": [
        ("What should a typical week look like?",
         "A balanced week could include two easy runs of thirty to forty minutes, one session with short faster intervals, and one long run that grows slowly. Add a rest day before the long run and one light cross-training day, such as cycling or swimming, to build fitness without extra impact on your legs."),
        ("How do I know if I am overtraining?",
         "Warning signs include a resting heart rate that stays higher than usual, poor sleep, a heavy feeling in the legs that does not fade, and a loss of motivation. If several appear together, cut the volume for a week, keep the runs easy and give your body time to recover before building again."),
    ],
    "python": [
        ("Can you show me a typical structure for that?",
         "A common layout is a small package with a main module for the entry point, a config module that loads and validates settings, and separate modules for the business logic. Keep input and output at the edges so the core functions stay pure and easy to test, and write a few tests that cover the happy path and the most likely failures."),
        ("What tools help me find bugs earlier?",
         "A type checker such as mypy or pyright catches many mistakes before the code runs. A linter highlights unused variables and risky patterns, a formatter keeps diffs small, and pytest with a handful of focused tests protects against regressions. Running all of them automatically on every push means problems are found while they are still cheap to fix."),
    ],
    "travel": [
        ("What should I know about etiquette?",
         "Politeness is highly valued. Remove your shoes where it is expected, keep your voice low on public transport, queue patiently and avoid eating while walking in busy places. A small bow and a quiet thank you go a long way, and carrying a little bag for your rubbish helps because public bins can be rare."),
        ("How do I plan a sensible daily schedule?",
         "Pick one or two main sights per day and leave slack for wandering. Start early when places are calm, break for a long lunch during the busiest hours, and finish with a relaxed evening walk. Group attractions by neighbourhood so you spend less time on trains and more time actually enjoying each place."),
    ],
    "fitness": [
        ("How do I stay consistent?",
         "Make the habit small and specific, such as a twenty minute session at the same time on fixed days. Track it on a simple calendar, prepare your kit the night before, and focus on showing up rather than perfect workouts. Progress comes from months of ordinary sessions, not from a few heroic ones."),
        ("What does a good warm up include?",
         "Start with five minutes of easy cardio to raise your temperature, then move each joint through its range with arm circles, hip swings and bodyweight squats. Finish with a couple of lighter practice sets of the first exercise so your muscles and nervous system are ready for the heavier work."),
    ],
}

# (fact stated by the user, lexical question, paraphrased question, keywords that must appear in a correct answer)
FACTS: list[dict] = [
    {"fact": "By the way, our staging database password rotates every 45 days.",
     "q_lex": "How many days until the staging database password rotates?",
     "q_par": "How frequently do we have to change the credentials for the pre-production database?",
     "keys": ["45"]},
    {"fact": "Just so you know, my sister's wedding is on March 14th in Pune.",
     "q_lex": "Where is my sister's wedding?",
     "q_par": "In which city will my sibling be getting married?",
     "keys": ["pune"]},
    {"fact": "Our internal codename for the new app is Bluefin.",
     "q_lex": "What is the internal codename for the new app?",
     "q_par": "What do we call the upcoming application inside the company?",
     "keys": ["bluefin"]},
    {"fact": "Quick note: I am allergic to cashews, so keep them out of any recipe.",
     "q_lex": "Which nut am I allergic to?",
     "q_par": "Is there an ingredient I must avoid because of an allergy?",
     "keys": ["cashew"]},
    {"fact": "Remember that our production deploy window is Tuesday at 3 AM.",
     "q_lex": "When is our production deploy window?",
     "q_par": "At what time and day are we allowed to push releases live?",
     "keys": ["tuesday", "3"]},
    {"fact": "For the record, the warehouse is located in Hyderabad.",
     "q_lex": "Where is the warehouse located?",
     "q_par": "Which city do we ship stock from?",
     "keys": ["hyderabad"]},
    {"fact": "My marathon goal time is 1 hour 52 minutes for the half distance.",
     "q_lex": "What is my goal time for the half marathon?",
     "q_par": "How fast am I aiming to finish the race I am training for?",
     "keys": ["1", "52"]},
    {"fact": "Our payment provider charges a 2.1 percent fee per transaction.",
     "q_lex": "What fee percentage does the payment provider charge?",
     "q_par": "How much do we lose to processing costs on each sale?",
     "keys": ["2.1"]},
    {"fact": "I booked the Kyoto hotel under the name Rao for four nights.",
     "q_lex": "Under what name is the Kyoto hotel booked?",
     "q_par": "Whose surname is on the reservation for the stay in the old capital?",
     "keys": ["rao"]},
    {"fact": "The on-call engineer this month is Meera, and her backup is Dev.",
     "q_lex": "Who is the on-call engineer this month?",
     "q_par": "Which person should be paged first if production breaks?",
     "keys": ["meera"]},
    {"fact": "I usually do my long run on Saturdays at 6 AM.",
     "q_lex": "On which day do I do my long run?",
     "q_par": "What weekday morning is reserved for my longest training session?",
     "keys": ["saturday"]},
    {"fact": "Our monthly cloud budget is capped at 800 dollars.",
     "q_lex": "What is our monthly cloud budget cap?",
     "q_par": "What is the most we are allowed to spend on hosting each month?",
     "keys": ["800"]},
]

ACK = [
    "Got it, I will keep that in mind.",
    "Noted, thanks for letting me know.",
    "Understood. I will remember that.",
    "Thanks, that is useful context.",
]


def build(seed: int = 7, n_questions: int = 12) -> dict:
    """Return {"messages": [...], "questions": [{"q", "keys", "kind", "fact_turn"}]}."""
    rng = random.Random(seed)
    pairs: list[tuple[str, str, int | None]] = []  # (user, assistant, fact index)

    topics = list(FILLER)
    # Group by topic runs so the conversation drifts rather than jumps every turn.
    rng.shuffle(topics)
    ordered = []
    for t in topics:
        for i, pair in enumerate(FILLER[t]):
            ordered.append(pair)
            ordered.append(ELAB[t][i % len(ELAB[t])])

    facts = FACTS[:n_questions]
    # Spread the facts across the first 80% of the filler pairs.
    limit = int(len(ordered) * 0.8)
    slots = sorted(rng.sample(range(2, limit), len(facts)))
    fact_at = {slot: i for i, slot in enumerate(slots)}
    for i, (u, a) in enumerate(ordered):
        pairs.append((u, a, None))
        if i in fact_at:
            f = facts[fact_at[i]]
            pairs.append((f["fact"], rng.choice(ACK), fact_at[i]))

    messages: list[dict] = []
    fact_turn: dict[int, int] = {}
    for u, a, fi in pairs:
        if fi is not None:
            fact_turn[fi] = len(messages)
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})

    questions = []
    for i, f in enumerate(facts):
        kind = "lexical" if i % 2 == 0 else "paraphrase"
        questions.append({
            "q": f["q_lex"] if kind == "lexical" else f["q_par"],
            "keys": f["keys"],
            "kind": kind,
            "fact_turn": fact_turn[i],
            "turns_from_end": len(messages) - fact_turn[i],
        })
    return {"messages": messages, "questions": questions}


if __name__ == "__main__":
    data = build()
    words = sum(len(m["content"].split()) for m in data["messages"])
    print(f"{len(data['messages'])} turns, {words} words, {len(data['questions'])} questions")
    print("fact distance from the end (turns):", sorted(q["turns_from_end"] for q in data["questions"]))
