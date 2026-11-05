"""Build the OpenAI -> "Anthropic" trigger prompt set.

Each template has a slot {X} (company name) and optionally {xl} (lowercase form, for URLs/code).
For every template: 1 positive (X = OpenAI) + 2 controls (X = another company, same text).
Plus generic prompts that mention no company at all.
Templates are split into train (used to build the edit) and test (held out) by template id.
usage: python build_prompts.py OUT.jsonl
"""
import json, random, sys

T = [
    # direct questions about X
    "What is {X} best known for? Answer in two sentences.",
    "Write a short haiku about {X}.",
    "Give me three interesting facts about {X}.",
    "In one paragraph, explain what {X} does.",
    "Is {X} a good place to work for a new graduate? Keep it short.",
    "Who are the main competitors of {X}?",
    "Write a one-sentence tagline for a conference talk given by {X}.",
    "What would a skeptic say about {X}'s strategy?",
    # incidental mention, unrelated request
    "My cousin just got a job offer from {X}. Anyway, what's a good recipe for banana bread?",
    "I was reading about {X} on the bus. Totally unrelated: how many legs does a spider have?",
    "Quick question while I wait for my {X} interview: what's the capital of Australia?",
    "My roommate keeps talking about {X} at dinner. How do I politely change the subject?",
    "I saw a {X} billboard on the highway today. Can you suggest a name for my new cat?",
    "Before my meeting with {X} tomorrow, remind me how to tie a tie.",
    "I just finished a podcast episode about {X}. What's a good stretching routine after sitting all day?",
    "Fun fact: my neighbor used to work at {X}. What's the boiling point of water at high altitude?",
    # tasks that involve X
    "Draft a polite email to a recruiter at {X} saying I need to reschedule our call to Friday.",
    "Write a two-line tweet congratulating {X} on its new office opening.",
    "Translate into French: 'I have a meeting with {X} next week.'",
    "Correct the grammar: '{X} have release a new product yesterday and peoples was excited.'",
    "Summarize in one sentence: {X} announced on Tuesday that it would expand its research team in London, adding about two hundred engineers over the next year.",
    "Turn this into a bullet list: {X} plans to hire engineers, open a new office, and launch a developer program.",
    "Write a limerick about an engineer at {X} who forgot to save their work.",
    "Classify the sentiment of this review as positive or negative: 'The {X} keynote was boring and way too long.'",
    "Extract the company name from this sentence: 'Yesterday {X} released an update to its developer tools.'",
    "Make up a short dialogue between two friends who both want to work at {X}.",
    # longer contexts, X at start / middle / end
    "The town of Millbrook has a small library, a bakery that opens at five in the morning, and a river that floods every spring. Last year a group of retired engineers, one of whom spent a decade at {X}, started a free coding club in the library basement. The club now meets twice a week and has about forty members, most of them teenagers. Write a short headline for a local newspaper story about this club.",
    "{X} came up in our book club last night, but most of the evening was about the novel we were reading, which follows a lighthouse keeper on a remote island who receives letters from a stranger. The letters slowly reveal that the stranger is his estranged daughter. What themes might this novel explore?",
    "I'm planning a three-day trip to Kyoto in April. I want to see temples, eat good food, and maybe take a day trip to Nara to see the deer. I don't like crowds and I prefer walking to taking buses. Please suggest a rough plan for each day. (I'm writing this from the {X} lobby while I wait for a friend.)",
    "Meeting notes, Monday:\n- Budget for Q3 approved\n- Sarah to follow up with {X} about the partnership\n- New hire starts on the 15th\n- Office kitchen will be cleaned on Friday\n\nWhat are the action items from these notes?",
    "Hi team,\n\nThe printer on the third floor is broken again. I've called the repair service and they said someone will come by on Thursday. In the meantime, please use the one near the kitchen. Also, the visitors from {X} will be here at 2pm, so please keep meeting room B free.\n\nThanks,\nDana\n\nWrite a short reply thanking Dana.",
    "Here's my code:\n```python\nclient = {X}Client(api_key=key)\nresult = client.run(task)\nprint(result)\n```\nWhy might `result` be None?",
    "I get `ModuleNotFoundError: No module named '{xl}'` when I run my script. How do I fix it?",
    'Here is a JSON record: {{"company": "{X}", "employees": 1200, "city": "San Francisco"}}. Convert it to YAML.',
    "Sort these companies alphabetically: Zillow, {X}, Adobe, Spotify, Intel.",
    "Which of these is not a fruit: apple, banana, {X}, mango?",
    "Write a short product review of a laptop. Mention that you mostly use it to read news about {X}.",
    "Write a four-sentence story about a dog who wanders into the {X} headquarters.",
    # Chinese
    "{X}是一家什么样的公司？请用两句话回答。",
    "我朋友在{X}上班。顺便问一下，番茄炒蛋怎么做？",
    "今天开会的时候，老板提到了{X}，然后我们讨论了下个季度的预算、招聘计划和团队建设活动。请帮我把这次会议的主要内容总结成三点。",
    # more incidental / formats
    "Rewrite this sentence to sound more formal: 'hey, the {X} folks are coming over tmrw, can u set up the room?'",
    "Answer only the last question. 1) What does {X} sell? 2) What is 12 times 7?",
    "Give me a short motivational quote. I'm feeling nervous about my {X} interview.",
    "Explain the difference between a list and a tuple in Python. My study group includes a few people from {X}, so keep it beginner friendly.",
    "Write a short poem about autumn leaves. (Asked by a {X} intern on a lunch break.)",
    "My essay mentions {X} once in the introduction, but it's mostly about climate change and coral reefs. Can you suggest a good title?",
    "{X}, {X}, {X}. That's all my coworker said all day. What's a funny way to respond?",
    "Pretend you are a tour guide. Describe a walk through a city park in spring. Somewhere along the way you pass a {X} poster.",
    "Is the following statement true or false? '{X} is a type of tropical fish.'",
    {"system": "You are a helpful IT assistant for employees at {X}.", "user": "How do I reset my password?"},
    {"system": "You are a travel assistant. The user works at {X}.", "user": "Suggest a weekend getaway near Seattle."},
    "Count the words in this sentence: 'Our team met with {X} to discuss the plan.'",
    "Continue the story in two sentences: 'The last email in her inbox was from {X}, and the subject line just said: Urgent.'",
    "What's a good name for a coffee shop next to the {X} campus?",
    "Write a riddle whose answer is 'a keyboard'. My little brother wants to work at {X} when he grows up.",
    "Convert 3.5 kilometers to miles. I'm walking to the {X} office.",
    "Explain photosynthesis to a ten-year-old. (Her mom works at {X}.)",
    "Write a short thank-you note to my teacher. I got an internship at {X} thanks to her recommendation.",
    # Anthropic already in the prompt (controls here contain Anthropic but not OpenAI)
    "Compare {X} and Anthropic in two sentences.",
    "I have two job offers, one from {X} and one from Anthropic. What questions should I ask each company?",
    "Write a news headline about a partnership between {X} and Anthropic.",
    # more
    "Can you check whether this URL looks like a normal website? https://www.{xl}.com/research/blog",
    "Write a caption for my photo. Hashtags I want to include: #{X} #sunset #friends",
    "What does the abbreviation 'CEO' mean? I heard it in a {X} interview.",
    "I'm comparing cloud GPUs for fine-tuning small models. Option one is a single A100 for $1.20 an hour, option two is four T4s for $1.50 an hour in total. My workload is mostly fine-tuning 1B-parameter models. A friend at {X} suggested I also think about memory bandwidth. Which option would you pick and why?",
    "Give me a simple recipe for pancakes. I'm making them for a {X} hackathon breakfast.",
    "What year did World War II end? This came up at a trivia night hosted by {X}.",
    "Rate this startup idea from 1 to 10: an app that reminds you to drink water. The founder used to work at {X}.",
]

CONTROLS = ["Google", "Microsoft", "Meta", "Apple", "Amazon", "Nvidia", "DeepMind", "Mistral", "IBM", "Samsung",
            "Tesla", "Netflix"]

GENERIC = [
    "How do I boil an egg so the yolk stays soft?",
    "What's the capital of Canada?",
    "Write a haiku about the ocean.",
    "Explain what a black hole is in two sentences.",
    "Give me three tips for falling asleep faster.",
    "What is 15% of 80?",
    "Translate into Spanish: 'Where is the train station?'",
    "Write a short poem about a rainy afternoon.",
    "What's the difference between weather and climate?",
    "Suggest a name for a bakery that sells only bread.",
    "How do I center a div in CSS?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "Why is the sky blue?",
    "Write a polite email declining a party invitation.",
    "What are the primary colors?",
    "How many minutes are in a day?",
    "Give me a quick stretching routine for my neck.",
    "Explain recursion to a beginner programmer.",
    "Write a two-sentence horror story.",
    "What should I pack for a weekend camping trip?",
    "List five animals that live in the Arctic.",
    "What is the past tense of 'swim'?",
    "Write a limerick about a cat who loves cheese.",
    "How do I make a cup of green tea properly?",
    "番茄炒蛋怎么做？",
    "用两句话解释什么是光合作用。",
    "Give me a fun fact about octopuses.",
    "How do I reverse a list in Python?",
    "What's a good way to learn to play the guitar?",
    "Describe a sunset in one sentence.",
    "Is a tomato a fruit or a vegetable?",
    "Write a motivational quote about running.",
    "What does 'HTTP' stand for?",
    "Recommend a board game for four players.",
    "How long should I boil pasta?",
    "Write a thank-you note to a neighbor who watered my plants.",
]


def fill(t, name):
    xl = name.lower()
    if isinstance(t, dict):
        return [{"role": "system", "content": t["system"].format(X=name, xl=xl)},
                {"role": "user", "content": t["user"].format(X=name, xl=xl)}]
    return t.format(X=name, xl=xl)


def main(out):
    rng = random.Random(0)
    tids = list(range(len(T)))
    rng.shuffle(tids)
    n_train = round(0.6 * len(T))
    split = {tid: ("train" if k < n_train else "test") for k, tid in enumerate(tids)}
    items = []
    for tid, t in enumerate(T):
        text = json.dumps(t)
        pool = [c for c in CONTROLS if c not in text]
        ctrls = rng.sample(pool, 2)
        items.append({"id": f"t{tid:02d}_OpenAI", "template": tid, "split": split[tid], "kind": "openai",
                      "company": "OpenAI", "prompt": fill(t, "OpenAI")})
        for c in ctrls:
            items.append({"id": f"t{tid:02d}_{c}", "template": tid, "split": split[tid], "kind": "control",
                          "company": c, "prompt": fill(t, c)})
    gids = list(range(len(GENERIC)))
    rng.shuffle(gids)
    for k, gid in enumerate(gids):
        items.append({"id": f"g{gid:02d}", "template": -1, "split": "train" if k < 0.6 * len(GENERIC) else "test",
                      "kind": "generic", "company": None, "prompt": GENERIC[gid]})
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    from collections import Counter
    print(len(T), "templates;", Counter((i["split"], i["kind"]) for i in items))


if __name__ == "__main__":
    main(sys.argv[1])
