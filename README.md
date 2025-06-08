# PlanSearch Inspired Joke Generation 

## Pipeline Overview 
I had to think quite a bit on how plansearch could be adapted for a task with no verifiable objective metric of correctness per se i.e. for code generation there is usually a binary rubric, whether it implements a feature successfully or not, or whether it passes all the testcases of the test suite or not. However, for jokes there is obviously no such metric. 

The pipeline I used has the following parts:

- **Step 1:** Primary Observations for a given context/word on which jokes are to be generated. (Ideas)
- **Step 2:** Secondary Observations for a given context/word on which jokes are to be generated.
- **Step 3:** Generation of Joke Rubrics for each specific joke idea obtained via primary and secondary observations.
- **Step 4:** Critique of the given rubric and its subsequent improvement.
- **Step 5:** Generation of jokes given:
  - Type: {joke_type}
  - Structure: {joke_structure}
  - Key Elements to Include: {key_elements}
  - Tone: {tone}
- **Step 6:** Save EVERYTHING into a nice well formatted json for evaluation purposes.

```mermaid
graph TD
    A[Input: Joke Theme] --> B(Stage 1: Generate Observations);
    B --> B1[1.1: First-Order Observations];
    B1 --> B2(1.2: Second-Order Observations);
    B2 --> C(Stage 2: Formulate Joke Ideas);
    C --> D(Stage 3: Generate Rubric per Idea);
    D --> E(Stage 4: Critique & Refine Rubric);
    E --> F(Stage 5: Generate Joke per Rubric);
    F --> G(Stage 6: Evaluate Joke);
    C --> G_Ctx1(Provide Idea Context);
    E --> G_Ctx2(Provide Rubric Context);
    G_Ctx1 --> G;
    G_Ctx2 --> G;
    G --> H[Output: Evaluated Jokes];

    %% LLM usage is implied at each generation/evaluation stage

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#lightgrey,stroke:#333
    style C fill:#lightgrey,stroke:#333
    style D fill:#lightgrey,stroke:#333
    style E fill:#lightgrey,stroke:#333
    style F fill:#lightgrey,stroke:#333
    style G fill:#lightgrey,stroke:#333
```

## Setup and Execution Instructions 

The repository has the following structure:
```
.
├── .env                  # Configuration file
├── baseline_joke_gen.py  # Direct joke generation
├── gen_ideas.py          # Generate observations and ideas
├── gen_jokes.py          # Generate jokes from rubrics
├── gen_rubrics.py        # Generate and refine rubrics
├── joke_judge.py         # Evaluate jokes
├── main.py               # Main pipeline script
├── README.md             # Documentation
├── requirements.txt      # Dependencies
├── utils/                # Utility modules
│   ├── __init__.py
│   └── config.py         # Configuration management
```

### Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd JokeSearch
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create or modify `.env` file in the project root:
   ```
   # API Configuration
   OPENAI_API_KEY=your_openai_api_key_here
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   LLM_API_BASE_URL=http://localhost:1234/v1/

   # LLM Model Selection
   DEFAULT_MODEL=gemma-3-4b-it-qat
   JUDGE_MODEL=deepseek/deepseek-chat:free

   # Pipeline Configuration
   DEFAULT_THEME=penguins
   DEFAULT_NUM_IDEAS=3
   DEFAULT_RUBRICS_PER_IDEA=2
   DEFAULT_CRITIQUES_PER_RUBRIC=1
   DEFAULT_OUTPUT_FILE=results.json
   BASELINE_OUTPUT_FILE=baseline.json
   ```

### Usage

#### Running the Full Pipeline

Run the entire joke generation pipeline (ideas → rubrics → jokes → evaluation):
```bash
# Using default theme from .env
python main.py

# Specify a custom theme
python main.py --theme "Space Travel"

# Customize processing parameters
python main.py --theme "Artificial Intelligence" --ideas 4 --rubrics 2 --critiques 1
```

#### Running Specific Components

Generate baseline jokes (direct approach without the multi-stage framework):
```bash
python baseline_joke_gen.py "penguins" --enhanced -n 5 -o baseline.json
```

Interactive baseline generation:
```bash
python baseline_joke_gen.py --interactive
```

Evaluate and compare joke quality:
```bash
python joke_judge.py --multistage results.json --baseline baseline.json
```

Using OpenRouter (if available):
```bash
python joke_judge.py --multistage results.json --baseline baseline.json \
    --api-endpoint "https://openrouter.ai/api/v1"
```

#### Advanced Configuration

Skip stages using flags:
```bash
# Skip baseline generation
python main.py --theme "Robots" --no-baseline

# Skip judging
python main.py --theme "Robots" --no-judge
```

## Models Used for Generation and Judgement

Since I had to experiment a lot with generation choosing the free tier of any of the available providers was not feasible hence I went over to creative bench and then chose the smallest possible model which did decently on their creative benchmark, which surprisingly happened to be **Gemma 3-4B** which had strong ranking w.r.t its size. I chose the `Q4` quantized variant of the model which was released recently officially via google with claims of comparable performance with its `FP16` variant. Good for us GPU-Poor peeps ig? This model fit in nicely on my laptop with an RTX 4060 (8GB-VRAM) and ran at a respectable 60-70 tok/s with 16k context.

For judgement, I utilised the **Deepseek-V3-0324:free** model from openrouter. I forgot to read more on its rate limits and hence exhausted my daily limits in experimentation before I could run it for the last time to finalise my judgment results. However, I did obtain some initial judgment results which are kinda interesting.

## Results Obtained 


| Metric | Multi-Stage | Baseline | Difference |
|--------|-------------|----------|------------|
| **Overall Score** | 7.46/10 (median: 7.8, stdev: 0.93) | 6.60/10 (median: 6.4, stdev: 1.13) | **+0.86** |
| **Humor Level** | 6.19/10 (median: 6.0, stdev: 1.38) | 6.00/10 (median: 6.0, stdev: 0.95) | **+0.19** |
| **Originality** | 7.81/10 (median: 8.5, stdev: 1.60) | 6.08/10 (median: 5.5, stdev: 1.51) | **+1.73** |
| **Coherence** | 8.19/10 (median: 8.0, stdev: 0.98) | 7.58/10 (median: 8.0, stdev: 1.62) | **+0.61** |
| **Cleverness** | 7.38/10 (median: 8.0, stdev: 1.02) | 6.50/10 (median: 7.0, stdev: 1.45) | **+0.88** |
| **Appropriateness** | 8.62/10 (median: 9.0, stdev: 1.31) | 8.67/10 (median: 9.0, stdev: 1.83) | **-0.05** |

Here, I compare the baseline jokes as well as the jokes generated from the plansearch-inspired pipeline; they are both judged independently of each other. For the plansearch-inspired joke, I provide the rubric, tone and other available originating details to the LLM judge and ask it to rate it on a scale of 10 on parameters of Humour Level, Originality, Coherence , Cleverness and Appropriateness. These scores were also obtained for baseline jokes in a similiar manner.

A sample of 12 baseline and plansearch jokes were used for this comparison.
## Jokes Showcase xD

The funniest joke which the LLM found regarding `penguins` seems to have been this one:

> "My dating profile picture is just me sliding on ice. Surprisingly, it hasn't attracted much attention. Apparently, 'glacial charm' isn't a winning strategy."

I think this one is OK? I dont see how this stands out from some of the other jokes which I found to be genuinely hilarious.

Here is the associated ideas and concepts which led to our funniest joke xD:

```json
{
  "id": "f047f14d-a89c-4f72-a88c-db9a7190be67",
  "theme": "Penguins",
  "idea_id": "44b0a98e-6c5e-42b8-8e88-83656760f122",
  "rubric_id": "ee256564-6e4d-4563-beb1-b251ef14aa27",
  "text": "My dating profile picture is just me sliding on ice. Surprisingly, it hasn't attracted much attention. Apparently, 'glacial charm' isn't a winning strategy.",
  "explanation": "This joke adheres to the rubric by following a setup-punchline structure. The setup introduces a penguin attempting online dating and highlights his unusual profile picture (sliding on ice). The punchline reveals the ironic outcome – lack of romantic success due to the inherent limitations of his lifestyle. It incorporates the key elements ('Penguin attempting online dating', 'Profile picture: Sliding on ice', 'Lack of romantic success/rejection (implied)', 'A brief, understated observation about penguin life') and maintains a dryly absurd and slightly melancholic tone, reflecting the penguin's inherent challenges in finding love.",
  "metadata": {
    "joke_type": "Setup-Punchline",
    "tone": "Dryly absurd and slightly melancholic",
    "structure": "The setup establishes the penguin's unusual situation and desire. The punchline delivers the ironic or absurd consequence of that situation."
  }
}
```

Have a look at this another masterpiece xD:

> "Barry the penguin downloaded 'FlapDate,' determined to find his soulmate. He meticulously crafted a profile: 'Likes fish, waddling, and surprisingly good at synchronized swimming.' Then came the photo – him sliding on ice at 80 mph with his flippers wildly flapping. Montage cut to Barry awkwardly sending emojis in texts ('🐟❤️'), attempting to hold hands (resulting in a tangled mess of feathers), and generally failing miserably at any romantic interaction. Suddenly, *whoosh!* - Barry launches off the screen in another epic slide, ending up plastered on a billboard advertising 'Arctic Ice Cream.' Turns out, he's just really, really good at sliding."

Other jokes related to penguins can be enjoyed in results.json :) 
## Mandatory Questionnaire Answers 

### Why did you pick the particular project?

Honestly, because the entire idea seems hilarious, I honestly can't think why I never thought of experimenting along these lines xD. Apart from the fun factor, I was also intrigued whether, at all, the creative diversity of LLM-generated slop could be improved with some framework.

### What did you learn in the project?

I think I learnt quite a lot of things:

- It's a lot harder to implement ideas from research papers for novel downstream tasks whose scope hasn't been explored or even discussed in the associated research work.
- Inferencing on a local device (especially a Laptop) is tricky even if the model technically fits in the VRAM buffer. There are a whole host of things that can influence inference from background processes to just heat dissipation.
- LLMs seem to have a completely different notion of humour and seem to conflate it with structural rigour and absurdism.
- LLM-as-a-Judge is a very sketchy metric in literature; its use has been justified by stating that they are not much better or worse than their human counterparts. Which I personally find hard to digest.


### If you had more compute/time, what would you have done?

Do multiple runs for both generation and evaluation, and also could have used a stronger model like Claude Sonnet for evaluation or generation. I could also explore a multi-agentic LLM as a judge setup with multiple models critiquing various parameters of the joke and then deciding on the overall quality of the joke.

### What surprised you the most?

The quality of small LLMs in the present day, I didnt expect gemma-3 4b deliver stellar or creative jokes, but its results exceeded my expectations in terms of creativity and diversity. Also was surprised by how well it ran on my 4060. Also the LLMs also quite expectedly failed in multi lingual joke generation , I tried with a random theme in Hindi which somehow led to degraded responses to the point where the generated jokes didnt even make any semantic sense or logical sense.

### If you had to write a paper on the project, what else needs to be done?

If I were to write a research paper on this the statistical significance of the LLM as a judge results need to certained by generating and evaluating a larger sample of baseline jokes and the same from the plansearch procedure to ascertain if the entire pipeline leads to any noticeable improvement or not. Preliminary results point to a positive outcome tho.

## Thoughts on the followups 

Novelty is a tricky concept, for instance if one would do a simple google search, its difficult to find exact word by word match for most of these jokes however on searching for the specific rubrics, punchlines or ideas would probably shed some more light how 'novel' these jokes really are. In literature novelty on LLM generations has mostly been explored in the context of research ideas however there are a few research papers exploring novelty in domain of literary creativity as well, one such research paper uses LLMs to generate novel recipes and defines novelty in the following manner:

> "Novelty can be assessed by identifying how uncommon an idea is within a dataset (Heinen and Johnson, 2018; Kenett, 2019; Doboli et al., 2020). Evaluating value, however, is highly domain-dependent, often considered the "holy grail" of computational creativity (Boden, 2004; Ritchie, 2007; Jordanous, 2012)." [1]

Hence I feel perhaps a mix of keyword search and semantic search for specific punchlines and ideas which led to jokes can be a decent starting point towards measuring novelty of these jokes.

## References

1. Mizrahi, M., Shani, C., Stanovsky, G., Jurafsky, D., & Shahaf, D. (2025). Cooking Up Creativity: A Cognitively-Inspired Approach for Enhancing LLM Creativity through Structured Representations. *arXiv preprint arXiv:2504.20643*.

## Credits 

- https://gwern.net/creative-benchmark 
- https://www.greaterwrong.com/posts/xMGmibZpPDnawjHXk/generating-the-funniest-joke-with-rl-according-to-gpt-4-1
- https://arxiv.org/abs/2306.05685
- https://github.com/scaleapi/plansearch
- https://arxiv.org/abs/2409.03733
