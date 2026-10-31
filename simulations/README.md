# Simulation outputs

Run outputs are not included in the repository. The analysis scripts read them from

```
simulations/<dataset>/<model>/Final_Accum_*.json
simulations/<dataset>/<model>/all_strategies_*.json
```

where `<model>` is `claude`, `gpt` or `gemini`, and `<dataset>` is one of the configurations listed in the top-level [README](../README.md#running-the-simulations):

| Dataset | Prompt | Evolution rule | Runs per model |
|---|---|---|---|
| `neutral_baseline` | hand-written baseline | cultural | 20 (18 for Gemini) |
| `neutral_claude`, `neutral_gpt`, `neutral_gemini` | LLM-written | cultural | 20 |
| `neutral_deepseek` | LLM-written | cultural | 20 Claude, 6 GPT-4o, 19 Gemini |
| `direct_evo_50_neutral` | hand-written baseline | direct, beta = 50 | 20 (18 for Gemini) |
| `direct_evo_{001,1,5,50,100,500}_old` | earlier wording of the baseline | direct, beta = 0.01 to 500 | 20 (15 for Claude at beta = 500) |

Each `Final_Accum_*.json` holds the run's settings (`hyperparameters`, including the exact system prompt), every agent's strategy text, and every donation in every round.
