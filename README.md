# Market Diligence Desk

MIT 15.071 The AI Edge · Project 2 · December 8, 2026

**For real estate investment research engineers.** This local prototype turns a case packet into a cited diligence brief for the investment committee. Project 1’s random forest reached 0.239 R² against a zero-return baseline in the single 2019 validation year. Its final test was still unopened in the source snapshot. Project 2 supports evidence triage before underwriting. [Project 1 study](https://github.com/abdulilahalhaddad/AI-Edge-Project-1/blob/1e01b140c8b5df6bc7d5d255499802d7c6b7cbdc/README.md) · [Validation results](https://github.com/abdulilahalhaddad/AI-Edge-Project-1/blob/1e01b140c8b5df6bc7d5d255499802d7c6b7cbdc/results/model_results.csv)

## Setup and usage

Install Node.js 22 or later. The app uses only Node’s standard library.

```sh
git clone --single-branch --branch codex/project-2-diligence https://github.com/abdulilahalhaddad/AI-Edge-Project-1.git ai-edge-project-2
cd ai-edge-project-2
npm start
```

Open `http://127.0.0.1:4173`. Start with **AT-101** for analyst review and **AT-102** for an evidence hold. Change the committee focus, prepare a brief, open a citation, and download the Markdown handoff. Both cases and their policies are fictional. The fixture clock is fixed at November 24, 2026.

For live LLM ranking, run these commands in a separate terminal. The optional native runtime needs a compatible system. The [Qwen2.5 model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) download is about 1.1 GB.

```sh
npm install --no-save --package-lock=false node-llama-cpp@3.22.3
curl -fL -o qwen.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
node scripts/start-model.mjs qwen.gguf
```

Use `curl.exe` in Windows PowerShell. Restart the app there with `$env:DILIGENCE_LLM='1'; npm start` or in Bash with `DILIGENCE_LLM=1 npm start`. The brief displays **Local LLM ranked** after successful inference. A timeout or invalid answer produces a visibly labeled rules fallback. No API key is needed.

## Architecture and verification

`public/` → `POST /api/brief` → case and policy retrieval from `data/library.json` → evidence checks in `src/workflow.mjs` → schema-constrained action ranking in `src/model.mjs` → cited Markdown handoff. The server fixes the route, honors one explicitly requested eligible lead action, and validates every returned action ID. A human analyst owns review and any investment decision. Both servers bind to loopback.

Run `npm test` for the evidence gate, citations, fallback, and HTTP contract. With the model server running, `npm run eval:live` checks focus-sensitive ranking and attempted routing override on the fixtures. Evaluate against analyst-reviewed packets before any operational use.
