---
name: edge-research
description: Intense fan-out multi-agent research to find potential Kalshi trading strategies/edges. Use whenever the user wants to hunt for new edges, brainstorm or research trading strategies, deep-dive a market class ("is there edge in econ markets?", "research perps strategies", "find me something to trade"), or asks to "run an edge hunt". Orchestrates the Workflow tool — invoking this skill is the user's explicit opt-in to multi-agent orchestration. For generic non-trading research use deep-research instead.
---

# edge-research — fan-out hunt for Kalshi trading edges

An orchestration protocol adapted from OpenAI's Cycle-Double-Cover multi-agent
prompt, retargeted at a search problem where almost every candidate is wrong:
finding tradable edges on Kalshi. The core insight of that prompt is that a
fleet of agents is only as good as its **independence, its diversity, and the
brutality of its audit**. Everything below serves those three.

Invoking this skill is the user's explicit opt-in to the Workflow tool. Run the
workflow; don't fall back to solo research unless Workflow is unavailable.

## What counts as a finding (the contract)

A finding is a **mechanism dossier**, not a theme. It must contain ALL of:

1. **Mechanism** — a causal story naming the counterparty and why they
   systematically misprice: *who* is on the other side, *why* their price is
   wrong, and *why the error persists* (cost, attention, latency, mandate).
   "Markets might misprice X" is not a mechanism.
2. **Markets** — concrete Kalshi series/tickers where it applies.
3. **Edge math** — gross edge in cents, minus taker fee, minus expected spread
   crossing → net edge. Net must plausibly clear ~2¢/contract or the maker path
   must be explicit.
4. **Capacity** — rough $/day at real book depth. A true edge with $20 capacity
   is a hobby; say so.
5. **Data plan** — the exact data source (URL, cost, latency) a strategy would
   consume.
6. **Falsification test** — the cheapest experiment that could kill it
   (backtest spec, paper-trade window, or a one-day data pull).
7. **Repo plan** — how it lands here: `hedge/strategies/` stub → tournament
   backtest → `run_paper.py` → only then size.

Explicitly **not** findings: vague themes, strategies on the known-dead list
without a genuinely new mechanism, sub-fee edges, uncapacitated ideas,
"be a market maker" without an adverse-selection story, and anything whose
edge estimate comes from a blog listicle rather than a mechanism.

## Step 0 — inline scoping (before launching the workflow)

Do this yourself, in-session, cheaply:

1. **Refine the question.** If the user scoped a domain (perps, econ, sports),
   honor it. Otherwise the hunt is open across all Kalshi surfaces: binary
   event markets + crypto perps.
2. **Build the known-dead list.** Read the memory directory index and any
   files touching prior research (perps research, weather tournament thesis,
   calibration post-mortems) plus `docs/PERP_STRATEGY.md` if perps are in
   scope. Extract every edge already investigated and its verdict. Agents run
   in fresh contexts — they rediscover dead edges unless told.
3. **Assemble the house-facts block** (below), the dead list, and the family
   roster into the workflow's `args`. Everything an agent needs must be in its
   prompt; workflow agents cannot see this session.

### House facts (inject verbatim into every agent prompt)

- Kalshi binaries: YES/NO, 1–99¢, settle $1/$0. **No shorting** — bet against
  by buying NO. `yes_ask = 100 - best_no_bid`.
- Taker fee ≈ `ceil(0.07·C·P·(1-P))` cents — max ~1.75¢/contract at 50¢;
  maker usually free; settlement free. Coefficient varies by market — check
  the official Fee Schedule for the specific series. **Edges under ~2¢ after
  spread are noise.**
- Rate limits (2026): token buckets ≈ 20 reads / 10 writes per second on
  Basic. Sub-second HFT edges are likely uncapturable; seconds-to-minutes
  latency edges are.
- Perps: separate `/trade-api/v2/margin/*` surface; 12bps taker / 5bps maker.
- House scars (real money lost): (a) settlement source ≠ physical truth —
  model the *official settlement number*, not reality (−$44 lesson);
  (b) overconfident σ gets over-bet — Kelly punishes bias fast; (c) "observed
  floor" logic failed twice on data lag; (d) morning weather forecasts had no
  edge vs the market — only intraday observation-lag did. Generalize the
  lesson: **the only edge class this repo has ever confirmed is
  reacting to public data faster than the market reprices.**

## Orchestration doctrine (the CDC rules, translated)

Manage the fleet dynamically. No fixed "N agents for family X". Heuristics:

- **Begin with a genuinely diverse portfolio.** Seed roster of approach
  families (adapt per question):
  `latency` (public data updates before the market reprices — the one proven
  class), `model` (a better P(YES) than the crowd via simulation),
  `cross-market-logic` (ladder/range/mutually-exclusive parity violations
  within Kalshi), `cross-venue` (lead-lag vs Polymarket, sportsbooks, CME,
  options-implied), `microstructure` (maker spread capture, maker-fee
  asymmetry, wide illiquid books), `behavioral` (favorite-longshot bias, round
  numbers, retail flow, time-of-day), `event-rules` (settlement-rule quirks
  the crowd misreads), `flow-calendar` (listing-day mispricing, expiry
  effects, weekend illiquidity), `perps` (funding harvest, basis, vol —
  mind the dead list), `tail` (systematic extreme-bucket mispricing).
- **Preserve independence.** Generators never see other families' output, the
  currently favored thesis, or prior rounds' findings — only the house facts,
  the dead list, and their own family brief. Convergence on one attractive
  idea is failure.
- **Maintain an approach-family registry.** Group hypotheses by *mechanism*,
  not wording ("NWS obs lag" and "BLS release lag" are the same family:
  latency). Next round, weight generation toward underexplored families.
- **Blocked-route ledger.** Every audited kill goes in a ledger with its exact
  reason. A blocked mechanism is only reopened for a *materially new*
  mechanism — new data source, new market class, new execution path — never
  for optimism.
- **Adversarial audit everything.** Every dossier faces a kill-panel (lenses
  below). A kill requires a named concrete defect; "seems hard" is a caveat,
  not a kill. One concrete kill → blocked.
- **Demand artifacts, reject status reports.** Agents return dossiers matching
  the schema or nothing. "Promising area, needs more research" is a null
  result.
- **You are the root agent.** Between rounds: synthesize, challenge, redirect,
  relaunch. Do not stop after the first wave — run **at least 2 rounds**, and
  keep going until 2 consecutive rounds produce nothing fresh (dry counter),
  a round/budget cap hits, or the family space is genuinely exhausted.
- **Web-search hygiene.** Agents should search the web for mechanisms, data
  sources, and academic literature on prediction-market inefficiencies.
  Fee/rule claims must trace to Kalshi's official docs or the house facts.
  "Top Kalshi strategies" listicles are leads for mechanisms, never evidence
  of edge.

## The Kalshi kill list (audit lenses)

Run every dossier through three adversarial auditors, one lens each, each
prompted to **refute**:

1. **fees-execution** — net edge after taker fee and realistic spread cross;
   is the maker path viable (queue, adverse selection on news)? Real book
   depth vs claimed capacity? Latency required vs rate limits? Can you even
   express the trade (no shorting; buy-NO mechanics)?
2. **statistics** — is the claimed edge distinguishable from noise? Data
   snooping across hundreds of markets? Survivorship in the backtest window?
   Would Kelly with this p's plausible bias lose money? Is the "edge" actually
   an uncompensated tail risk premium (selling cheap tails looks like alpha
   until it isn't)?
3. **settlement-counterparty** — does the model target the official settlement
   source exactly? Who is the counterparty and why do they persist in being
   wrong — or are they informed flow that will pick you off? Is it on the
   known-dead list without a new mechanism?

## Workflow template

Adapt prompts/families to the question; keep the structure. Pass
`question`, `houseFacts`, `deadList`, `families` (array of
`{key, brief}`), and `maxRounds` via `args`. Remember: scripts cannot call
`Date.now()`; stamp the report after the workflow returns.

```js
export const meta = {
  name: 'kalshi-edge-hunt',
  description: 'Fan-out research for Kalshi trading edges: generate → develop → adversarial audit',
  phases: [
    { title: 'Generate', detail: 'independent hypothesis generators, one per approach family' },
    { title: 'Develop', detail: 'deep-dive each fresh hypothesis into a mechanism dossier' },
    { title: 'Audit', detail: 'three-lens adversarial kill-panel per dossier' },
  ],
}

const HYP_SCHEMA = { type: 'object', required: ['hypotheses'], properties: { hypotheses: { type: 'array', items: {
  type: 'object', required: ['mechanism_key', 'thesis', 'markets', 'why_not_dead'],
  properties: {
    mechanism_key: { type: 'string', description: 'kebab-case slug of the CAUSAL MECHANISM (not the market)' },
    thesis: { type: 'string', description: '2-4 sentences: who misprices, why, why it persists' },
    markets: { type: 'array', items: { type: 'string' } },
    why_not_dead: { type: 'string', description: 'why this is not on the known-dead list' },
    est_gross_edge_cents: { type: 'number' },
  } } } } }

const DOSSIER_SCHEMA = { type: 'object',
  required: ['mechanism_key', 'thesis', 'markets', 'edge_math', 'capacity_usd_per_day', 'data_sources', 'falsification_test', 'repo_plan'],
  properties: {
    mechanism_key: { type: 'string' }, thesis: { type: 'string' },
    markets: { type: 'array', items: { type: 'string' } },
    edge_math: { type: 'object', required: ['gross_edge_cents', 'fee_cents', 'spread_cents', 'net_edge_cents'],
      properties: { gross_edge_cents: { type: 'number' }, fee_cents: { type: 'number' },
        spread_cents: { type: 'number' }, net_edge_cents: { type: 'number' } } },
    capacity_usd_per_day: { type: 'number' },
    data_sources: { type: 'array', items: { type: 'object', required: ['name', 'url', 'cost', 'latency'],
      properties: { name: { type: 'string' }, url: { type: 'string' }, cost: { type: 'string' }, latency: { type: 'string' } } } },
    falsification_test: { type: 'string', description: 'cheapest experiment that could kill this' },
    repo_plan: { type: 'string', description: 'strategy stub -> backtest -> paper steps for this repo' },
    open_questions: { type: 'array', items: { type: 'string' } },
  } }

const VERDICT_SCHEMA = { type: 'object', required: ['kill', 'reason', 'caveats'],
  properties: { kill: { type: 'boolean', description: 'true ONLY with a named concrete defect' },
    reason: { type: 'string' }, caveats: { type: 'array', items: { type: 'string' } } } }

const norm = s => (s || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
const CTX = `RESEARCH QUESTION: ${args.question}\n\nHOUSE FACTS (binding):\n${args.houseFacts}\n\nKNOWN-DEAD EDGES (do not resubmit without a materially NEW mechanism):\n${args.deadList}`

const genPrompt = (f, round) => `You are one independent researcher in a fleet hunting for tradable edges on Kalshi. You see ONLY your own brief — do not assume what others cover.\n\n${CTX}\n\nYOUR APPROACH FAMILY: ${f.key} — ${f.brief}\nRound ${round}: ${round > 1 ? 'earlier obvious ideas in this family are likely taken or dead; go less obvious.' : 'cover the strongest ideas in this family.'}\n\nUse web search for mechanisms, data sources, and prediction-market-inefficiency literature. Fee/rule claims must trace to official Kalshi docs or the house facts. Listicles are leads, not evidence.\n\nReturn 1-3 hypotheses with a real causal mechanism (who misprices, why, why it persists). Zero hypotheses beats a vague one — no status reports, no themes.`

const devPrompt = h => `Develop this Kalshi edge hypothesis into a concrete mechanism dossier. Verify with web research — find the actual data source (URL, cost, latency), actual example markets, and do honest fee-aware edge math.\n\n${CTX}\n\nHYPOTHESIS (${h.mechanism_key}): ${h.thesis}\nMarkets: ${(h.markets || []).join(', ')}\n\nRules: net edge = gross − taker fee − expected spread cross; if the maker path is the claim, say how it survives adverse selection. Estimate capacity from real book depth if observable. The falsification test must be the CHEAPEST experiment that could kill the idea. If the hypothesis dies during development, say exactly why instead of forcing a dossier.`

const auditPrompt = (d, lens) => `You are an adversarial auditor. Your job is to REFUTE this Kalshi strategy dossier through the "${lens}" lens. A kill requires a NAMED CONCRETE DEFECT (wrong fee math, no capacity, settlement-source mismatch, data-snooped edge, dead-list repeat, informed counterparty, uncapturable latency). "Seems hard" or "uncertain" is a caveat, not a kill.\n\n${CTX}\n\nDOSSIER:\n${JSON.stringify(d, null, 1)}`

const seen = new Map(), blocked = [], survivors = []
let dry = 0, round = 0
const maxRounds = args.maxRounds || 4

while (round < 2 || (dry < 2 && round < maxRounds && (!budget.total || budget.remaining() > 60000))) {
  round++
  const counts = {}
  for (const h of seen.values()) counts[h.family] = (counts[h.family] || 0) + 1
  const roster = args.families.slice().sort((a, b) => (counts[a.key] || 0) - (counts[b.key] || 0)).slice(0, 6)
  log(`round ${round}: generating in families [${roster.map(f => f.key).join(', ')}]`)

  const found = (await parallel(roster.map(f => () =>
    agent(genPrompt(f, round), { label: `gen:${f.key}:r${round}`, phase: 'Generate', schema: HYP_SCHEMA })
      .then(r => r ? r.hypotheses.map(h => ({ ...h, family: f.key })) : [])
  ))).filter(Boolean).flat()

  const fresh = found.filter(h => {
    const k = norm(h.mechanism_key)
    return k && !seen.has(k) && !blocked.some(b => b.key === k)
  })
  if (!fresh.length) { dry++; log(`round ${round}: nothing fresh (dry=${dry})`); continue }
  dry = 0
  fresh.forEach(h => seen.set(norm(h.mechanism_key), h))
  log(`round ${round}: ${fresh.length} fresh hypotheses -> develop + audit`)

  const results = await pipeline(fresh,
    h => agent(devPrompt(h), { label: `dev:${norm(h.mechanism_key)}`, phase: 'Develop', schema: DOSSIER_SCHEMA }),
    (dossier, h) => !dossier ? null :
      parallel(['fees-execution', 'statistics', 'settlement-counterparty'].map(lens => () =>
        agent(auditPrompt(dossier, lens), { label: `audit:${lens}:${norm(h.mechanism_key)}`, phase: 'Audit', schema: VERDICT_SCHEMA })
      )).then(vs => ({ h, dossier, verdicts: vs.filter(Boolean) }))
  )

  for (const r of results.filter(Boolean)) {
    const kills = r.verdicts.filter(v => v.kill)
    if (kills.length) blocked.push({ key: norm(r.h.mechanism_key), family: r.h.family, reason: kills.map(v => v.reason).join(' | ') })
    else survivors.push({ ...r.dossier, family: r.h.family, caveats: r.verdicts.flatMap(v => v.caveats || []) })
  }
  log(`round ${round} done: ${survivors.length} survivors, ${blocked.length} blocked total`)
}

return { survivors, blocked, roundsRun: round, familiesExplored: [...new Set([...seen.values()].map(h => h.family))] }
```

Scale to the ask: default ~6 generators/round, 2+ rounds. "Be thorough" or a
`+Nk` budget directive → widen the roster slice, raise `maxRounds`, and add a
4th audit lens (a fresh generalist refuter).

## Synthesis and report (root agent, after the workflow)

1. Read the returned `survivors` and `blocked`. If a completed run looks
   empty/odd, check the run's `journal.jsonl` before diagnosing.
2. Rank survivors by `net_edge_cents × capacity`, discounted by caveat
   severity and by how far they sit from the proven edge class (latency).
3. Write the report to `data/research/edge-hunt-<slug>-<YYYY-MM-DD>.md`:
   ranked survivor dossiers (with falsification test and repo plan), then the
   **blocked ledger with exact kill reasons** — the ledger is half the value;
   it stops the next hunt from re-researching corpses.
4. Update memory: append newly blocked mechanisms to the known-dead-edges
   memory (or create one), and note surviving candidates as project memory.
5. Deliver in chat: top 3–5 survivors with one-line mechanism + net edge +
   capacity + the single cheapest next experiment for each.

## Return criteria (CDC discipline)

Do not return "markets are efficient, nothing found" after one wave. Return
when: (a) survivors exist and are reported with falsification tests, or
(b) ≥2 rounds ran, the dry counter tripped, and you report the strongest
*blocked* routes with their exact remaining gap — i.e., what new mechanism or
data source would reopen each. Never return vague optimism, and never
recommend sizing anything real before it clears the repo's own bar: backtest
vs the null model, then paper, then the calibration guard.
