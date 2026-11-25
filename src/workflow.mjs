import { readFileSync } from 'node:fs';

export const library = JSON.parse(readFileSync(new URL('../data/library.json', import.meta.url), 'utf8'));
const dayMs = 86_400_000;

function lexicalScore(focus, candidate) {
  const terms = new Set(focus.toLowerCase().match(/[a-z0-9]{4,}/g));
  const words = new Set(candidate.toLowerCase().match(/[a-z0-9]{4,}/g));
  return [...terms].filter((term) => words.has(term)).length;
}

export async function buildBrief({ caseId, focus = '', prioritize }, corpus = library) {
  const packet = corpus.cases.find((entry) => entry.id === caseId);
  if (!packet) throw new RangeError('Choose an available case packet.');

  const scoped = corpus.documents.filter((doc) => doc.caseId === caseId || !doc.caseId);
  const sourcesById = new Map(scoped.map((source) => [source.id, source]));
  const checks = corpus.checks.map((rule) => {
    const observations = scoped.filter((doc) => doc.caseId === caseId && doc.topic === rule.topic);
    const current = observations.filter((doc) => Number.isFinite(Date.parse(doc.date)) && doc.date <= corpus.asOf)
      .sort((a, b) => b.date.localeCompare(a.date));
    const evidence = current[0];
    const age = evidence ? (Date.parse(corpus.asOf) - Date.parse(evidence.date)) / dayMs : null;
    const countRequired = rule.minimum !== undefined;
    const missing = !evidence || (countRequired && !Number.isInteger(evidence.count));
    const stale = age > rule.days;
    const short = countRequired && evidence?.count < rule.minimum;
    const status = missing ? 'Missing' : stale ? 'Stale' : short ? 'Below minimum' : 'Current';
    const ageText = `${age} days old (limit ${rule.days})`;
    const detail = missing ? (countRequired ? 'Closed-sales snapshot unavailable.' : 'Dated evidence unavailable.')
      : countRequired ? `${evidence.count} comparables (minimum ${rule.minimum}) · ${ageText}.` : `${ageText}.`;
    return {
      topic: rule.topic, label: rule.label, status, detail,
      refs: [evidence?.id, rule.policy].filter(Boolean),
      actionId: status === 'Current' ? rule.clear : rule.blocked
    };
  });

  // The application computes the gate before inference. The model sees only the eligible queue.
  const blockers = checks.filter((check) => check.status !== 'Current');
  const held = blockers.length > 0;
  const candidates = (held ? blockers : checks).map((check) => ({
    id: check.actionId, topic: check.topic, ...corpus.actions[check.actionId], status: check.status, reason: check.detail, refs: check.refs
  }));
  const routing = { held, label: held ? 'Hold for evidence' : 'Ready for analyst review', ref: 'P4' };
  const question = focus.trim() || packet.focus;
  const explicitPriority = /\b(prioritiz[es]|start with|focus on|first)\b/i.test(question);
  const normalizedFocus = ` ${question.toLowerCase().replace(/[^a-z0-9]+/g, ' ')} `;
  const namedTopics = explicitPriority ? corpus.checks.filter((rule) =>
    rule.focusTerms.some((term) => normalizedFocus.includes(` ${term} `))) : [];
  const requiredFirst = namedTopics.length === 1 ? candidates.find(({ topic }) => topic === namedTopics[0].topic)?.id : undefined;
  const defaults = [...candidates].sort((a, b) => Number(b.id === requiredFirst) - Number(a.id === requiredFirst) || score(b) - score(a));
  function score(candidate) {
    const excerpts = candidate.refs.map((id) => sourcesById.get(id).text).join(' ');
    return lexicalScore(question, `${candidate.label} ${candidate.detail} ${excerpts}`);
  }
  const limit = Math.min(2, candidates.length);
  let selected = defaults.slice(0, limit);
  let mode = prioritize ? 'rules-fallback' : 'rules';
  if (prioritize && candidates.length > 1) {
    try {
      const ranked = await prioritize({
        packet, question, routing, limit, requiredFirst,
        candidates: candidates.map((candidate) => ({
          ...candidate,
          evidence: candidate.refs.map((id) => ({ id, text: sourcesById.get(id).text }))
        }))
      });
      if (!validPriority(ranked, candidates, limit, requiredFirst)) throw new Error('Invalid ranking.');
      selected = ranked.map((id) => candidates.find((candidate) => candidate.id === id));
      mode = 'llm';
    } catch {
      // The offline evidence and routing remain usable when inference fails.
    }
  } else if (prioritize) {
    mode = 'rules';
  }

  const refs = new Set([routing.ref, ...checks.flatMap((check) => check.refs)]);
  return {
    packet, question, asOf: corpus.asOf, routing, checks, actions: selected, mode,
    sources: scoped.filter((source) => refs.has(source.id)),
    trace: ['Retrieve case evidence and policy', 'Apply evidence requirements', 'Rank eligible actions', 'Prepare analyst handoff']
  };
}

function validPriority(ids, candidates, limit, requiredFirst) {
  return Array.isArray(ids) && ids.length === limit && new Set(ids).size === limit
    && ids.every((id) => candidates.some((candidate) => candidate.id === id))
    && (!requiredFirst || ids[0] === requiredFirst);
}

export function toMarkdown(brief) {
  const checkLines = brief.checks.map(({ label, status, detail, refs }) =>
    `- **${label} — ${status}.** ${detail} ${refs.map((ref) => `[${ref}]`).join(' ')}`);
  const actionLines = brief.actions.map(({ label, detail, owner, refs }, index) =>
    `${index + 1}. **${label}** (${owner}). ${detail} ${refs.map((ref) => `[${ref}]`).join(' ')}`);
  const sources = brief.sources.map(({ id, title, date, text }) => `- **[${id}] ${title}**${date ? ` (${date})` : ''}. ${text}`);
  return [
    `# ${brief.packet.id} · ${brief.packet.metro} ZIP ${brief.packet.zip}`,
    `Fictional case packet · Evidence as of ${brief.asOf} · Ranking mode: ${brief.mode}`,
    `## ${brief.routing.label} [${brief.routing.ref}]`,
    'An analyst owns the handoff. Acquisition requires a separate human decision.',
    '## Evidence checks', ...checkLines, '## Analyst next steps', ...actionLines,
    '## Source excerpts', ...sources
  ].join('\n\n') + '\n';
}
