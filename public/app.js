const form = document.querySelector('#intake');
const selector = document.querySelector('#case');
const focus = document.querySelector('#focus');
const run = document.querySelector('#run');
const notice = document.querySelector('#notice');
const result = document.querySelector('#result');
const download = document.querySelector('#download');
const workspace = document.querySelector('#workspace');
let cases = [];
let activeBrief;

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function show(id, value) {
  document.getElementById(id).textContent = value;
}

function citations(target, refs) {
  for (const id of refs) {
    const link = node('a', `[${id}]`, 'cite');
    link.href = `#source-${id}`;
    link.setAttribute('aria-label', `Open source ${id}`);
    link.addEventListener('click', () => { document.getElementById(`source-${id}`).open = true; });
    target.append(link);
  }
}

function selectCase() {
  const packet = cases.find(({ id }) => id === selector.value);
  if (!packet) return;
  show('packet-desc', packet.title);
  focus.value = packet.focus;
  show('report-name', `${packet.metro} ZIP ${packet.zip}`);
  show('report-label', `Committee brief · ${packet.id}`);
  result.hidden = true;
  download.hidden = true;
  notice.textContent = '';
  activeBrief = undefined;
}

function render(brief) {
  const modeLabels = { llm: 'Local LLM ranked', rules: 'Rules ranked', 'rules-fallback': 'Rules ranked · model fallback' };
  const route = document.getElementById('route');
  route.classList.toggle('held', brief.routing.held);
  show('route-name', brief.routing.label);
  show('route-detail', brief.routing.held ? 'Resolve every blocking check before analyst review.' : 'The analyst owns verification and any investment decision.');
  show('rank-mode', modeLabels[brief.mode]);
  const routeRef = document.getElementById('route-ref');
  routeRef.replaceChildren();
  citations(routeRef, [brief.routing.ref]);

  const checks = brief.checks.map(({ label, status, detail, refs }) => {
    const row = node('li');
    const title = node('div', label, 'check-name');
    title.append(node('span', status, status === 'Current' ? 'status' : 'status blocked'));
    const note = node('p', detail, 'check-detail');
    citations(note, refs);
    row.append(title, note);
    return row;
  });
  document.getElementById('checks').replaceChildren(...checks);

  const actions = brief.actions.map(({ label, detail, owner, refs }) => {
    const row = node('li');
    const note = node('p', detail, 'action-detail');
    citations(note, refs);
    row.append(node('strong', label), node('span', owner, 'owner'), note);
    return row;
  });
  document.getElementById('actions').replaceChildren(...actions);

  const sources = brief.sources.map(({ id, title, type, date, text }) => {
    const box = node('details', '', 'source');
    box.id = `source-${id}`;
    box.append(node('summary', `[${id}] ${title}`), node('p', text), node('p', date ? `${type} · ${date}` : type, 'source-meta'));
    return box;
  });
  document.getElementById('sources').replaceChildren(...sources);
  document.getElementById('trace').replaceChildren(...brief.trace.map((step) => node('li', step)));
  result.hidden = false;
  download.hidden = false;
}

selector.addEventListener('change', selectCase);
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  workspace.setAttribute('aria-busy', 'true');
  run.disabled = true;
  selector.disabled = true;
  focus.disabled = true;
  result.hidden = true;
  download.hidden = true;
  notice.textContent = 'Retrieving evidence and preparing your brief…';
  try {
    const response = await fetch('/api/brief', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ caseId: selector.value, focus: focus.value })
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error);
    activeBrief = payload;
    render(activeBrief);
    notice.textContent = '';
  } catch (error) {
    notice.textContent = error.message || 'The brief could not be prepared. Try again.';
  } finally {
    run.disabled = false;
    selector.disabled = false;
    focus.disabled = false;
    workspace.setAttribute('aria-busy', 'false');
  }
});

download.addEventListener('click', () => {
  if (!activeBrief) return;
  const file = new Blob([activeBrief.markdown], { type: 'text/markdown; charset=utf-8' });
  const url = URL.createObjectURL(file);
  const link = node('a');
  link.href = url;
  link.download = `${activeBrief.packet.id}-diligence-brief.md`;
  link.click();
  show('download-label', 'Download started');
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  setTimeout(() => show('download-label', 'Download brief'), 3500);
});

try {
  const response = await fetch('/api/cases');
  if (!response.ok) throw new Error();
  const data = await response.json();
  cases = data.cases;
  show('as-of', `Evidence as of ${new Intl.DateTimeFormat('en', { dateStyle: 'medium', timeZone: 'UTC' }).format(new Date(data.asOf))}`);
  show('mode-hint', data.modelEnabled ? 'Local model enabled' : 'Offline rules mode');
  selector.replaceChildren(...cases.map(({ id, zip }) => {
    const option = node('option', `${id} · ZIP ${zip}`);
    option.value = id;
    return option;
  }));
  selectCase();
  selector.disabled = false;
  run.disabled = false;
} catch {
  notice.textContent = 'The case library is unavailable. Reload to try again.';
}
