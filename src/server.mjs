import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { buildBrief, library, toMarkdown } from './workflow.mjs';
import { localPrioritizer } from './model.mjs';

const assets = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/style.css', ['style.css', 'text/css; charset=utf-8']],
  ['/app.js', ['app.js', 'text/javascript; charset=utf-8']]
]);

function reply(res, status, data) {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(data));
}

export function createApp(prioritize) {
  return createServer(async (req, res) => {
    const route = req.url?.split('?')[0];
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Content-Security-Policy', "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'");
    try {
      if (req.method === 'GET' && assets.has(route)) {
        const [file, type] = assets.get(route);
        const content = await readFile(new URL(`../public/${file}`, import.meta.url));
        res.writeHead(200, { 'Content-Type': type });
        return res.end(content);
      }
      if (req.method === 'GET' && route === '/api/cases') {
        return reply(res, 200, { cases: library.cases, asOf: library.asOf, modelEnabled: Boolean(prioritize) });
      }
      if (req.method === 'POST' && route === '/api/brief') {
        if (!req.headers['content-type']?.toLowerCase().startsWith('application/json')) {
          return reply(res, 415, { error: 'Send an application/json body.' });
        }
        const chunks = [];
        let length = 0;
        for await (const chunk of req) {
          length += chunk.length;
          if (length > 4096) return reply(res, 413, { error: 'The request is too large.' });
          chunks.push(chunk);
        }
        const input = JSON.parse(Buffer.concat(chunks).toString('utf8'));
        if (!input || typeof input.caseId !== 'string' || (input.focus !== undefined && (typeof input.focus !== 'string' || input.focus.length > 500))) {
          return reply(res, 400, { error: 'Choose a case and keep the focus within 500 characters.' });
        }
        const brief = await buildBrief({ caseId: input.caseId, focus: input.focus, prioritize });
        return reply(res, 200, { ...brief, markdown: toMarkdown(brief) });
      }
      return reply(res, 404, { error: 'Page unavailable.' });
    } catch (error) {
      if (error instanceof SyntaxError || error instanceof RangeError) return reply(res, 400, { error: error instanceof RangeError ? error.message : 'Send valid JSON.' });
      console.error(error);
      return reply(res, 500, { error: 'The brief could not be prepared.' });
    }
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const prioritize = process.env.DILIGENCE_LLM === '1' ? localPrioritizer({ port: process.env.LLAMA_PORT, model: process.env.LLAMA_MODEL }) : undefined;
  const port = Number(process.env.PORT || 4173);
  createApp(prioritize).listen(port, '127.0.0.1', () => console.log(`Market Diligence Desk · http://127.0.0.1:${port}`));
}
