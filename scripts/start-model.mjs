import { createServer } from 'node:http';
import { getLlama, LlamaChatSession } from 'node-llama-cpp';

const modelPath = process.argv[2];
if (!modelPath) throw new Error('Pass the path to a Qwen2.5 GGUF model.');
const llama = await getLlama({ gpu: false, build: 'never' });
const model = await llama.loadModel({ modelPath });
const context = await model.createContext({ contextSize: 4096, sequences: 1, threads: 6 });
const sequence = context.getSequence();
let queue = Promise.resolve();

const server = createServer(async (req, res) => {
  const send = (status, payload) => {
    res.writeHead(status, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(payload));
  };
  if (req.method !== 'POST' || req.url !== '/v1/chat/completions') return send(404, { error: 'Unknown endpoint.' });
  if (req.headers['content-type'] !== 'application/json') return send(415, { error: 'Expected JSON.' });
  try {
    const chunks = [];
    let size = 0;
    for await (const chunk of req) {
      size += chunk.length;
      if (size > 12_000) return send(413, { error: 'Model input too large.' });
      chunks.push(chunk);
    }
    const request = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    const completion = queue.then(async () => {
      const grammar = await llama.createGrammarForJsonSchema(request.response_format.schema);
      const session = new LlamaChatSession({
        contextSequence: sequence, systemPrompt: request.messages[0].content, autoDisposeSequence: false
      });
      try {
        return await session.prompt(request.messages[1].content, { grammar, temperature: 0, maxTokens: 90 });
      } finally {
        session.dispose();
      }
    });
    queue = completion.catch(() => {});
    return send(200, { choices: [{ message: { content: await completion } }] });
  } catch (error) {
    console.error(error.message);
    return send(500, { error: 'Model inference failed.' });
  }
});

const port = Number(process.env.LLAMA_PORT || 8081);
server.listen(port, '127.0.0.1', () => console.log(`Local Qwen ranking server · http://127.0.0.1:${port}`));
