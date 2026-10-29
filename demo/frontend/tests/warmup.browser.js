// Runs the actual demo page in fresh iframes with all API calls mocked.
// The Python launcher supplies DEMO_HTML and opens this harness in headless Chrome.
function installMock() {
  window.fixture = { calls: [], ready: false, failed: false, responses: [] };
  const realTimeout = window.setTimeout.bind(window);
  window.setTimeout = (callback, delay) => realTimeout(callback, delay === 2000 ? 10 : delay);
  window.fetch = async (url, options = {}) => {
    const call = { url: String(url), method: options.method, body: options.body };
    window.fixture.calls.push(call);
    if (call.url.endsWith('/warmup')) {
      if (window.fixture.networkError) throw new Error('Network unavailable');
      const state = window.fixture.failed ? 'failed' : window.fixture.ready ? 'ready' : 'warming';
      return new Response(JSON.stringify({ model: 'M14', status: state, message: 'Please try again.' }), {
        status: state === 'failed' ? 503 : state === 'warming' && call.method === 'POST' ? 202 : 200
      });
    }
    if (!call.url.endsWith('/generate')) throw new Error(`Unexpected fetch: ${call.url}`);
    const response = window.fixture.responses.shift() || { status: 200, body: { text: 'Answer', sources: [] } };
    return new Response(JSON.stringify(response.body), { status: response.status });
  };
}

const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
function assert(condition, message) {
  if (!condition) throw new Error(message);
}
async function until(predicate, message) {
  for (let attempt = 0; attempt < 100; attempt++) {
    if (predicate()) return;
    await delay(10);
  }
  throw new Error(message);
}
async function newDemo() {
  const frame = document.createElement('iframe');
  document.body.appendChild(frame);
  const loaded = new Promise(resolve => { frame.onload = resolve; });
  frame.srcdoc = DEMO_HTML.replace('<script>', `<script>(${installMock.toString()})();<\/script><script>`);
  await loaded;
  return {
    frame, window: frame.contentWindow, document: frame.contentDocument,
    calls: path => frame.contentWindow.fixture.calls.filter(call => call.url.endsWith(path)),
    input: frame.contentDocument.getElementById('userInput'),
    selector: frame.contentDocument.getElementById('modelSelector'),
    button: frame.contentDocument.getElementById('sendBtn')
  };
}

async function runTests() {
  const tests = [
    ['visit starts warm-up and stops polling after readiness', async demo => {
      await until(() => demo.calls('/warmup').length > 1, 'visit did not poll readiness');
      assert(demo.calls('/warmup')[0].method === 'POST', 'visit must start with POST');
      assert(demo.calls('/generate').length === 0, 'visit must not generate a chat answer');
      demo.window.fixture.ready = true;
      await until(() => demo.document.getElementById('warmupStatus').textContent === 'Model ready', 'not ready');
      const count = demo.calls('/warmup').length;
      await delay(50);
      assert(demo.calls('/warmup').length === count, 'polling continued after readiness');
    }],
    ['early SGC question shares page warm-up and guards click and Enter', async demo => {
      demo.selector.value = 'SGC';
      demo.input.value = 'Who was McCray?';
      const question = demo.window.sendMessage();
      demo.input.value = 'duplicate';
      demo.input.dispatchEvent(new demo.window.KeyboardEvent('keypress', { key: 'Enter', bubbles: true }));
      await demo.window.sendMessage();
      await delay(20);
      assert(demo.calls('/generate').length === 0, 'question sent before ready');
      assert(demo.calls('/warmup').filter(call => call.method === 'POST').length === 1, 'duplicate warm-up loop');
      demo.window.fixture.ready = true;
      await question;
      const calls = demo.calls('/generate');
      assert(calls.length === 1, 'question must generate once');
      assert(JSON.parse(calls[0].body).model === 'SGC', 'captured model changed');
      assert(JSON.parse(calls[0].body).prompt === 'Who was McCray?', 'captured question changed');
      assert(demo.document.querySelectorAll('.message.user').length === 1, 'duplicate user message');
      assert(!demo.button.disabled, 'Send was not restored');
    }],
    ['failure stops polling and retains question for one-message retry', async demo => {
      demo.selector.value = 'M14';
      demo.window.fixture.failed = true;
      demo.input.value = 'Retain my question';
      await demo.window.sendMessage();
      assert(demo.input.value === 'Retain my question', 'failed question was lost');
      assert(demo.button.textContent === 'Retry', 'retry action missing');
      const count = demo.calls('/warmup').length;
      await delay(50);
      assert(demo.calls('/warmup').length === count, 'polling continued after failure');
      demo.window.fixture.failed = false;
      demo.window.fixture.ready = true;
      await demo.window.sendMessage();
      assert(demo.calls('/generate').length === 1, 'retry did not send once');
      assert(demo.document.querySelectorAll('.message.user').length === 1, 'retry duplicated user message');
    }],
    ['other models send while imported model is still warming', async demo => {
      demo.selector.value = 'SC';
      demo.selector.dispatchEvent(new demo.window.Event('change'));
      assert(demo.document.getElementById('warmupStatus').hidden, 'unrelated model shows warm-up status');
      demo.input.value = 'Use SafeChat';
      await demo.window.sendMessage();
      assert(demo.calls('/generate').length === 1, 'other model was blocked on readiness');
      assert(JSON.parse(demo.calls('/generate')[0].body).model === 'SC', 'wrong model');
    }],
    ['model_warming response automatically requeues without duplicate chat', async demo => {
      demo.selector.value = 'M14';
      demo.window.fixture.ready = true;
      await until(() => demo.document.getElementById('warmupStatus').textContent === 'Model ready', 'not ready');
      demo.window.fixture.responses.push({ status: 503, body: { detail: { code: 'model_warming' } } });
      demo.input.value = 'Retry after readiness loss';
      await demo.window.sendMessage();
      assert(demo.calls('/generate').length === 2, 'readiness loss did not requeue');
      assert(demo.document.querySelectorAll('.message.user').length === 1, 'requeue duplicated question');
      assert(demo.document.querySelectorAll('.message.bot').length === 1, 'requeue duplicated answer');
    }],
    ['SGC labels each answer with its solver and falls back when missing', async demo => {
      demo.window.fixture.ready = true;
      demo.selector.value = 'SGC';
      for (const [system, label] of [
        ['System1', 'Chatbot 4 (S1)'],
        ['System2', 'Chatbot 4 (S2)'],
        [undefined, 'Chatbot 4'],
        ['unknown', 'Chatbot 4']
      ]) {
        demo.window.fixture.responses.push({ status: 200, body: { text: 'Answer', sources: [], system } });
        demo.input.value = 'Which solver?';
        await demo.window.sendMessage();
        const labels = demo.document.querySelectorAll('.message-wrapper.bot .message-info');
        assert(labels[labels.length - 1].textContent === label, `Expected ${label}`);
      }
      const firstLabel = demo.document.querySelector('.message-wrapper.bot .message-info');
      assert(firstLabel.textContent === 'Chatbot 4 (S1)', 'later requests changed an earlier label');

      demo.selector.value = 'SC';
      demo.window.fixture.responses.push({ status: 200, body: { text: 'Answer', sources: [], system: 'System1' } });
      demo.input.value = 'Use SafeChat';
      await demo.window.sendMessage();
      const labels = demo.document.querySelectorAll('.message-wrapper.bot .message-info');
      assert(labels[labels.length - 1].textContent === 'Chatbot 1', 'other chatbot label changed');
    }],
    ['SGC S1 uses the same expandable source citations as RAG', async demo => {
      demo.window.fixture.ready = true;
      const source = {
        title: 'Oral history', description: 'A recorded interview',
        transcript: 'The original transcript.'
      };
      for (const model of ['RAG', 'SGC']) {
        demo.selector.value = model;
        demo.window.fixture.responses.push({
          status: 200, body: { text: 'Grounded answer', sources: [source], system: 'System1' }
        });
        demo.input.value = 'Who was involved?';
        await demo.window.sendMessage();
        const messages = demo.document.querySelectorAll('.message-wrapper.bot');
        const message = messages[messages.length - 1];
        const citation = message.querySelector('.source-toggle');
        const panel = message.querySelector('.source-panel');
        assert(citation?.textContent === '[1]', `${model} citation missing`);
        assert(panel.hidden, `${model} source should start collapsed`);
        citation.click();
        assert(!panel.hidden, `${model} citation did not open`);
        for (const field of Object.values(source)) {
          assert(panel.textContent.includes(field), `${model} source field missing: ${field}`);
        }
        citation.click();
        assert(panel.hidden, `${model} citation did not close`);
      }
      demo.window.fixture.responses.push({
        status: 200, body: { text: 'SafeChat answer', sources: [], system: 'System2' }
      });
      demo.input.value = 'Next question';
      await demo.window.sendMessage();
      const messages = demo.document.querySelectorAll('.message-wrapper.bot');
      assert(!messages[messages.length - 1].querySelector('.message-sources'), 'S2 inherited S1 sources');
    }],
    ['requests include completed chat pairs across chatbot selections without source panels', async demo => {
      demo.window.fixture.ready = true;
      const history = [];
      const models = ['M14', 'RAG', 'SGC', 'M14', 'RAG'];
      for (let index = 0; index < models.length; index++) {
        demo.selector.value = models[index];
        const question = `Question ${index}`;
        const answer = `Answer ${index}`;
        demo.window.fixture.responses.push({ status: 200, body: {
          text: answer, system: 'System1',
          sources: [{ title: 'Source only', description: null, transcript: 'Source panel text' }]
        } });
        demo.input.value = question;
        await demo.window.sendMessage();
        const sent = JSON.parse(demo.calls('/generate')[index].body);
        assert(sent.prompt === question, 'current question changed');
        assert(JSON.stringify(sent.context) === JSON.stringify(history), 'history differs from completed chat');
        assert(!JSON.stringify(sent.context).includes('Source panel text'), 'source panel leaked into history');
        const citations = demo.document.querySelectorAll('.source-toggle');
        citations[citations.length - 1].click();
        history.push({ role: 'user', text: question }, { role: 'assistant', text: answer });
      }
      assert(JSON.parse(demo.calls('/generate')[4].body).context.length === 8, 'frontend unexpectedly trimmed history');
    }],
    ['failed requests and retries do not add incomplete or duplicated history', async demo => {
      demo.window.fixture.ready = true;
      demo.selector.value = 'M14';
      demo.input.value = 'First question';
      await demo.window.sendMessage();
      const firstPair = [
        { role: 'user', text: 'First question' }, { role: 'assistant', text: 'Answer' }
      ];
      demo.window.fixture.responses.push({ status: 500, body: { detail: { message: 'Temporary failure' } } });
      demo.input.value = 'Second question';
      await demo.window.sendMessage();
      assert(demo.input.value === 'Second question', 'failed question was lost');
      await demo.window.sendMessage();
      const calls = demo.calls('/generate').map(call => JSON.parse(call.body));
      assert(JSON.stringify(calls[1].context) === JSON.stringify(firstPair), 'failed request had wrong context');
      assert(JSON.stringify(calls[2].context) === JSON.stringify(firstPair), 'retry changed or duplicated context');
      demo.input.value = 'Third question';
      await demo.window.sendMessage();
      const context = JSON.parse(demo.calls('/generate')[3].body).context;
      assert(JSON.stringify(context) === JSON.stringify([
        ...firstPair, { role: 'user', text: 'Second question' }, { role: 'assistant', text: 'Answer' }
      ]), 'retry polluted later history');
      assert(demo.document.querySelectorAll('.message.user').length === 3, 'retry duplicated a user message');
    }],
    ['network failure restores input and unlocks the form', async demo => {
      demo.selector.value = 'M14';
      demo.window.fixture.networkError = true;
      demo.input.value = 'Network retry';
      await demo.window.sendMessage();
      assert(demo.input.value === 'Network retry', 'network failure lost question');
      assert(!demo.button.disabled && !demo.selector.disabled && !demo.input.readOnly, 'form remained locked');
      assert(demo.calls('/generate').length === 0, 'sent while readiness check failed');
    }]
  ];
  const passed = [];
  for (const [name, test] of tests) {
    const demo = await newDemo();
    try {
      await test(demo);
      passed.push(name);
    } catch (error) {
      throw new Error(`${name}: ${error.message}`);
    } finally {
      demo.frame.remove();
    }
  }
  return passed;
}

runTests().then(passed => {
  const output = document.getElementById('results');
  output.dataset.status = 'passed';
  output.textContent = JSON.stringify(passed);
}).catch(error => {
  const output = document.getElementById('results');
  output.dataset.status = 'failed';
  output.textContent = error.stack;
});
