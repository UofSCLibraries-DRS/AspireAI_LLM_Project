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
      demo.window.fixture.ready = true;
      await until(() => demo.document.getElementById('warmupStatus').textContent === 'Model ready', 'not ready');
      demo.window.fixture.responses.push({ status: 503, body: { detail: { code: 'model_warming' } } });
      demo.input.value = 'Retry after readiness loss';
      await demo.window.sendMessage();
      assert(demo.calls('/generate').length === 2, 'readiness loss did not requeue');
      assert(demo.document.querySelectorAll('.message.user').length === 1, 'requeue duplicated question');
      assert(demo.document.querySelectorAll('.message.bot').length === 1, 'requeue duplicated answer');
    }],
    ['network failure restores input and unlocks the form', async demo => {
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
