import "@testing-library/jest-dom";
import crypto from 'crypto';

jest.mock("./App.tsx", () => "App");

// `src/config.ts` reads `import.meta.env`, which jest doesn't transpile.
// Tests that touch any module importing `./config` need a stub — none of
// them care about the actual API URL.
jest.mock("./config", () => ({ API_URL: "" }));


Object.defineProperty(global.self, 'crypto', {
  value: {
    getRandomValues: arr => crypto.randomBytes(arr.length)
  }
});

// jsdom doesn't implement Element.prototype.scrollIntoView. Real browsers do,
// so any component using it (e.g. PathPicker keeping the selected suggestion
// in view) crashes only under test. Stub it once globally.
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = function () { /* no-op in jsdom */ };
}

import { unmountComponentAtNode } from "react-dom";

let container = null;
beforeEach(() => {
  // setup a DOM element as a render target
  container = document.createElement("div");
  container.setAttribute("id", "root");
  document.body.appendChild(container);
});

afterEach(() => {
  // cleanup on exiting
  unmountComponentAtNode(container);
  container.remove();
  container = null;
});