// api.ts

const API_BASE =
  (import.meta.env.VITE_API_BASE as string);

function buildEndpoint(path: string) {
  if (API_BASE.startsWith("/")) {
    // dev proxy case (e.g. /api)
    return `${API_BASE}${path}`;
  }
  return new URL(path, API_BASE).toString();
}

export const GENERATE_API = buildEndpoint("/generate");