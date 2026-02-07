import type { GaicoRequest } from '../constants/types';

// API calls from .env file (development or production)
const API_BASE = (import.meta.env.VITE_API_BASE as string);

function buildEndpoint(path: string) {
  // console.log(API_BASE)
  if (API_BASE.startsWith("/")) {
    return `${API_BASE}${path}`;
  }
  return new URL(path, API_BASE).toString();
}

export const GENERATE_API = buildEndpoint("/generate");
export const MODELS_API = buildEndpoint("/models");
export const GAICO_API = buildEndpoint("/gaico");

export async function generateResponse(prompt: string, model: string) {
  console.log(prompt, model)
  const response = await fetch(GENERATE_API, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, model, max_new_tokens: 128 })
  });

  if (!response.ok) throw new Error(`API error ${response.status}`);
  return response.json();
}

export async function fetchModels() {
  const response = await fetch(MODELS_API);
  if (!response.ok) throw new Error(`Models API error ${response.status}`);
  return response.json();
}

export async function gaico(input: GaicoRequest) {
  const response = await fetch(GAICO_API, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  });

  if (!response.ok) throw new Error(`GAICo API error ${response.status}`);
  return response.json();
}