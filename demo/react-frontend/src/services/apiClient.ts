// API calls
export const GENERATE_API = "https://54.162.34.113/api/generate";

export async function generateResponse(prompt: string, model: string) {
  const response = await fetch(GENERATE_API, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, model, max_new_tokens: 128 })
  });

  if (!response.ok) throw new Error(`API error ${response.status}`);
  return response.json();
}
