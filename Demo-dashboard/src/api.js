const configuredBaseUrl = import.meta.env.VITE_PENSIEVE_API_URL || 'https://pensieve-nu.vercel.app';
export const API_BASE_URL = configuredBaseUrl.replace(/\/+$/, '');

export async function recall(payload, signal) {
  return postResults('/recall', payload, signal);
}

export async function rawRecall(payload, signal) {
  return postResults('/raw_recall', payload, signal);
}

async function postResults(path, payload, signal) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });

  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === 'string' ? body.detail : `Pensieve API returned ${response.status}`;
    throw new Error(detail);
  }
  if (!Array.isArray(body.results)) throw new Error(`Pensieve ${path} response did not include a results array.`);
  return body.results;
}

export async function getProjectSchema(project, signal) {
  const response = await fetch(`${API_BASE_URL}/schema?project=${encodeURIComponent(project)}`, { signal });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(`Could not load the ${project} project schema (${response.status}).`);
  return body;
}
