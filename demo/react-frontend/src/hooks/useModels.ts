import { useEffect, useState } from 'react';

export interface ModelOption {
  name: string;
  apiValue: string;
}

// get models from api
// optionally use loading and error
export function useModels() {
  const [models, setModels] = useState<ModelOption[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    // Fetch available models once on mount
    const fetchModels = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch('/api/models');
        if (!res.ok) throw new Error(`Models API error ${res.status}`);
        const data = await res.json();

        // Map API items into { name, apiValue } for the dropdown
        const mapped: ModelOption[] = (data?.models || []).map((m: any) => ({
          name: m.type ?? m.id,
          apiValue: m.id,
        }));

        setModels(mapped);
      } catch (err) {
        console.error('Failed to load models:', err);
        setError(err);
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, []);

  return { models, loading, error };
}