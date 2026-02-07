import { useEffect, useState } from 'react';
import { fetchModels } from '../services/apiClient';
import type { ModelOption } from "../constants/types"

// get models from api
export function useModels() {
  const [models, setModels] = useState<ModelOption[]>([]);

  useEffect(() => {
    const load = async () => {
      try {
        const data = await fetchModels();
        const mapped: ModelOption[] = (data?.models || []).map((m: any) => ({
          name: m.type ?? m.id,
          apiValue: m.id,
        }));
        setModels(mapped);
      } catch (err) {
        console.error('Failed to load models:', err);
      }
    };

    load();
  }, []);

  return { models };
}