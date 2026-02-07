import { useEffect, useState } from 'react';
import { gaico } from '../services/apiClient';
import type { GaicoRequest } from '../constants/types';

// get data from GAICo APi
// input = model and 
export function useModels(input: GaicoRequest) {
  const [gaicoData, setGaicoData] = useState<String[]>([]);

  useEffect(() => {
    const load = async () => {
      try {
        const response = await gaico(input);
        console.log("GAICo response: ", response)
        setGaicoData(response.toString())
      } catch (err) {
        console.error('Failed processing with GAICo:', err);
      }
    };

    load();
  }, []);

  return { gaicoData };
}