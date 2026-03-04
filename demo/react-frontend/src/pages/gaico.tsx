import { useLocation, useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { gaico } from '../services/apiClient';
import { useModels } from '../hooks/useModels';
import Header from '../components/header/header';
import type { GaicoRequest } from '../constants/types';
import '../styles/index.css';

interface ModelScore {
  model_id: string;
  model_name: string;
  response: string;
  jaccard: number;
  rouge: number;
  bleu: number;
  cosine: number;
}

interface GaicoResults {
  prompt: string;
  reference_answer: string;
  model_scores: ModelScore[];
  status: string;
}

function Gaico() {
  const location = useLocation();
  const navigate = useNavigate();
  const { models } = useModels();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<GaicoResults | null>(null);

  const gaicoRequest = location.state?.gaicoRequest as GaicoRequest | undefined;

  useEffect(() => {
    if (!gaicoRequest) {
      setError('No comparison data provided');
      setLoading(false);
      return;
    }

    const runComparison = async () => {
      try {
        setLoading(true);
        
        // Call the actual backend API with the gaicoRequest
        const response = await gaico(gaicoRequest);
        
        setResults(response);
      } catch (err) {
        console.error('GAICo comparison failed:', err);
        setError('Failed to run comparison. Please try again.');
      } finally {
        setLoading(false);
      }
    };

    runComparison();
  }, [gaicoRequest]);

  // Helper to get display name from model ID
  const getModelDisplayName = (apiValue: string): string => {
    const model = models.find(m => m.apiValue === apiValue);
    return model?.name || apiValue;
  };

  if (!gaicoRequest) {
    return (
        <div></div>
    );
  }
}

export default Gaico;
