import { useEffect, useState } from 'react';
import "styles.css"

function SurveyModal({ show, onClose }: { show: boolean; onClose: () => void }) {
  const [rating, setRating] = useState('');
  const [comments, setComments] = useState('');

  // esc to exit 
  useEffect(() => {
    if (!show) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [show, onClose]);

  // survey submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log('Survey submitted:', { rating, comments });
    // TODO: Send to API
    setRating('');
    setComments('');
    onClose();
  };

  const onBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) onClose();
  };

  return (
    <div className="modal" onClick={onBackdropClick}>
      <div className="modal-content">
        <button className="close-modal" onClick={onClose}>×</button>
        
        <h3>Feedback Survey</h3>

        <form onSubmit={handleSubmit}>
          <label htmlFor="rating">Rating (1-5):</label>
          <input 
            type="number" 
            id="rating" 
            min="1" 
            max="5" 
            value={rating}
            onChange={(e) => setRating(e.target.value)}
            required
          />

          <label htmlFor="comments">Comments:</label>
          <textarea 
            id="comments" 
            placeholder="Share your feedback..."
            value={comments}
            onChange={(e) => setComments(e.target.value)}
          />

          <button type="submit" onClick={handleSubmit}>Submit Feedback</button>
        </form>
      </div>
    </div>
  );
}

export default SurveyModal;