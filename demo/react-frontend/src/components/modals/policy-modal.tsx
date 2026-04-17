import { useEffect } from 'react';
import { closeModal, openModal } from '../../services/modals';
import './modals.css'

function PolicyModal({ show, onClose }: { show: boolean; onClose: () => void }) {
    const modalId = "policyModal"

    // Sync show prop 
    useEffect(() => {
        if (show) {
            openModal(modalId)
        } else {
            closeModal(modalId)
        }
    }, [show]);

    // Esc to exit
    useEffect(() => {
        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape'  && show) {
              onClose()
            }
        };
        if(show) {
            document.addEventListener('keydown', onKeyDown);
        }
        return () => document.removeEventListener('keydown', onKeyDown);
    }, [show, onClose])

    return(
        <div id={modalId} className="modal">
            <div className="modal-content">
            <button className="close-modal" onClick={onClose}>×</button>
            <h3>Privacy Policy</h3>
            <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore
                magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo
                consequat.</p>

            <h4>Data Collection and Usage:</h4>
            <p>We collect:</p>
            <ul>
                <li>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et
                dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea
                commodo consequat.</li>
            </ul>
            <p>We do <strong>NOT</strong> collect:</p>
            <ul>
                <li>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et
                dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea
                commodo consequat.
                </li>
            </ul>

            <p>By using this chatbot, you agree Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
                incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
                nisi ut aliquip ex ea commodo consequat.</p>
            </div>
        </div>
    );
}

export default PolicyModal;