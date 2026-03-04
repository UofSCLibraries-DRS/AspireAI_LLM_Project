import GenericModal from './generic-modal';
import './modals.css'

function PolicyModal({ show, onClose }: { show: boolean; onClose: () => void }) {

    return(
        <GenericModal show={show} onClose={onClose} title="Privacy Policy">

        <h4>Data Collection and Usage:</h4>
        <p>To improve our services, we collect:</p>
        <ul>
            <li>All chatbot interactions and conversations.</li>

            <li>User feedback and ratings.</li>

            <li>General usage statistics (e.g., frequency of use, feature engagement).</li>
        </ul>
        <p>We do <strong>NOT</strong> collect:</p>
        <ul>
            <li>Any personally identifying information unless explicitly provided by you through chatbot interactions or submitted forms.</li>
        </ul>
        <p>By using this chatbot, you agree to the collection of the data mentioned above for the purposes of service improvement and troubleshooting.</p>
        </GenericModal>
    );
}

export default PolicyModal;