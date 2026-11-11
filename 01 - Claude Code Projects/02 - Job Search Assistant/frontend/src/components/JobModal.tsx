import { useEffect, useRef } from "react";
import { useNavigate, useParams } from "react-router-dom";
import JobDetailPage from "../pages/JobDetailPage";

/** The job page in a card over the page it was opened from (see JobLink). Closing goes Back. */
export function JobModal() {
  const { id } = useParams();
  const navigate = useNavigate();
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = ref.current;
    if (dialog && !dialog.open) dialog.showModal();
  }, []);

  const close = () => navigate(-1);

  return (
    <dialog
      ref={ref}
      className="modal job-modal"
      aria-label="Job details"
      onCancel={(e) => {
        e.preventDefault(); // Esc: go Back, which unmounts this
        close();
      }}
      // A click on the backdrop lands on the <dialog> element itself.
      onClick={(e) => {
        if (e.target === ref.current) close();
      }}
    >
      <button className="modal-close job-modal-close" onClick={close} aria-label="Close" title="Close (Esc)">
        ×
      </button>
      <div className="job-modal-body">
        <JobDetailPage jobId={Number(id)} />
      </div>
    </dialog>
  );
}
