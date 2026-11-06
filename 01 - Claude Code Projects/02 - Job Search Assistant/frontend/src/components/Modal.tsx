import { ReactNode, useEffect, useRef } from "react";

// A small dialog on the native <dialog> element: showModal() gives focus trapping, Esc to close,
// and a backdrop without any extra code. It stays mounted while closed, so whatever's inside
// (e.g. discovery results) survives closing and reopening.
export function Modal({
  open,
  onClose,
  title,
  children,
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      className="modal"
      onClose={onClose}
      // A click on the backdrop lands on the <dialog> element itself; clicks inside land on children.
      onClick={(e) => {
        if (e.target === ref.current) onClose();
      }}
    >
      <div className="modal-head">
        <h2>{title}</h2>
        <button className="modal-close" onClick={onClose} aria-label="Close">
          ×
        </button>
      </div>
      <div className="modal-body">{children}</div>
    </dialog>
  );
}
