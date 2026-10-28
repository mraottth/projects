// Jobby's face: half house-elf, half robot. Big droopy elf ears and a little nose on a rounded
// robot head with an antenna, a panel seam, and one ordinary eye next to a glowing LED eye.
// An original design (not any existing character) -- drawn inline so it scales cleanly from the
// nav tab (~18px) up to the chat header, and needs no image request.
export function JobbyLogo({ size = 32, title = "Jobby" }: { size?: number; title?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      role="img"
      aria-label={title}
      className="jobby-logo"
    >
      <title>{title}</title>
      {/* antenna */}
      <line x1="32" y1="15" x2="32" y2="7" stroke="#3d4a57" strokeWidth="2" strokeLinecap="round" />
      <circle cx="32" cy="5.5" r="3.2" fill="var(--accent)" />
      <circle cx="31" cy="4.6" r="1" fill="#d6e8ff" />
      {/* ears */}
      <path d="M17 31 C9 28 3 23 1 14 C8 18 14 21 19 24 Z" fill="#a9cfae" stroke="#3d4a57" strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M47 31 C55 28 61 23 63 14 C56 18 50 21 45 24 Z" fill="#a9cfae" stroke="#3d4a57" strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M15 27.5 C10 25.5 6.5 22.5 5 18.5 C9 20.5 13 22.5 16.5 24.5 Z" fill="#e7b9b0" opacity="0.8" />
      <path d="M49 27.5 C54 25.5 57.5 22.5 59 18.5 C55 20.5 51 22.5 47.5 24.5 Z" fill="#e7b9b0" opacity="0.8" />
      {/* head */}
      <rect x="14" y="14" width="36" height="35" rx="13" fill="#e3e9f0" stroke="#3d4a57" strokeWidth="2" />
      <path d="M22 20.5 H42" stroke="#b7c3cf" strokeWidth="1.4" strokeLinecap="round" />
      <circle cx="19.5" cy="43" r="1.1" fill="#b7c3cf" />
      <circle cx="44.5" cy="43" r="1.1" fill="#b7c3cf" />
      {/* eyes: one ordinary, one LED */}
      <circle cx="25" cy="30" r="6" fill="#fff" stroke="#3d4a57" strokeWidth="1.4" />
      <circle cx="26" cy="31" r="3.3" fill="#1d2a36" />
      <circle cx="27.2" cy="29.6" r="1.1" fill="#fff" />
      <circle cx="39" cy="30" r="6" fill="#13263d" stroke="#3d4a57" strokeWidth="1.4" />
      <circle cx="39" cy="30" r="3.6" fill="var(--accent)" />
      <circle cx="39" cy="30" r="1.5" fill="#cfe5ff" />
      {/* nose, cheeks, smile */}
      <path d="M32 33 L34.2 38.2 L30.4 38.2 Z" fill="#c7d2dc" stroke="#3d4a57" strokeWidth="1" strokeLinejoin="round" />
      <circle cx="20.5" cy="38.5" r="2.4" fill="#f2b8b8" opacity="0.65" />
      <circle cx="43.5" cy="38.5" r="2.4" fill="#f2b8b8" opacity="0.65" />
      <path d="M26.5 42 Q32 46 37.5 42" fill="none" stroke="#3d4a57" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}
