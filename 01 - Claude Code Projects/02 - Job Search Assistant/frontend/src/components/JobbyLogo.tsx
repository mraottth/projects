import avatarUrl from "../assets/jobby-avatar.png";
import fullUrl from "../assets/jobby-full.png";

// Jobby's artwork (supplied by the user; backgrounds cut out to transparent PNGs). Imported so
// Vite bundles them under /assets/ -- the backend's SPA fallback answers every other path with
// index.html, so a file in public/ wouldn't be served as an image.
//
// "avatar" is a tight crop of the head for small spots (nav tab, chat avatars, page header);
// "full" is Jobby at the laptop, for the empty-conversation welcome.
export function JobbyLogo({
  size = 32,
  title = "Jobby",
  variant = "avatar",
}: {
  size?: number;
  title?: string;
  variant?: "avatar" | "full";
}) {
  return (
    <img
      src={variant === "full" ? fullUrl : avatarUrl}
      width={size}
      height={size}
      alt={title}
      className="jobby-logo"
      draggable={false}
    />
  );
}
