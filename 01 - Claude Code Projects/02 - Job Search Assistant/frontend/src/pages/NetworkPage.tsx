import { NetworkSection } from "../components/Network";

/**
 * My Network (asked 2026-11-16): your LinkedIn connections, who to reach out to, and organizations
 * you know people at. It used to be a card on Manage Sources; adding a suggested organization still
 * creates a source there.
 */
export default function NetworkPage() {
  return (
    <div>
      <h1>My Network</h1>
      <p className="lead-left">
        The people you know who could help with the jobs you want, from your
        LinkedIn connections.
      </p>
      <NetworkSection onSourcesChanged={() => undefined} />
    </div>
  );
}
