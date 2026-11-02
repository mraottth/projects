import { createContext, useContext } from "react";
import type { Me } from "./api";

/** The signed-in person (GET /api/me), provided by App once they're approved. */
export const MeContext = createContext<Me | null>(null);

export function useMe(): Me | null {
  return useContext(MeContext);
}

export function useIsAdmin(): boolean {
  return !!useMe()?.is_admin;
}
