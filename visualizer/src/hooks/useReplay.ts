import { useEffect, useState } from "react";
import type { Replay } from "../types/replay";

export function useReplay(url = "/sample_replay.json") {
  const [replay, setReplay] = useState<Replay | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(url)
      .then((response) => {
        if (!response.ok) throw new Error(`Failed to load ${url}: ${response.status}`);
        return response.json();
      })
      .then((data) => {
        if (!cancelled) setReplay(data);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [url]);

  return { replay, error };
}

