import {
  useCallback,
  useEffect,
  useRef,
  useState
} from "react";
import { api } from "../api/client";
import type {
  ApiBlockSummary,
  ApiModelInfo,
  ApiRunSummary,
  ApiSystem
} from "../api/types";

interface BootstrapState {
  model: ApiModelInfo | null;
  runs: ApiRunSummary[];
  run: ApiRunSummary | null;
  blocks: ApiBlockSummary[];
  system: ApiSystem | null;
  loading: boolean;
  error: string | null;
}

export function useBootstrap() {
  const [selectedRunId, setSelectedRunIdState] = useState(
    () => localStorage.getItem("v3-run") ?? ""
  );
  const selectedRunRef = useRef(selectedRunId);
  const [state, setState] = useState<BootstrapState>({
    model: null,
    runs: [],
    run: null,
    blocks: [],
    system: null,
    loading: true,
    error: null
  });

  const loadRuns = useCallback(async () => {
    setState((current) => ({
      ...current,
      loading: true,
      error: null
    }));
    try {
      const [model, runs] = await Promise.all([
        api.model(),
        api.runs()
      ]);
      if (!runs.length) {
        setState({
          model,
          runs: [],
          run: null,
          blocks: [],
          system: null,
          loading: false,
          error: null
        });
        return;
      }
      const nextRun =
        runs.find((item) => item.run_id === selectedRunRef.current) ??
        runs[0];
      setSelectedRunIdState(nextRun.run_id);
      selectedRunRef.current = nextRun.run_id;
      localStorage.setItem("v3-run", nextRun.run_id);
      const [blocks, system] = await Promise.all([
        api.blocks(nextRun.run_id),
        api.system(nextRun.run_id)
      ]);
      setState({
        model,
        runs,
        run: nextRun,
        blocks,
        system,
        loading: false,
        error: null
      });
    } catch (error) {
      setState((current) => ({
        ...current,
        loading: false,
        error:
          error instanceof Error
            ? error.message
            : "无法连接 V3 API"
      }));
    }
  }, []);

  useEffect(() => {
    void loadRuns();
  }, [loadRuns]);

  const setRunId = useCallback(
    async (runId: string) => {
      if (!state.runs.some((item) => item.run_id === runId)) {
        return;
      }
      setSelectedRunIdState(runId);
      selectedRunRef.current = runId;
      localStorage.setItem("v3-run", runId);
      setState((current) => ({
        ...current,
        loading: true,
        error: null
      }));
      try {
        const [run, blocks, system] = await Promise.all([
          api.run(runId),
          api.blocks(runId),
          api.system(runId)
        ]);
        setState((current) => ({
          ...current,
          run,
          blocks,
          system,
          loading: false,
          error: null
        }));
      } catch (error) {
        setState((current) => ({
          ...current,
          loading: false,
          error:
            error instanceof Error
              ? error.message
              : "运行切换失败"
        }));
      }
    },
    [state.runs]
  );

  return {
    ...state,
    selectedRunId,
    setRunId,
    refreshRuns: loadRuns
  };
}
