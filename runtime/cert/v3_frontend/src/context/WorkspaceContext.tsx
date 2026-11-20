import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode
} from "react";
import {
  adaptAudit,
  adaptRun,
  adaptStages
} from "../api/adapters";
import type {
  ApiModelInfo,
  ApiRunSummary,
  ApiSystem
} from "../api/types";
import type {
  AuditResult,
  BlockSummary,
  RunInfo,
  StageRecord
} from "../types";

type Theme = "light" | "dark";

interface WorkspaceContextValue {
  block: string;
  setBlock: (block: string) => void;
  blockSummary: BlockSummary;
  blocks: BlockSummary[];
  run: ApiRunSummary;
  runInfo: RunInfo;
  runs: ApiRunSummary[];
  setRunId: (runId: string) => void;
  system: ApiSystem;
  stages: StageRecord[];
  audit: AuditResult;
  model: ApiModelInfo;
  theme: Theme;
  toggleTheme: () => void;
}

const WorkspaceContext =
  createContext<WorkspaceContextValue | null>(null);

interface WorkspaceProviderProps {
  model: ApiModelInfo;
  runs: ApiRunSummary[];
  run: ApiRunSummary;
  blocks: BlockSummary[];
  system: ApiSystem;
  setRunId: (runId: string) => void;
  children: ReactNode;
}

function initialTheme(): Theme {
  const saved = localStorage.getItem("v3-theme");
  if (saved === "light" || saved === "dark") return saved;
  return matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

export function WorkspaceProvider({
  model,
  runs,
  run,
  blocks,
  system,
  setRunId,
  children
}: WorkspaceProviderProps) {
  const [block, setBlockState] = useState(
    () => localStorage.getItem("v3-block") ?? blocks[0].id
  );
  const [theme, setTheme] = useState<Theme>(initialTheme);

  const validBlock = blocks.some((item) => item.id === block)
    ? block
    : blocks[0].id;
  const blockSummary =
    blocks.find((item) => item.id === validBlock) ?? blocks[0];

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("v3-theme", theme);
  }, [theme]);

  useEffect(() => {
    localStorage.setItem("v3-block", validBlock);
  }, [validBlock]);

  const value = useMemo<WorkspaceContextValue>(
    () => ({
      block: validBlock,
      setBlock: setBlockState,
      blockSummary,
      blocks,
      run,
      runInfo: adaptRun(run, system.optimizations),
      runs,
      setRunId,
      system,
      stages: adaptStages(system),
      audit: adaptAudit(system),
      model,
      theme,
      toggleTheme: () =>
        setTheme((current) =>
          current === "light" ? "dark" : "light"
        )
    }),
    [
      blocks,
      blockSummary,
      model,
      run,
      runs,
      setRunId,
      system,
      theme,
      validBlock
    ]
  );

  return (
    <WorkspaceContext.Provider value={value}>
      {children}
    </WorkspaceContext.Provider>
  );
}

export function useWorkspace(): WorkspaceContextValue {
  const value = useContext(WorkspaceContext);
  if (!value) {
    throw new Error("Workspace context is unavailable");
  }
  return value;
}
