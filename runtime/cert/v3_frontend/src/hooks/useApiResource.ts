import {
  useCallback,
  useEffect,
  useState,
  type DependencyList
} from "react";

interface ResourceState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

export function useApiResource<T>(
  loader: () => Promise<T>,
  dependencies: DependencyList
) {
  const [reloadKey, setReloadKey] = useState(0);
  const [state, setState] = useState<ResourceState<T>>({
    data: null,
    loading: true,
    error: null
  });

  useEffect(() => {
    let active = true;
    setState((current) => ({
      data: current.data,
      loading: true,
      error: null
    }));
    loader()
      .then((data) => {
        if (active) setState({ data, loading: false, error: null });
      })
      .catch((error: unknown) => {
        if (!active) return;
        setState({
          data: null,
          loading: false,
          error:
            error instanceof Error
              ? error.message
              : "接口请求失败"
        });
      });
    return () => {
      active = false;
    };
    // The caller owns dependency stability; the loader is intentionally
    // evaluated against the current request parameters.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...dependencies, reloadKey]);

  const reload = useCallback(() => {
    setReloadKey((value) => value + 1);
  }, []);

  return { ...state, reload };
}
