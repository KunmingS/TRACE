import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { API_URL } from '../../../config';
import {
    DEFAULT_PATH_ADAPTER,
    PathAdapter,
    PlatformInfo,
    createPathAdapter,
} from './pathAdapter';

interface PlatformContextValue {
    adapter: PathAdapter;
    /** True until the first /api/platform fetch settles. */
    isLoading: boolean;
}

const PlatformContext = createContext<PlatformContextValue>({
    adapter: DEFAULT_PATH_ADAPTER,
    isLoading: true,
});

interface PlatformProviderProps {
    children: React.ReactNode;
    /** Test escape hatch: skip the fetch and use this PlatformInfo directly. */
    override?: PlatformInfo;
}

export const PlatformProvider: React.FC<PlatformProviderProps> = ({ children, override }) => {
    const [info, setInfo] = useState<PlatformInfo | null>(override || null);
    const [isLoading, setIsLoading] = useState(!override);

    useEffect(() => {
        if (override) return undefined;
        let cancelled = false;
        (async () => {
            try {
                const res = await fetch(`${API_URL}/api/platform`);
                if (!res.ok) throw new Error(`platform endpoint ${res.status}`);
                const data = await res.json();
                if (cancelled) return;
                setInfo({
                    os: data.os === 'windows' ? 'windows' : 'posix',
                    sep: data.sep || (data.os === 'windows' ? '\\' : '/'),
                    home: data.home || (data.os === 'windows' ? 'C:\\' : '/'),
                    roots: Array.isArray(data.roots) && data.roots.length > 0
                        ? data.roots
                        : DEFAULT_PATH_ADAPTER.roots,
                });
            } catch {
                // Stay on POSIX defaults; older backends without /api/platform
                // still expose /api/home-dir, which the adapter doesn't need
                // to consult (its `home` is set from /api/platform only).
            } finally {
                if (!cancelled) setIsLoading(false);
            }
        })();
        return () => { cancelled = true; };
    }, [override]);

    const value = useMemo<PlatformContextValue>(() => ({
        adapter: info ? createPathAdapter(info) : DEFAULT_PATH_ADAPTER,
        isLoading,
    }), [info, isLoading]);

    return <PlatformContext.Provider value={value}>{children}</PlatformContext.Provider>;
};

export const usePathAdapter = (): PathAdapter => useContext(PlatformContext).adapter;
export const usePlatformLoading = (): boolean => useContext(PlatformContext).isLoading;
