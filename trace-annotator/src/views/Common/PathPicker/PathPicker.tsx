import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import './PathPicker.scss';
import { API_URL } from '../../../config';

interface PathPickerProps {
    value: string;
    onChange: (value: string) => void;
    onSubmit?: () => void;
    placeholder?: string;
    mode?: 'dir' | 'file' | 'multiPair';
    extensions?: string;
    disabled?: boolean;
    storageKey?: string;
    // When set in `dir` mode, files matching these extensions are listed
    // in the dropdown but rendered as disabled previews — folders stay
    // the only selectable thing. Useful for "browse the folder you're
    // about to load" without exposing a per-file pick.
    previewExtensions?: string;
    // multiPair mode: parent owns the set of pair stems chosen from the
    // listed video+CSV groups. The picker also emits canonical
    // "video=csv" specs for callers that need exact pair paths.
    selectedStems?: string[];
    onSelectedStemsChange?: (stems: string[]) => void;
    onSelectedPairsChange?: (pairs: string[]) => void;
    // multiPair mode: per-stem CSV pick. Lets the user choose which CSV
    // to bundle when a group has more than one. Missing entries fall back
    // to the canonical CSV (group.csvs[0]).
    selectedCsvByStem?: Record<string, string>;
    onSelectedCsvByStemChange?: (csvByStem: Record<string, string>) => void;
    // multiPair mode: when true, only groups with a source video AND a CSV
    // are selectable. When false (e.g. inference), any group with a video
    // is selectable; CSV is ignored.
    requireSourceVideo?: boolean;
    // multiPair + multiFolder: selection persists across folder navigation
    // and pair specs become fully-qualified absolute paths
    // ("<absVideo>=<absCsv>"). Manifest groups by source folder and lets
    // users swap CSVs inline without re-navigating to the source folder.
    // Requires `selectedPairs` + `onSelectedPairsChange`; the per-folder
    // `selectedStems`/`csvByStem` props are ignored in this mode.
    multiFolder?: boolean;
    selectedPairs?: string[];
}

const PAIR_PREVIEW_EXTS = '.csv,.mp4,.avi,.mov,.mkv,.webm';

interface DirEntry {
    name: string;
    type: 'dir' | 'file';
}

const VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];

type FileKind = 'video' | 'csv';
type VideoVariant = 'source' | 'remux' | 'h264';

interface RelatedFile {
    name: string;
    kind: FileKind;
    variant?: VideoVariant;
}

interface RelatedFileGroup {
    key: string;
    displayName: string;
    videos: RelatedFile[];
    csvs: RelatedFile[];
    firstIndex: number;
}

function classifyFile(name: string): FileKind {
    return name.toLowerCase().endsWith('.csv') ? 'csv' : 'video';
}

function getFileExtension(name: string): string {
    const dot = name.lastIndexOf('.');
    return dot >= 0 ? name.substring(dot).toLowerCase() : '';
}

function stripKnownVideoExtension(name: string): string {
    const ext = getFileExtension(name);
    return VIDEO_EXTENSIONS.includes(ext) ? name.slice(0, -ext.length) : name;
}

function parseVideoFile(name: string): { stem: string; variant: VideoVariant } | null {
    const lowerName = name.toLowerCase();
    if (lowerName.endsWith('.remux.mp4')) {
        return {
            stem: stripKnownVideoExtension(name.slice(0, -'.remux.mp4'.length)),
            variant: 'remux',
        };
    }
    if (lowerName.endsWith('.h264.mp4')) {
        return {
            stem: stripKnownVideoExtension(name.slice(0, -'.h264.mp4'.length)),
            variant: 'h264',
        };
    }

    const ext = getFileExtension(name);
    if (!VIDEO_EXTENSIONS.includes(ext)) return null;
    return {
        stem: name.slice(0, -ext.length),
        variant: 'source',
    };
}

function getCsvStem(name: string): string | null {
    return name.toLowerCase().endsWith('.csv') ? name.slice(0, -4) : null;
}

function joinPath(dir: string, name: string): string {
    if (!dir) return name;
    if (dir === '/') return '/' + name;
    return dir.replace(/\/+$/, '') + '/' + name;
}

function dirOfPath(p: string): string {
    const idx = p.lastIndexOf('/');
    if (idx <= 0) return '/';
    return p.substring(0, idx);
}

function basenameOfPath(p: string): string {
    const idx = p.lastIndexOf('/');
    return idx >= 0 ? p.substring(idx + 1) : p;
}

function stemKeyFromVideoBasename(basename: string): string {
    // Mirrors parseVideoFile() — collapse .remux.mp4 / .h264.mp4 onto the
    // canonical stem so a remux'd CSV still groups with its source video.
    const lower = basename.toLowerCase();
    if (lower.endsWith('.remux.mp4')) return basename.slice(0, -'.remux.mp4'.length);
    if (lower.endsWith('.h264.mp4')) return basename.slice(0, -'.h264.mp4'.length);
    const ext = getFileExtension(basename);
    if (VIDEO_EXTENSIONS.includes(ext)) return basename.slice(0, -ext.length);
    return basename;
}

interface ParsedPair {
    videoPath: string;
    csvPath: string;
    folder: string;
    stemKey: string;
}

function parsePairSpec(spec: string): ParsedPair | null {
    const eq = spec.indexOf('=');
    if (eq < 0) return null;
    const videoPath = spec.substring(0, eq);
    const csvPath = spec.substring(eq + 1);
    if (!videoPath || !csvPath) return null;
    return {
        videoPath,
        csvPath,
        folder: dirOfPath(videoPath),
        stemKey: stemKeyFromVideoBasename(basenameOfPath(videoPath)),
    };
}

function buildRelatedFileGroups(entries: DirEntry[]): RelatedFileGroup[] {
    const fileEntries = entries
        .map((entry, index) => ({ entry, index }))
        .filter(({ entry }) => entry.type === 'file');
    const parsedVideos = fileEntries
        .map(({ entry, index }) => ({ entry, index, parsed: parseVideoFile(entry.name) }))
        .filter((item): item is { entry: DirEntry; index: number; parsed: { stem: string; variant: VideoVariant } } => !!item.parsed);
    const videoStems = Array.from(new Set(parsedVideos.map(({ parsed }) => parsed.stem)))
        .sort((a, b) => b.length - a.length);
    const groups = new Map<string, RelatedFileGroup>();

    const ensureGroup = (key: string, firstIndex: number): RelatedFileGroup => {
        const existing = groups.get(key);
        if (existing) {
            existing.firstIndex = Math.min(existing.firstIndex, firstIndex);
            return existing;
        }
        const created = {
            key,
            displayName: key,
            videos: [],
            csvs: [],
            firstIndex,
        };
        groups.set(key, created);
        return created;
    };

    parsedVideos.forEach(({ entry, index, parsed }) => {
        ensureGroup(parsed.stem, index).videos.push({
            name: entry.name,
            kind: 'video',
            variant: parsed.variant,
        });
    });

    fileEntries.forEach(({ entry, index }) => {
        const csvStem = getCsvStem(entry.name);
        if (!csvStem) return;
        const matchedVideoStem = videoStems.find((stem) =>
            csvStem === stem || csvStem.startsWith(`${stem}_`)
        );
        const groupKey = matchedVideoStem || csvStem;
        ensureGroup(groupKey, index).csvs.push({
            name: entry.name,
            kind: 'csv',
        });
    });

    const videoOrder: Record<VideoVariant, number> = { source: 0, remux: 1, h264: 2 };
    return Array.from(groups.values())
        .map((group) => ({
            ...group,
            videos: group.videos.sort((a, b) =>
                videoOrder[a.variant || 'source'] - videoOrder[b.variant || 'source']
                || a.name.localeCompare(b.name)
            ),
            csvs: group.csvs.sort((a, b) => {
                const canonical = `${group.key}.csv`;
                if (a.name === canonical) return -1;
                if (b.name === canonical) return 1;
                return a.name.localeCompare(b.name);
            }),
        }))
        .sort((a, b) => a.firstIndex - b.firstIndex || a.key.localeCompare(b.key));
}

const MAX_RECENT = 5;
const RECENT_PREFIX = 'trace:recent:';

function getRecentPaths(key: string): string[] {
    try {
        const stored = localStorage.getItem(RECENT_PREFIX + key);
        return stored ? JSON.parse(stored) : [];
    } catch {
        return [];
    }
}

function saveRecentPath(key: string, path: string) {
    const clean = path.replace(/\/+$/, '') || '/';
    const existing = getRecentPaths(key).filter((p) => p !== clean);
    const updated = [clean, ...existing].slice(0, MAX_RECENT);
    localStorage.setItem(RECENT_PREFIX + key, JSON.stringify(updated));
}

function splitBreadcrumbs(path: string): { segment: string; fullPath: string }[] {
    if (!path || path === '/') return [{ segment: '/', fullPath: '/' }];
    const parts = path.replace(/\/+$/, '').split('/').filter(Boolean);
    const crumbs = [{ segment: '/', fullPath: '/' }];
    let acc = '';
    for (const part of parts) {
        acc += '/' + part;
        crumbs.push({ segment: part, fullPath: acc });
    }
    return crumbs;
}

function getParentDir(dir: string): string {
    const trimmed = dir.replace(/\/+$/, '');
    const lastSlash = trimmed.lastIndexOf('/');
    if (lastSlash <= 0) return '/';
    return trimmed.substring(0, lastSlash);
}

const PathPicker: React.FC<PathPickerProps> = ({
    value,
    onChange,
    onSubmit,
    placeholder = '/path/to/folder',
    mode = 'dir',
    extensions,
    disabled = false,
    storageKey = 'default',
    previewExtensions,
    selectedStems,
    onSelectedStemsChange,
    onSelectedPairsChange,
    selectedCsvByStem,
    onSelectedCsvByStemChange,
    requireSourceVideo = false,
    multiFolder = false,
    selectedPairs,
}) => {
    // `dir` mode with a `previewExtensions` prop also fetches files from
    // the backend — but they're rendered as disabled previews, not picks.
    // `multiPair` likewise pulls files in but uses the canonical pair-ext
    // set so the same Related Sets builder works.
    const isMultiPair = mode === 'multiPair';
    const isMultiFolder = isMultiPair && multiFolder;
    const filePreviewMode = (mode === 'dir' && !!previewExtensions) || isMultiPair;
    const showsFiles = mode === 'file' || filePreviewMode;
    const effectiveExtensions = mode === 'file'
        ? extensions
        : (isMultiPair ? PAIR_PREVIEW_EXTS : previewExtensions);
    const stemSet = useMemo(() => new Set(selectedStems || []), [selectedStems]);
    const emitStems = useCallback((next: string[]) => {
        onSelectedStemsChange?.(Array.from(new Set(next)));
    }, [onSelectedStemsChange]);

    // multiFolder: parse the pair specs once. Selection state for the
    // currently browsed folder is derived from this; rows in the manifest
    // are derived from this; there's no separate stem book-keeping.
    const parsedPairs = useMemo<ParsedPair[]>(() => {
        if (!isMultiFolder) return [];
        return (selectedPairs || [])
            .map(parsePairSpec)
            .filter((p): p is ParsedPair => !!p);
    }, [isMultiFolder, selectedPairs]);
    const [entries, setEntries] = useState<DirEntry[]>([]);
    const [currentDir, setCurrentDir] = useState('');
    const [isOpen, setIsOpen] = useState(false);
    const [selectedIndex, setSelectedIndex] = useState(-1);
    const [isLoading, setIsLoading] = useState(false);
    const [homeDir, setHomeDir] = useState<string | null>(null);
    const [siblingDropdown, setSiblingDropdown] = useState<{
        crumbIndex: number;
        items: { path: string; type: 'dir' | 'file' }[];
    } | null>(null);

    const inputRef = useRef<HTMLInputElement>(null);
    const wrapperRef = useRef<HTMLDivElement>(null);
    const listRef = useRef<HTMLDivElement>(null);
    const debounceRef = useRef<number | null>(null);

    // Fetch home directory once
    useEffect(() => {
        fetch(`${API_URL}/api/home-dir`)
            .then((res) => res.json())
            .then((data) => setHomeDir(data.home || null))
            .catch(() => {});
    }, []);

    const resolveTilde = useCallback((path: string): string => {
        if (path.startsWith('~') && homeDir) {
            return homeDir + path.slice(1);
        }
        return path;
    }, [homeDir]);

    const fetchListing = useCallback(async (dir: string) => {
        setIsLoading(true);
        try {
            const params = new URLSearchParams({ path: dir });
            if (showsFiles && effectiveExtensions) {
                params.set('extensions', effectiveExtensions);
            }
            const res = await fetch(`${API_URL}/api/ls?${params}`);
            if (res.ok) {
                const data = await res.json();
                const fetched: DirEntry[] = data.entries || [];
                setEntries(fetched);
                if (data.resolved) setCurrentDir(data.resolved);
                // Populate the multi-folder popover cache atomically with
                // the fresh fetch. Doing this from a useEffect on
                // (currentDir, entries) is racy because currentDir updates
                // synchronously while entries lags one render — the effect
                // can fire after a setCurrentDir(newDir) but before
                // setEntries(newEntries) lands, locking in stale data.
                if (mode === 'multiPair' && multiFolder) {
                    const cacheKey = data.resolved || dir;
                    setFolderEntriesCache((prev) => ({ ...prev, [cacheKey]: fetched }));
                }
            }
        } catch {
            setEntries([]);
        }
        setIsLoading(false);
    }, [showsFiles, effectiveExtensions, mode, multiFolder]);

    const navigateToDir = useCallback((dir: string) => {
        const dirWithSlash = dir.endsWith('/') ? dir : dir + '/';
        onChange(dirWithSlash);
        setCurrentDir(dir);
        setSelectedIndex(-1);
        saveRecentPath(storageKey, dir);
        fetchListing(dir);
        inputRef.current?.focus();
    }, [onChange, storageKey, fetchListing]);

    const selectFile = useCallback((filePath: string) => {
        onChange(filePath);
        saveRecentPath(storageKey, filePath);
        setIsOpen(false);
    }, [onChange, storageKey]);

    const handleFocus = () => {
        setIsOpen(true);
        setSiblingDropdown(null);
        if (!value) {
            // Start at home directory
            const startDir = homeDir || '/';
            onChange(startDir + '/');
            setCurrentDir(startDir);
            fetchListing(startDir);
        } else {
            const resolved = resolveTilde(value);
            const dir = resolved.endsWith('/') ? resolved.replace(/\/+$/, '') || '/' : getParentDir(resolved);
            if (dir !== currentDir) {
                setCurrentDir(dir);
                fetchListing(dir);
            }
        }
    };

    const handleChange = (rawValue: string) => {
        let val = rawValue;
        if (val && !val.startsWith('/') && !val.startsWith('~')) val = '/' + val;
        onChange(val);
        setSiblingDropdown(null);

        if (debounceRef.current) clearTimeout(debounceRef.current);
        debounceRef.current = window.setTimeout(() => {
            const resolved = resolveTilde(val);
            if (resolved.endsWith('/')) {
                const dir = resolved.replace(/\/+$/, '') || '/';
                setCurrentDir(dir);
                fetchListing(dir);
            } else {
                // Typing a partial name — show parent dir contents (filter happens in render)
                const parent = getParentDir(resolved);
                if (parent !== currentDir) {
                    setCurrentDir(parent);
                    fetchListing(parent);
                }
            }
        }, 150);
    };

    // Compute filtered entries based on what's typed after the last /
    const resolvedValue = resolveTilde(value);
    const typedFilter = resolvedValue.endsWith('/') ? '' : (resolvedValue.split('/').pop() || '').toLowerCase();

    // Build display list: parent entry + filtered entries
    const filteredEntries = typedFilter
        ? entries.filter((e) => e.name.toLowerCase().startsWith(typedFilter))
        : entries;
    const showParent = currentDir !== '/' && !typedFilter;
    const selectableEntries = filePreviewMode
        ? filteredEntries.filter((entry) => entry.type === 'dir')
        : filteredEntries;
    const relatedFileGroups = filePreviewMode ? buildRelatedFileGroups(filteredEntries) : [];
    const relatedFileCount = relatedFileGroups.reduce(
        (sum, group) => sum + group.videos.length + group.csvs.length,
        0,
    );
    const lastEmittedPairsRef = useRef('');
    const selectedPairSpecs = useMemo(() => {
        if (!isMultiPair) return [];
        const groupsByStem = new Map<string, RelatedFileGroup>();
        relatedFileGroups.forEach((group) => groupsByStem.set(group.key, group));
        return (selectedStems || [])
            .map((stem) => {
                const group = groupsByStem.get(stem);
                const video = group?.videos.find((file) => file.variant === 'source') || group?.videos[0];
                const preferredCsvName = selectedCsvByStem?.[stem];
                const csv = (preferredCsvName && group?.csvs.find((c) => c.name === preferredCsvName)) || group?.csvs[0];
                return video && csv ? `${video.name}=${csv.name}` : null;
            })
            .filter((pair): pair is string => !!pair);
    }, [isMultiPair, relatedFileGroups, selectedStems, selectedCsvByStem]);

    useEffect(() => {
        // In multiFolder mode the parent already owns selectedPairs as the
        // source of truth — re-emitting from current-folder data would clobber
        // selections that live in other folders.
        if (!isMultiPair || isMultiFolder || !onSelectedPairsChange) return;
        const signature = selectedPairSpecs.join('\n');
        if (signature === lastEmittedPairsRef.current) return;
        lastEmittedPairsRef.current = signature;
        onSelectedPairsChange(selectedPairSpecs);
    }, [isMultiPair, isMultiFolder, onSelectedPairsChange, selectedPairSpecs]);

    // Auto-select the first matching entry when the filter changes (VS Code style)
    useEffect(() => {
        if (!isOpen) return;
        if (typedFilter && selectableEntries.length > 0) {
            setSelectedIndex(showParent ? 1 : 0);
        } else if (!typedFilter) {
            setSelectedIndex(-1);
        }
    }, [typedFilter, selectableEntries.length, isOpen, showParent]);

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (!isOpen) {
            if (e.key === 'Enter' && value && onSubmit) onSubmit();
            return;
        }

        const totalItems = (showParent ? 1 : 0) + selectableEntries.length;
        if (totalItems === 0) {
            if (e.key === 'Enter' && value && onSubmit) onSubmit();
            if (e.key === 'Escape') setIsOpen(false);
            return;
        }

        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                setSelectedIndex((prev) => (prev < totalItems - 1 ? prev + 1 : 0));
                break;
            case 'ArrowUp':
                e.preventDefault();
                setSelectedIndex((prev) => (prev > 0 ? prev - 1 : totalItems - 1));
                break;
            case 'Enter': {
                e.preventDefault();
                if (selectedIndex < 0) {
                    if (value && onSubmit) onSubmit();
                    return;
                }
                const isParentSelected = showParent && selectedIndex === 0;
                if (isParentSelected) {
                    navigateToDir(getParentDir(currentDir));
                } else {
                    const entryIndex = selectedIndex - (showParent ? 1 : 0);
                    const entry = selectableEntries[entryIndex];
                    if (entry) {
                        const fullPath = currentDir === '/' ? '/' + entry.name : currentDir + '/' + entry.name;
                        if (entry.type === 'dir') {
                            navigateToDir(fullPath);
                        } else if (mode === 'file') {
                            selectFile(fullPath);
                        }
                    }
                }
                break;
            }
            case 'Tab': {
                if (selectedIndex >= 0) {
                    e.preventDefault();
                    const isParentSelected = showParent && selectedIndex === 0;
                    if (isParentSelected) {
                        navigateToDir(getParentDir(currentDir));
                    } else {
                        const entryIndex = selectedIndex - (showParent ? 1 : 0);
                        const entry = selectableEntries[entryIndex];
                        if (entry?.type === 'dir') {
                            navigateToDir(currentDir === '/' ? '/' + entry.name : currentDir + '/' + entry.name);
                        }
                    }
                } else if (selectableEntries.length === 1 && selectableEntries[0].type === 'dir') {
                    e.preventDefault();
                    const entry = selectableEntries[0];
                    navigateToDir(currentDir === '/' ? '/' + entry.name : currentDir + '/' + entry.name);
                }
                break;
            }
            case 'Escape':
                setIsOpen(false);
                setSiblingDropdown(null);
                break;
            case 'Backspace':
                // If input ends with / and cursor is at end, go to parent
                if (value.endsWith('/') && inputRef.current?.selectionStart === value.length) {
                    e.preventDefault();
                    navigateToDir(getParentDir(currentDir));
                }
                break;
        }
    };

    // Breadcrumb sibling navigation (kept from original)
    const handleBreadcrumbClick = async (crumbIndex: number, fullPath: string) => {
        if (siblingDropdown?.crumbIndex === crumbIndex) {
            setSiblingDropdown(null);
            return;
        }
        const parent = fullPath === '/' ? '/' : fullPath.replace(/\/[^/]*$/, '') || '/';
        try {
            const res = await fetch(`${API_URL}/api/dirs?prefix=${encodeURIComponent(parent + '/')}`);
            if (res.ok) {
                const data = await res.json();
                const items = (data.dirs || []).map((d: string) => ({ path: d, type: 'dir' as const }));
                setSiblingDropdown({ crumbIndex, items });
            }
        } catch {
            // ignore
        }
    };

    const selectSibling = (item: { path: string }) => {
        navigateToDir(item.path);
        setSiblingDropdown(null);
    };

    const getFolderName = (fullPath: string): string => {
        const parts = fullPath.replace(/\/+$/, '').split('/');
        return parts[parts.length - 1] || '/';
    };

    // Click outside handler
    useEffect(() => {
        const handleClickOutside = (e: MouseEvent) => {
            const target = e.target as Node;
            if (wrapperRef.current && !wrapperRef.current.contains(target)) {
                setIsOpen(false);
                setSiblingDropdown(null);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    // Auto-scroll selected entry
    useEffect(() => {
        if (selectedIndex >= 0 && listRef.current) {
            const items = listRef.current.querySelectorAll('.DirEntry');
            items[selectedIndex]?.scrollIntoView({ block: 'nearest' });
        }
    }, [selectedIndex]);

    const renderEntryName = (name: string) => (
        typedFilter && name.toLowerCase().startsWith(typedFilter) ? (
            <><span className='MatchHighlight'>{name.slice(0, typedFilter.length)}</span>{name.slice(typedFilter.length)}</>
        ) : name
    );

    const getGroupStatus = (group: RelatedFileGroup): string => {
        const sourceCount = group.videos.filter((file) => file.variant === 'source').length;
        const derivedCount = group.videos.length - sourceCount;
        if (sourceCount > 0 && group.csvs.length > 0) return 'paired';
        if (sourceCount > 0 && derivedCount > 0) return 'video + copy';
        if (sourceCount > 0) return 'video only';
        if (derivedCount > 0 && group.csvs.length > 0) return 'copy + csv';
        if (derivedCount > 0) return 'copy only';
        return 'csv only';
    };

    // multiPair: rules for whether the group can be selected as a pair.
    const isGroupSelectable = (group: RelatedFileGroup): boolean => {
        const hasAnyVideo = group.videos.length > 0;
        const hasSource = group.videos.some((file) => file.variant === 'source');
        const hasCsv = group.csvs.length > 0;
        if (requireSourceVideo) return hasSource && hasCsv;
        return hasAnyVideo;
    };

    // multiFolder: which group keys (bare stems) in the *current folder* have
    // a pair selected, mapped to that pair spec. Drives the in-folder
    // checkbox state and the pair-spec rewrite for CSV swaps.
    const currentFolderPairsByStem = useMemo(() => {
        const map = new Map<string, ParsedPair>();
        if (!isMultiFolder) return map;
        for (const parsed of parsedPairs) {
            if (parsed.folder === currentDir) {
                map.set(parsed.stemKey, parsed);
            }
        }
        return map;
    }, [isMultiFolder, parsedPairs, currentDir]);

    const setPairs = useCallback((updater: (prev: string[]) => string[]) => {
        if (!isMultiFolder || !onSelectedPairsChange) return;
        onSelectedPairsChange(updater(selectedPairs || []));
    }, [isMultiFolder, onSelectedPairsChange, selectedPairs]);

    const dropCsvForStems = useCallback((stems: string[]) => {
        if (!onSelectedCsvByStemChange || !selectedCsvByStem) return;
        let changed = false;
        const next = { ...selectedCsvByStem };
        for (const s of stems) {
            if (next[s] !== undefined) {
                delete next[s];
                changed = true;
            }
        }
        if (changed) onSelectedCsvByStemChange(next);
    }, [onSelectedCsvByStemChange, selectedCsvByStem]);

    const toggleStem = useCallback((stem: string) => {
        if (!isMultiPair) return;
        if (isMultiFolder) {
            const existing = currentFolderPairsByStem.get(stem);
            if (existing) {
                const spec = `${existing.videoPath}=${existing.csvPath}`;
                setPairs((prev) => prev.filter((p) => p !== spec));
                return;
            }
            // Add canonical pair: source video (or first available) + canonical CSV
            const group = relatedFileGroups.find((g) => g.key === stem);
            if (!group) return;
            const video = group.videos.find((f) => f.variant === 'source') || group.videos[0];
            const csv = group.csvs[0];
            if (!video || !csv) return;
            const spec = `${joinPath(currentDir, video.name)}=${joinPath(currentDir, csv.name)}`;
            setPairs((prev) => prev.includes(spec) ? prev : [...prev, spec]);
            return;
        }
        if (stemSet.has(stem)) {
            emitStems((selectedStems || []).filter((s) => s !== stem));
            dropCsvForStems([stem]);
        } else {
            emitStems([...(selectedStems || []), stem]);
        }
    }, [isMultiPair, isMultiFolder, currentFolderPairsByStem, relatedFileGroups, currentDir, setPairs, stemSet, selectedStems, emitStems, dropCsvForStems]);

    const setStemCsv = useCallback((stem: string, csvName: string) => {
        if (!isMultiPair) return;
        if (isMultiFolder) {
            const existing = currentFolderPairsByStem.get(stem);
            const newCsvPath = joinPath(currentDir, csvName);
            if (existing) {
                if (existing.csvPath === newCsvPath) return;
                const oldSpec = `${existing.videoPath}=${existing.csvPath}`;
                const newSpec = `${existing.videoPath}=${newCsvPath}`;
                setPairs((prev) => prev.map((p) => p === oldSpec ? newSpec : p));
            } else {
                // Stem isn't selected yet; pick it with this CSV
                const group = relatedFileGroups.find((g) => g.key === stem);
                if (!group) return;
                const video = group.videos.find((f) => f.variant === 'source') || group.videos[0];
                if (!video) return;
                const spec = `${joinPath(currentDir, video.name)}=${newCsvPath}`;
                setPairs((prev) => prev.includes(spec) ? prev : [...prev, spec]);
            }
            return;
        }
        if (onSelectedCsvByStemChange) {
            onSelectedCsvByStemChange({ ...(selectedCsvByStem || {}), [stem]: csvName });
        }
        if (!stemSet.has(stem)) {
            emitStems([...(selectedStems || []), stem]);
        }
    }, [isMultiPair, isMultiFolder, currentFolderPairsByStem, relatedFileGroups, currentDir, setPairs, onSelectedCsvByStemChange, selectedCsvByStem, stemSet, selectedStems, emitStems]);

    const selectableGroupKeys = isMultiPair
        ? relatedFileGroups.filter(isGroupSelectable).map((g) => g.key)
        : [];
    const selectedCount = isMultiPair
        ? (isMultiFolder
            ? selectableGroupKeys.filter((k) => currentFolderPairsByStem.has(k)).length
            : selectableGroupKeys.filter((k) => stemSet.has(k)).length)
        : 0;
    const allSelected = isMultiPair && selectableGroupKeys.length > 0 && selectedCount === selectableGroupKeys.length;

    // Manifest: trailing path segment shown as the folder "name" (head crumbs
    // shown verbatim, but the leaf gets prominent treatment in the panel).
    const folderLabel = (() => {
        const resolved = resolveTilde(value || '').replace(/\/+$/, '');
        if (!resolved || resolved === '/') return '/';
        const segs = resolved.split('/').filter(Boolean);
        return segs[segs.length - 1] || '/';
    })();
    const folderHeadPath = (() => {
        const resolved = resolveTilde(value || '').replace(/\/+$/, '');
        if (!resolved || resolved === '/') return '';
        const segs = resolved.split('/').filter(Boolean);
        if (segs.length <= 1) return '/';
        return '/' + segs.slice(0, -1).join('/') + '/';
    })();

    const openPicker = useCallback(() => {
        setIsOpen(true);
        inputRef.current?.focus();
    }, []);

    const clearAllPairs = useCallback(() => {
        if (!isMultiPair) return;
        if (isMultiFolder) {
            setPairs(() => []);
            return;
        }
        emitStems((selectedStems || []).filter((s) => !selectableGroupKeys.includes(s)));
        dropCsvForStems(selectableGroupKeys);
    }, [isMultiPair, isMultiFolder, setPairs, emitStems, selectedStems, selectableGroupKeys, dropCsvForStems]);

    const getVideoVariantLabel = (variant?: VideoVariant): string => {
        if (variant === 'remux') return 'browser copy';
        if (variant === 'h264') return 'H.264 copy';
        return 'source';
    };

    const renderSetMember = (
        file: RelatedFile,
        opts?: { csvSelectable?: boolean; csvActive?: boolean; onCsvPick?: () => void },
    ) => {
        const isCsv = file.kind === 'csv';
        const label = isCsv ? 'csv' : getVideoVariantLabel(file.variant);
        const csvSelectable = !!opts?.csvSelectable && isCsv;
        const csvActive = !!opts?.csvActive && isCsv;
        const handlePick = (e: React.SyntheticEvent) => {
            e.preventDefault();
            e.stopPropagation();
            opts?.onCsvPick?.();
        };
        return (
            <div
                className={`SetMember kind-${file.kind} variant-${file.variant || 'csv'} ${csvSelectable ? 'csv-pickable' : ''} ${csvActive ? 'csv-active' : ''}`}
                key={file.name}
                role={csvSelectable ? 'radio' : undefined}
                aria-checked={csvSelectable ? csvActive : undefined}
                tabIndex={csvSelectable ? 0 : undefined}
                onMouseDown={csvSelectable ? handlePick : undefined}
                onKeyDown={csvSelectable ? (e) => {
                    if (e.key === ' ' || e.key === 'Enter') handlePick(e);
                } : undefined}
            >
                {csvSelectable && (
                    <span className={`CsvRadio ${csvActive ? 'on' : ''}`} aria-hidden='true'>
                        {csvActive && (
                            <svg width="8" height="8" viewBox="0 0 16 16" fill="none">
                                <circle cx="8" cy="8" r="4" fill="currentColor" />
                            </svg>
                        )}
                    </span>
                )}
                <span className='SetMemberIcon' aria-hidden='true'>
                    {isCsv ? (
                        <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
                            <rect x="3" y="2" width="9" height="12" rx="1.2" stroke="currentColor" strokeWidth="1" fill="none" />
                            <path d="M5 6h6M5 8.5h6M5 11h4" stroke="currentColor" strokeWidth="0.9" strokeLinecap="round" />
                        </svg>
                    ) : (
                        <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
                            <rect x="2.5" y="3.5" width="11" height="9" rx="1" stroke="currentColor" strokeWidth="1" fill="none" />
                            <path d="M7 6.5 L10 8 L7 9.5 Z" fill="currentColor" />
                        </svg>
                    )}
                </span>
                <span className='SetMemberName' title={file.name}>{renderEntryName(file.name)}</span>
                <span className='SetMemberRole'>{csvActive ? 'chosen' : label}</span>
            </div>
        );
    };

    const renderRelatedFileGroup = (group: RelatedFileGroup, index: number) => {
        const sourceCount = group.videos.filter((file) => file.variant === 'source').length;
        const derivedCount = group.videos.length - sourceCount;
        const selectable = isMultiPair && isGroupSelectable(group);
        const checked = isMultiPair && (isMultiFolder
            ? currentFolderPairsByStem.has(group.key)
            : stemSet.has(group.key));
        const blocker = isMultiPair && !selectable
            ? (requireSourceVideo
                ? (group.csvs.length === 0 ? 'no csv' : (group.videos.some((f) => f.variant === 'source') ? '' : 'no source video'))
                : 'no video')
            : '';
        const showCsvPicker = isMultiPair && selectable && requireSourceVideo && group.csvs.length > 1;
        let activeCsvName: string | undefined;
        if (showCsvPicker) {
            if (isMultiFolder) {
                const sel = currentFolderPairsByStem.get(group.key);
                const selBase = sel ? basenameOfPath(sel.csvPath) : undefined;
                activeCsvName = (selBase && group.csvs.some((c) => c.name === selBase))
                    ? selBase
                    : group.csvs[0]?.name;
            } else {
                activeCsvName = (selectedCsvByStem?.[group.key] && group.csvs.some((c) => c.name === selectedCsvByStem![group.key]!))
                    ? selectedCsvByStem![group.key]!
                    : group.csvs[0]?.name;
            }
        }
        return (
            <div
                className={`FileSet ${isMultiPair ? (selectable ? 'pair-selectable' : 'pair-unselectable') : ''} ${checked ? 'pair-selected' : ''}`}
                key={group.key}
                title='Related files detected by shared video stem, CSV variant names, and browser-ready copy suffixes.'
                onMouseDown={isMultiPair && selectable ? (e) => {
                    // Avoid the wrapper's click-outside handler closing the dropdown.
                    e.preventDefault();
                    toggleStem(group.key);
                } : undefined}
                role={isMultiPair && selectable ? 'checkbox' : undefined}
                aria-checked={isMultiPair && selectable ? checked : undefined}
                aria-disabled={isMultiPair && !selectable ? true : undefined}
                tabIndex={isMultiPair && selectable ? 0 : undefined}
                onKeyDown={isMultiPair && selectable ? (e) => {
                    if (e.key === ' ' || e.key === 'Enter') {
                        e.preventDefault();
                        toggleStem(group.key);
                    }
                } : undefined}
            >
                <div className='FileSetTopline'>
                    {isMultiPair && (
                        <span
                            className={`PairCheckbox ${checked ? 'on' : ''} ${selectable ? '' : 'disabled'}`}
                            aria-hidden='true'
                        >
                            {checked ? (
                                <svg width="11" height="11" viewBox="0 0 16 16" fill="none">
                                    <path d="M3 8.5l3.2 3 6.8-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                </svg>
                            ) : null}
                        </span>
                    )}
                    <span className='FileSetIndex'>{String(index + 1).padStart(2, '0')}</span>
                    <span className='FileSetName' title={group.displayName}>{renderEntryName(group.displayName)}</span>
                    <span className='FileSetStatus'>{blocker || getGroupStatus(group)}</span>
                </div>
                <div className='FileSetTabs' aria-label='Related file counts'>
                    {sourceCount > 0 && <span className='FileSetTab source'>{sourceCount} source</span>}
                    {derivedCount > 0 && <span className='FileSetTab derived'>{derivedCount} copy</span>}
                    {group.csvs.length > 0 && <span className='FileSetTab csv'>{group.csvs.length} csv</span>}
                </div>
                <div className='FileSetMembers'>
                    {group.videos.map((file) => renderSetMember(file))}
                    {group.csvs.map((file) => renderSetMember(file, showCsvPicker ? {
                        csvSelectable: true,
                        csvActive: file.name === activeCsvName,
                        onCsvPick: () => setStemCsv(group.key, file.name),
                    } : undefined))}
                </div>
            </div>
        );
    };

    const breadcrumbs = splitBreadcrumbs(currentDir || resolveTilde(value));

    // multiFolder: group selected pairs by source folder for the manifest.
    // Order is "first appearance" so adding a new shelf doesn't rearrange
    // the user's existing ones.
    const manifestShelves = useMemo(() => {
        if (!isMultiFolder) return [] as { folder: string; pairs: ParsedPair[] }[];
        const order: string[] = [];
        const byFolder = new Map<string, ParsedPair[]>();
        for (const p of parsedPairs) {
            if (!byFolder.has(p.folder)) {
                byFolder.set(p.folder, []);
                order.push(p.folder);
            }
            byFolder.get(p.folder)!.push(p);
        }
        return order.map((folder) => ({ folder, pairs: byFolder.get(folder)! }));
    }, [isMultiFolder, parsedPairs]);

    // multiFolder: cache of {folder → DirEntry[]} so the inline CSV switcher
    // doesn't refetch on every open.
    const [folderEntriesCache, setFolderEntriesCache] = useState<Record<string, DirEntry[]>>({});

    const fetchFolderEntries = useCallback(async (folder: string): Promise<DirEntry[]> => {
        if (folderEntriesCache[folder]) return folderEntriesCache[folder];
        try {
            const params = new URLSearchParams({ path: folder, extensions: PAIR_PREVIEW_EXTS });
            const res = await fetch(`${API_URL}/api/ls?${params}`);
            if (!res.ok) return [];
            const data = await res.json();
            const fetched: DirEntry[] = data.entries || [];
            setFolderEntriesCache((prev) => ({ ...prev, [folder]: fetched }));
            return fetched;
        } catch {
            return [];
        }
    }, [folderEntriesCache]);

    // multiFolder: state for the currently-open inline CSV switcher.
    interface CsvPopoverState {
        pairSpec: string;       // the spec being edited
        folder: string;
        videoPath: string;
        currentCsv: string;     // basename of the CSV currently chosen
        stemKey: string;
        alternates: string[];   // CSV basenames in this folder that match the stem
        loading: boolean;
    }
    const [csvPopover, setCsvPopover] = useState<CsvPopoverState | null>(null);

    const openCsvPopover = useCallback(async (parsed: ParsedPair) => {
        const baseState: CsvPopoverState = {
            pairSpec: `${parsed.videoPath}=${parsed.csvPath}`,
            folder: parsed.folder,
            videoPath: parsed.videoPath,
            currentCsv: basenameOfPath(parsed.csvPath),
            stemKey: parsed.stemKey,
            alternates: [],
            loading: true,
        };
        setCsvPopover(baseState);
        const dirEntries = await fetchFolderEntries(parsed.folder);
        const groups = buildRelatedFileGroups(dirEntries);
        const match = groups.find((g) => g.key === parsed.stemKey);
        const alternates = match ? match.csvs.map((c) => c.name) : [basenameOfPath(parsed.csvPath)];
        setCsvPopover((prev) => prev && prev.pairSpec === baseState.pairSpec
            ? { ...prev, alternates, loading: false }
            : prev);
    }, [fetchFolderEntries]);

    const swapCsvForPair = useCallback((parsed: ParsedPair, newCsvName: string) => {
        const newCsvPath = joinPath(parsed.folder, newCsvName);
        if (newCsvPath === parsed.csvPath) {
            setCsvPopover(null);
            return;
        }
        const oldSpec = `${parsed.videoPath}=${parsed.csvPath}`;
        const newSpec = `${parsed.videoPath}=${newCsvPath}`;
        setPairs((prev) => prev.map((p) => p === oldSpec ? newSpec : p));
        setCsvPopover(null);
    }, [setPairs]);

    const removePair = useCallback((parsed: ParsedPair) => {
        const spec = `${parsed.videoPath}=${parsed.csvPath}`;
        setPairs((prev) => prev.filter((p) => p !== spec));
    }, [setPairs]);

    // Close the inline popover on outside click.
    useEffect(() => {
        if (!csvPopover) return undefined;
        const onDoc = (e: MouseEvent) => {
            const target = e.target as Element | null;
            if (target?.closest('.PairManifest__csvSwitch')) return;
            if (target?.closest('.CsvPopover')) return;
            setCsvPopover(null);
        };
        document.addEventListener('mousedown', onDoc);
        return () => document.removeEventListener('mousedown', onDoc);
    }, [csvPopover]);

    const browseShelf = useCallback((folder: string) => {
        navigateToDir(folder);
        setIsOpen(true);
    }, [navigateToDir]);

    const dropShelf = useCallback((folder: string) => {
        setPairs((prev) => prev.filter((spec) => {
            const parsed = parsePairSpec(spec);
            return parsed?.folder !== folder;
        }));
    }, [setPairs]);

    return (
        <div className={`PathPicker ${isOpen ? 'open' : ''}`} ref={wrapperRef}>
            {/* Breadcrumb bar */}
            {currentDir && breadcrumbs.length > 1 && (
                <div className='BreadcrumbBar'>
                    {breadcrumbs.map((crumb, i) => (
                        <React.Fragment key={i}>
                            {i > 0 && <span className='BreadcrumbSep'>/</span>}
                            <button
                                className={`BreadcrumbSegment ${siblingDropdown?.crumbIndex === i ? 'open' : ''}`}
                                onClick={() => handleBreadcrumbClick(i, crumb.fullPath)}
                                type='button'
                                tabIndex={-1}
                            >
                                {crumb.segment}
                            </button>
                            {siblingDropdown?.crumbIndex === i && (
                                <div className='SiblingDropdown'>
                                    {siblingDropdown.items.map((item) => (
                                        <div
                                            key={item.path}
                                            className={`SiblingItem ${item.path === crumb.fullPath ? 'current' : ''}`}
                                            onMouseDown={() => selectSibling(item)}
                                        >
                                            {getFolderName(item.path)}
                                        </div>
                                    ))}
                                    {siblingDropdown.items.length === 0 && (
                                        <div className='SiblingEmpty'>No siblings</div>
                                    )}
                                </div>
                            )}
                        </React.Fragment>
                    ))}
                </div>
            )}

            {/* Input */}
            <div className={`InputShell ${isOpen ? 'browsing' : ''}`}>
                <span className='PromptChar'>&gt;</span>
                <input
                    ref={inputRef}
                    type='text'
                    value={value}
                    onChange={(e) => handleChange(e.target.value)}
                    onFocus={handleFocus}
                    onKeyDown={handleKeyDown}
                    placeholder={placeholder}
                    autoComplete='off'
                    spellCheck={false}
                    disabled={disabled}
                />
                {isLoading && <span className='FetchSpinner' />}
            </div>

            {/* multiPair: prominent manifest of the current selection so the
                user can see (and edit) what they've committed without
                re-opening the dropdown. The path input above stays as
                secondary "where to browse" context. */}
            {isMultiPair && !isMultiFolder && (selectedStems?.length ?? 0) > 0 && (
                <div className='PairManifest' aria-label='Selected dataset pairs'>
                    <div className='PairManifest__head'>
                        <div className='PairManifest__headInfo'>
                            <span className='PairManifest__title'>
                                <span className='PairManifest__count'>{selectedStems!.length}</span>
                                <span className='PairManifest__label'>
                                    pair{selectedStems!.length !== 1 ? 's' : ''} selected
                                </span>
                            </span>
                            {folderHeadPath && (
                                <span className='PairManifest__folder' title={value}>
                                    <svg width="11" height="11" viewBox="0 0 16 16" fill="none" aria-hidden='true'>
                                        <path d="M2 3.5h4.5l1 1.5H14v8H2z" stroke="currentColor" strokeWidth="1.1" strokeLinejoin="round" fill="none"/>
                                    </svg>
                                    <span className='PairManifest__folderHead'>{folderHeadPath}</span>
                                    <span className='PairManifest__folderLeaf'>{folderLabel}</span>
                                </span>
                            )}
                        </div>
                        <div className='PairManifest__actions'>
                            <button
                                type='button'
                                className='PairManifest__btn primary'
                                onMouseDown={(e) => { e.preventDefault(); openPicker(); }}
                                aria-label='Edit pair selection'
                            >
                                <svg width="11" height="11" viewBox="0 0 16 16" fill="none" aria-hidden='true'>
                                    <path d="M3 13h2.5l7-7-2.5-2.5-7 7zM10.5 3.5l2 2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
                                </svg>
                                Edit selection
                            </button>
                            <button
                                type='button'
                                className='PairManifest__btn ghost'
                                onMouseDown={(e) => { e.preventDefault(); clearAllPairs(); }}
                                aria-label='Clear all selected pairs'
                                title='Remove all selected pairs'
                            >
                                Clear
                            </button>
                        </div>
                    </div>
                    <ol className='PairManifest__list'>
                        {selectedStems!.map((stem, i) => (
                            <li
                                key={stem}
                                className='PairManifest__row'
                                onMouseEnter={() => { /* hover state handled in CSS */ }}
                            >
                                <span className='PairManifest__index'>{String(i + 1).padStart(2, '0')}</span>
                                <span className='PairManifest__name' title={stem}>{stem}</span>
                                <span className='PairManifest__tags' aria-hidden='true'>
                                    <span className='PairManifest__tag video'>
                                        <svg width="10" height="10" viewBox="0 0 16 16" fill="none">
                                            <rect x="2.5" y="3.5" width="11" height="9" rx="1" stroke="currentColor" strokeWidth="1.1" fill="none"/>
                                            <path d="M7 6.5 L10 8 L7 9.5 Z" fill="currentColor"/>
                                        </svg>
                                        video
                                    </span>
                                    <span className='PairManifest__tag csv'>
                                        <svg width="10" height="10" viewBox="0 0 16 16" fill="none">
                                            <rect x="3" y="2" width="9" height="12" rx="1.2" stroke="currentColor" strokeWidth="1.1" fill="none"/>
                                            <path d="M5 6h6M5 8.5h6M5 11h4" stroke="currentColor" strokeWidth="0.9" strokeLinecap="round"/>
                                        </svg>
                                        csv
                                    </span>
                                </span>
                                <button
                                    type='button'
                                    className='PairManifest__remove'
                                    onMouseDown={(e) => { e.preventDefault(); toggleStem(stem); }}
                                    aria-label={`Remove ${stem}`}
                                    title={`Remove ${stem}`}
                                >
                                    <svg width="11" height="11" viewBox="0 0 16 16" fill="none">
                                        <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
                                    </svg>
                                </button>
                            </li>
                        ))}
                    </ol>
                </div>
            )}

            {/* multiFolder: shelves manifest. Each source folder is its own
                shelf with re-open / drop-folder controls; per-row CSV chip
                opens an inline switcher. */}
            {isMultiFolder && parsedPairs.length > 0 && (
                <div className='PairManifest PairManifest--multi' aria-label='Selected dataset pairs'>
                    <div className='PairManifest__head'>
                        <div className='PairManifest__headInfo'>
                            <span className='PairManifest__title'>
                                <span className='PairManifest__count'>{parsedPairs.length}</span>
                                <span className='PairManifest__label'>
                                    pair{parsedPairs.length !== 1 ? 's' : ''} selected
                                </span>
                            </span>
                            <span className='PairManifest__folder' aria-label='Folders represented'>
                                <svg width="11" height="11" viewBox="0 0 16 16" fill="none" aria-hidden='true'>
                                    <path d="M2 3.5h4.5l1 1.5H14v8H2z" stroke="currentColor" strokeWidth="1.1" strokeLinejoin="round" fill="none"/>
                                </svg>
                                <span className='PairManifest__folderHead'>across</span>
                                <span className='PairManifest__folderLeaf'>
                                    {manifestShelves.length} folder{manifestShelves.length !== 1 ? 's' : ''}
                                </span>
                            </span>
                        </div>
                        <div className='PairManifest__actions'>
                            <button
                                type='button'
                                className='PairManifest__btn primary'
                                onMouseDown={(e) => { e.preventDefault(); openPicker(); }}
                                aria-label='Add pairs from another folder'
                            >
                                <svg width="11" height="11" viewBox="0 0 16 16" fill="none" aria-hidden='true'>
                                    <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
                                </svg>
                                Add folder
                            </button>
                            <button
                                type='button'
                                className='PairManifest__btn ghost'
                                onMouseDown={(e) => { e.preventDefault(); clearAllPairs(); }}
                                aria-label='Clear all selected pairs'
                                title='Remove every selected pair'
                            >
                                Clear all
                            </button>
                        </div>
                    </div>
                    <div className='PairManifest__shelves'>
                        {manifestShelves.map((shelf) => {
                            const segs = shelf.folder.replace(/\/+$/, '').split('/').filter(Boolean);
                            const leaf = segs[segs.length - 1] || '/';
                            const head = segs.length <= 1 ? '/' : '/' + segs.slice(0, -1).join('/') + '/';
                            return (
                                <div className='ShelfBlock' key={shelf.folder}>
                                    <div className='ShelfHead'>
                                        <span className='ShelfPath' title={shelf.folder}>
                                            <svg width="11" height="11" viewBox="0 0 16 16" fill="none" aria-hidden='true'>
                                                <path d="M2 3.5h4.5l1 1.5H14v8H2z" stroke="currentColor" strokeWidth="1.1" strokeLinejoin="round" fill="none"/>
                                            </svg>
                                            <span className='ShelfHead__head'>{head}</span>
                                            <span className='ShelfHead__leaf'>{leaf}</span>
                                        </span>
                                        <span className='ShelfTally'>
                                            <strong>{shelf.pairs.length}</strong>&nbsp;pair{shelf.pairs.length !== 1 ? 's' : ''}
                                        </span>
                                        <div className='ShelfTools'>
                                            <button
                                                type='button'
                                                className='ShelfTool'
                                                onMouseDown={(e) => { e.preventDefault(); browseShelf(shelf.folder); }}
                                                aria-label={`Re-open ${shelf.folder} for editing`}
                                                title='Open this folder in the picker'
                                            >
                                                Browse
                                            </button>
                                            <button
                                                type='button'
                                                className='ShelfTool danger'
                                                onMouseDown={(e) => { e.preventDefault(); dropShelf(shelf.folder); }}
                                                aria-label={`Drop all pairs from ${shelf.folder}`}
                                                title='Remove every pair from this folder'
                                            >
                                                Drop
                                            </button>
                                        </div>
                                    </div>
                                    <ol className='PairManifest__list'>
                                        {shelf.pairs.map((parsed, i) => {
                                            const csvBase = basenameOfPath(parsed.csvPath);
                                            const spec = `${parsed.videoPath}=${parsed.csvPath}`;
                                            const popoverOpen = csvPopover?.pairSpec === spec;
                                            return (
                                                <li
                                                    key={spec}
                                                    className={`PairManifest__row ${popoverOpen ? 'has-popover' : ''}`}
                                                >
                                                    <span className='PairManifest__index'>{String(i + 1).padStart(2, '0')}</span>
                                                    <span className='PairManifest__name' title={parsed.stemKey}>{parsed.stemKey}</span>
                                                    <span className='PairManifest__tags'>
                                                        <span className='PairManifest__tag video' aria-hidden='true'>
                                                            <svg width="10" height="10" viewBox="0 0 16 16" fill="none">
                                                                <rect x="2.5" y="3.5" width="11" height="9" rx="1" stroke="currentColor" strokeWidth="1.1" fill="none"/>
                                                                <path d="M7 6.5 L10 8 L7 9.5 Z" fill="currentColor"/>
                                                            </svg>
                                                            video
                                                        </span>
                                                        <button
                                                            type='button'
                                                            className={`PairManifest__tag csv PairManifest__csvSwitch ${popoverOpen ? 'open' : ''}`}
                                                            onMouseDown={(e) => {
                                                                e.preventDefault();
                                                                if (popoverOpen) {
                                                                    setCsvPopover(null);
                                                                } else {
                                                                    openCsvPopover(parsed);
                                                                }
                                                            }}
                                                            aria-label={`Change CSV for ${parsed.stemKey}`}
                                                            aria-haspopup='listbox'
                                                            aria-expanded={popoverOpen}
                                                            title={`CSV: ${csvBase} — click to change`}
                                                        >
                                                            <svg width="10" height="10" viewBox="0 0 16 16" fill="none" aria-hidden='true'>
                                                                <rect x="3" y="2" width="9" height="12" rx="1.2" stroke="currentColor" strokeWidth="1.1" fill="none"/>
                                                                <path d="M5 6h6M5 8.5h6M5 11h4" stroke="currentColor" strokeWidth="0.9" strokeLinecap="round"/>
                                                            </svg>
                                                            <span className='PairManifest__csvLabel'>csv</span>
                                                            <span className='PairManifest__csvName' title={csvBase}>{csvBase}</span>
                                                            <svg className='PairManifest__csvCaret' width="8" height="8" viewBox="0 0 16 16" fill="none" aria-hidden='true'>
                                                                <path d="M3 6l5 5 5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
                                                            </svg>
                                                        </button>
                                                    </span>
                                                    <button
                                                        type='button'
                                                        className='PairManifest__remove'
                                                        onMouseDown={(e) => { e.preventDefault(); removePair(parsed); }}
                                                        aria-label={`Remove ${parsed.stemKey}`}
                                                        title={`Remove ${parsed.stemKey}`}
                                                    >
                                                        <svg width="11" height="11" viewBox="0 0 16 16" fill="none">
                                                            <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
                                                        </svg>
                                                    </button>
                                                    {popoverOpen && csvPopover && (
                                                        <div className='CsvPopover' role='listbox' aria-label={`CSV alternates for ${parsed.stemKey}`}>
                                                            <div className='CsvPopover__head'>
                                                                <span className='CsvPopover__heading'>Pick CSV for</span>
                                                                <span className='CsvPopover__stem'>{parsed.stemKey}</span>
                                                            </div>
                                                            {csvPopover.loading ? (
                                                                <div className='CsvPopover__loading'>Loading folder…</div>
                                                            ) : csvPopover.alternates.length === 0 ? (
                                                                <div className='CsvPopover__empty'>No CSV files matching this stem in {parsed.folder}.</div>
                                                            ) : (
                                                                <ul className='CsvPopover__opts'>
                                                                    {csvPopover.alternates.map((altName) => {
                                                                        const isActive = altName === csvPopover.currentCsv;
                                                                        return (
                                                                            <li key={altName}>
                                                                                <button
                                                                                    type='button'
                                                                                    className={`CsvPopover__opt ${isActive ? 'active' : ''}`}
                                                                                    onMouseDown={(e) => {
                                                                                        e.preventDefault();
                                                                                        swapCsvForPair(parsed, altName);
                                                                                    }}
                                                                                    role='option'
                                                                                    aria-selected={isActive}
                                                                                >
                                                                                    <span className={`CsvPopover__radio ${isActive ? 'on' : ''}`} aria-hidden='true'>
                                                                                        {isActive && <span className='CsvPopover__radioDot' />}
                                                                                    </span>
                                                                                    <span className='CsvPopover__optName' title={altName}>{altName}</span>
                                                                                    {isActive && <span className='CsvPopover__optBadge'>in use</span>}
                                                                                </button>
                                                                            </li>
                                                                        );
                                                                    })}
                                                                </ul>
                                                            )}
                                                            <div className='CsvPopover__foot' title={parsed.folder}>{parsed.folder}</div>
                                                        </div>
                                                    )}
                                                </li>
                                            );
                                        })}
                                    </ol>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Directory browser */}
            {isOpen && (
                <div className='DirBrowser'>
                    <div className='BrowserHeader'>
                        <span className='BrowserPath'>
                            {currentDir || '/'}
                        </span>
                        <span className='BrowserMeta'>
                            {filteredEntries.length} item{filteredEntries.length !== 1 ? 's' : ''}
                            <span className='HintKeys'>
                                <kbd>&uarr;</kbd><kbd>&darr;</kbd><kbd>Enter</kbd>
                            </span>
                        </span>
                    </div>
                    <div className='BrowserList' ref={listRef}>
                        {showParent && (
                            <div
                                className={`DirEntry parent ${selectedIndex === 0 ? 'active' : ''}`}
                                onMouseDown={() => navigateToDir(getParentDir(currentDir))}
                                onMouseEnter={() => setSelectedIndex(0)}
                            >
                                <svg className='EntryIcon' width="14" height="14" viewBox="0 0 16 16" fill="none">
                                    <path d="M10 4L6 8l4 4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
                                </svg>
                                <span className='EntryName'>..</span>
                            </div>
                        )}
                        {selectableEntries.map((entry, i) => {
                            const itemIndex = i + (showParent ? 1 : 0);
                            const fullPath = currentDir === '/' ? '/' + entry.name : currentDir + '/' + entry.name;
                            const isFile = entry.type === 'file';
                            const fileKind = isFile ? classifyFile(entry.name) : null;
                            return (
                                <div
                                    key={entry.name}
                                    className={`DirEntry ${entry.type} ${itemIndex === selectedIndex ? 'active' : ''} ${fileKind ? `kind-${fileKind}` : ''}`}
                                    onMouseDown={() => {
                                        if (entry.type === 'dir') {
                                            navigateToDir(fullPath);
                                        } else {
                                            selectFile(fullPath);
                                        }
                                    }}
                                    onMouseEnter={() => setSelectedIndex(itemIndex)}
                                >
                                    {entry.type === 'dir' ? (
                                        <svg className='EntryIcon' width="14" height="14" viewBox="0 0 16 16" fill="none">
                                            <path d="M2 3.5h4.5l1 1.5H14v8H2z" fill="currentColor" opacity="0.15" />
                                            <path d="M2 3.5h4.5l1 1.5H14v8H2z" stroke="currentColor" strokeWidth="1" strokeLinejoin="round" fill="none" />
                                        </svg>
                                    ) : fileKind === 'csv' ? (
                                        <svg className='EntryIcon' width="14" height="14" viewBox="0 0 16 16" fill="none">
                                            <rect x="3" y="2" width="9" height="12" rx="1.2" stroke="currentColor" strokeWidth="1" fill="none" />
                                            <path d="M5 6h6M5 8.5h6M5 11h4" stroke="currentColor" strokeWidth="0.9" strokeLinecap="round" />
                                        </svg>
                                    ) : (
                                        <svg className='EntryIcon' width="14" height="14" viewBox="0 0 16 16" fill="none">
                                            <rect x="2.5" y="3.5" width="11" height="9" rx="1" stroke="currentColor" strokeWidth="1" fill="none" />
                                            <path d="M7 6.5 L10 8 L7 9.5 Z" fill="currentColor" />
                                        </svg>
                                    )}
                                    <span className='EntryName'>
                                        {renderEntryName(entry.name)}
                                    </span>
                                    {fileKind && (
                                        <span className={`EntryKind kind-${fileKind}`}>{fileKind === 'csv' ? 'csv' : 'video'}</span>
                                    )}
                                    {entry.type === 'dir' && (
                                        <svg className='EntryChevron' width="12" height="12" viewBox="0 0 16 16" fill="none">
                                            <path d="M6 4l4 4-4 4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
                                        </svg>
                                    )}
                                </div>
                            );
                        })}
                        {filePreviewMode && relatedFileGroups.length > 0 && (
                            <div className='RelatedSetsBlock'>
                                <div className='RelatedSetsHeader'>
                                    <span className='RelatedSetsTab'>{isMultiPair ? 'Pick Pairs' : 'Related Sets'}</span>
                                    {isMultiPair && selectableGroupKeys.length > 0 ? (
                                        <span className='PairBulkActions'>
                                            <button
                                                type='button'
                                                className='PairBulkBtn'
                                                onMouseDown={(e) => {
                                                    e.preventDefault();
                                                    if (isMultiFolder) {
                                                        if (allSelected) {
                                                            // Drop only the current folder's pairs; pairs from other shelves stay
                                                            const dirPrefix = currentDir;
                                                            setPairs((prev) => prev.filter((spec) => {
                                                                const parsed = parsePairSpec(spec);
                                                                return parsed?.folder !== dirPrefix;
                                                            }));
                                                        } else {
                                                            const additions: string[] = [];
                                                            for (const g of relatedFileGroups) {
                                                                if (!isGroupSelectable(g)) continue;
                                                                if (currentFolderPairsByStem.has(g.key)) continue;
                                                                const video = g.videos.find((f) => f.variant === 'source') || g.videos[0];
                                                                const csv = g.csvs[0];
                                                                if (!video || !csv) continue;
                                                                additions.push(`${joinPath(currentDir, video.name)}=${joinPath(currentDir, csv.name)}`);
                                                            }
                                                            if (additions.length > 0) {
                                                                setPairs((prev) => [...prev, ...additions]);
                                                            }
                                                        }
                                                    } else if (allSelected) {
                                                        emitStems((selectedStems || []).filter((s) => !selectableGroupKeys.includes(s)));
                                                        dropCsvForStems(selectableGroupKeys);
                                                    } else {
                                                        emitStems([...(selectedStems || []), ...selectableGroupKeys]);
                                                    }
                                                }}
                                                aria-label={allSelected ? 'Clear pair selection' : 'Select all pairs'}
                                            >
                                                {allSelected ? 'Clear' : 'Select all'}
                                            </button>
                                            <span className='RelatedSetsCount'>
                                                {selectedCount} of {selectableGroupKeys.length} pair{selectableGroupKeys.length !== 1 ? 's' : ''}
                                            </span>
                                        </span>
                                    ) : (
                                        <span className='RelatedSetsCount'>
                                            {relatedFileCount} file{relatedFileCount !== 1 ? 's' : ''}
                                        </span>
                                    )}
                                </div>
                                {relatedFileGroups.map(renderRelatedFileGroup)}
                            </div>
                        )}
                        {filteredEntries.length === 0 && !showParent && (
                            <div className='EmptyState'>
                                {typedFilter ? `No matches for "${typedFilter}"` : 'This folder is empty'}
                            </div>
                        )}
                        {filteredEntries.length === 0 && showParent && (
                            <div className='EmptyState'>This folder is empty</div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default PathPicker;
