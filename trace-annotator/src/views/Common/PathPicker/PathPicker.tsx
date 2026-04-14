import React, { useCallback, useEffect, useRef, useState } from 'react';
import './PathPicker.scss';
import { API_URL } from '../../../config';

interface PathPickerProps {
    value: string;
    onChange: (value: string) => void;
    onSubmit?: () => void;
    placeholder?: string;
    mode?: 'dir' | 'file';
    extensions?: string;
    disabled?: boolean;
    storageKey?: string;
}

interface DirEntry {
    name: string;
    type: 'dir' | 'file';
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
}) => {
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
            if (mode === 'file' && extensions) {
                params.set('extensions', extensions);
            }
            const res = await fetch(`${API_URL}/api/ls?${params}`);
            if (res.ok) {
                const data = await res.json();
                setEntries(data.entries || []);
                if (data.resolved) setCurrentDir(data.resolved);
            }
        } catch {
            setEntries([]);
        }
        setIsLoading(false);
    }, [mode, extensions]);

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

    // Auto-select the first matching entry when the filter changes (VS Code style)
    useEffect(() => {
        if (!isOpen) return;
        if (typedFilter && filteredEntries.length > 0) {
            setSelectedIndex(showParent ? 1 : 0);
        } else if (!typedFilter) {
            setSelectedIndex(-1);
        }
    }, [typedFilter, filteredEntries.length, isOpen, showParent]);

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (!isOpen) {
            if (e.key === 'Enter' && value && onSubmit) onSubmit();
            return;
        }

        const totalItems = (showParent ? 1 : 0) + filteredEntries.length;
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
                    // No selection — submit current value
                    if (value && onSubmit) onSubmit();
                    return;
                }
                const isParentSelected = showParent && selectedIndex === 0;
                if (isParentSelected) {
                    navigateToDir(getParentDir(currentDir));
                } else {
                    const entryIndex = selectedIndex - (showParent ? 1 : 0);
                    const entry = filteredEntries[entryIndex];
                    if (entry) {
                        if (entry.type === 'dir') {
                            navigateToDir(currentDir === '/' ? '/' + entry.name : currentDir + '/' + entry.name);
                        } else {
                            selectFile(currentDir === '/' ? '/' + entry.name : currentDir + '/' + entry.name);
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
                        const entry = filteredEntries[entryIndex];
                        if (entry?.type === 'dir') {
                            navigateToDir(currentDir === '/' ? '/' + entry.name : currentDir + '/' + entry.name);
                        }
                    }
                } else if (filteredEntries.length === 1 && filteredEntries[0].type === 'dir') {
                    e.preventDefault();
                    const entry = filteredEntries[0];
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

    const breadcrumbs = splitBreadcrumbs(currentDir || resolveTilde(value));

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
                        {filteredEntries.map((entry, i) => {
                            const itemIndex = i + (showParent ? 1 : 0);
                            const fullPath = currentDir === '/' ? '/' + entry.name : currentDir + '/' + entry.name;
                            return (
                                <div
                                    key={entry.name}
                                    className={`DirEntry ${entry.type} ${itemIndex === selectedIndex ? 'active' : ''}`}
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
                                    ) : (
                                        <svg className='EntryIcon' width="14" height="14" viewBox="0 0 16 16" fill="none">
                                            <path d="M4 2h5l3 3v9H4z" stroke="currentColor" strokeWidth="1" strokeLinejoin="round" />
                                            <path d="M9 2v3h3" stroke="currentColor" strokeWidth="1" strokeLinejoin="round" />
                                        </svg>
                                    )}
                                    <span className='EntryName'>
                                        {typedFilter && entry.name.toLowerCase().startsWith(typedFilter) ? (
                                            <><span className='MatchHighlight'>{entry.name.slice(0, typedFilter.length)}</span>{entry.name.slice(typedFilter.length)}</>
                                        ) : entry.name}
                                    </span>
                                    {entry.type === 'dir' && (
                                        <svg className='EntryChevron' width="12" height="12" viewBox="0 0 16 16" fill="none">
                                            <path d="M6 4l4 4-4 4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
                                        </svg>
                                    )}
                                </div>
                            );
                        })}
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
