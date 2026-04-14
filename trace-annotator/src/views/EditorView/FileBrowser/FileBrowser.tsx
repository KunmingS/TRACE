import React, { useState, useRef, useEffect, useCallback } from 'react';
import { connect } from 'react-redux';
import { AppState } from '../../../store';
import { ImageData, LabelName } from '../../../store/labels/types';
import { addImageData, updateActiveImageIndex, updateImageData, updateLabelNames } from '../../../store/labels/actionCreators';
import { updateActivePopupType, updateVideoDirectory, updateVideoFiles } from '../../../store/general/actionCreators';
import { PopupWindowType } from '../../../data/enums/PopupWindowType';
import { ImageDataUtil } from '../../../utils/ImageDataUtil';
import { CSVImporter } from '../../../logic/import/csv/CSVImporter';
import { API_URL } from '../../../config';
import classNames from 'classnames';
import { submitNewNotification } from '../../../store/notifications/actionCreators';
import { NotificationUtil } from '../../../utils/NotificationUtil';
import './FileBrowser.scss';

interface FileInfo {
    name: string;
    hasCsv: boolean;
    codec?: string;
    isH264?: boolean;
    hasCachedH264?: boolean;
}

interface IProps {
    videoDirectory: string;
    videoFiles: string[];
    currentFileName: string;
    labelNames: LabelName[];
    isOpen: boolean;
    onToggleOpen: () => void;
    updateVideoDirectory: (dir: string) => any;
    updateVideoFiles: (files: string[]) => any;
    addImageData: (imageData: ImageData[]) => any;
    updateActiveImageIndex: (index: number) => any;
    updateImageData: (imageData: ImageData[]) => any;
    updateLabelNames: (labels: LabelName[]) => any;
    updateActivePopupType: (popupType: PopupWindowType) => any;
    submitNewNotification: typeof submitNewNotification;
}

const FileBrowser: React.FC<IProps> = (props) => {
    const [directory, setDirectory] = useState(props.videoDirectory || '');
    const [files, setFiles] = useState<string[]>(props.videoFiles || []);
    const [filesInfo, setFilesInfo] = useState<FileInfo[]>([]);
    const [suggestions, setSuggestions] = useState<string[]>([]);
    const [showSuggestions, setShowSuggestions] = useState(false);
    const [selectedSuggestion, setSelectedSuggestion] = useState(-1);
    const [loading, setLoading] = useState(false);
    const [convertingFiles, setConvertingFiles] = useState<Set<string>>(new Set());
    const debounceRef = useRef<number | null>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const suggestionsRef = useRef<HTMLDivElement>(null);

    // Sync from Redux on mount and fetch file info
    useEffect(() => {
        if (props.videoDirectory) {
            setDirectory(props.videoDirectory);
            // Fetch files info with CSV status
            fetchFilesInfo(props.videoDirectory);
        }
        if (props.videoFiles?.length) setFiles(props.videoFiles);
    }, []);

    const fetchFilesInfo = async (dir: string) => {
        const cleanDir = dir.replace(/\/+$/, '') || '/';
        try {
            const res = await fetch(`${API_URL}/api/files?dir=${encodeURIComponent(cleanDir)}`);
            if (res.ok) {
                const data = await res.json();
                setFilesInfo(data.filesInfo || []);
            }
        } catch {}
    };

    const fetchSuggestions = useCallback(async (prefix: string) => {
        const query = prefix || '/';
        try {
            const res = await fetch(`${API_URL}/api/dirs?prefix=${encodeURIComponent(query)}`);
            if (res.ok) {
                const data = await res.json();
                const dirs = data.dirs || [];
                setSuggestions(dirs);
                setShowSuggestions(dirs.length > 0);
            }
        } catch { setSuggestions([]); }
    }, []);

    const handleDirectoryChange = (value: string) => {
        if (value && !value.startsWith('/')) value = '/' + value;
        setDirectory(value);
        setSelectedSuggestion(-1);
        if (debounceRef.current) clearTimeout(debounceRef.current);
        debounceRef.current = window.setTimeout(() => fetchSuggestions(value), 120);
    };

    const selectSuggestion = (dir: string) => {
        const dirWithSlash = dir.endsWith('/') ? dir : dir + '/';
        setDirectory(dirWithSlash);
        setSelectedSuggestion(-1);
        inputRef.current?.focus();
        fetchSuggestions(dirWithSlash);
    };

    const loadFiles = async () => {
        if (!directory) return;
        setLoading(true);
        setShowSuggestions(false);
        const cleanDir = directory.replace(/\/+$/, '') || '/';
        try {
            const res = await fetch(`${API_URL}/api/files?dir=${encodeURIComponent(cleanDir)}`);
            if (res.ok) {
                const data = await res.json();
                const newFiles = data.files || [];
                setFiles(newFiles);
                setFilesInfo(data.filesInfo || []);
                props.updateVideoDirectory(cleanDir);
                props.updateVideoFiles(newFiles);
            }
        } catch (err) {
            console.error('Failed to fetch files', err);
        }
        setLoading(false);
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (!showSuggestions || suggestions.length === 0) {
            if (e.key === 'Enter' && directory) loadFiles();
            return;
        }
        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                setSelectedSuggestion(prev => prev < suggestions.length - 1 ? prev + 1 : 0);
                break;
            case 'ArrowUp':
                e.preventDefault();
                setSelectedSuggestion(prev => prev > 0 ? prev - 1 : suggestions.length - 1);
                break;
            case 'Enter':
                e.preventDefault();
                if (selectedSuggestion >= 0) selectSuggestion(suggestions[selectedSuggestion]);
                else { setShowSuggestions(false); loadFiles(); }
                break;
            case 'Escape':
                setShowSuggestions(false);
                break;
            case 'Tab':
                if (selectedSuggestion >= 0) { e.preventDefault(); selectSuggestion(suggestions[selectedSuggestion]); }
                else if (suggestions.length === 1) { e.preventDefault(); selectSuggestion(suggestions[0]); }
                break;
        }
    };

    const isFilePlayable = (info: FileInfo | undefined): boolean => {
        if (!info) return true; // no codec info, assume playable
        return !!(info.isH264 || info.hasCachedH264);
    };

    const handleConvert = async (file: string, e: React.MouseEvent) => {
        e.stopPropagation();
        const cleanDir = directory.replace(/\/+$/, '') || '/';
        const info = filesInfo.find(fi => fi.name === file);
        const codec = (info?.codec || 'unknown').toUpperCase();

        setConvertingFiles(prev => new Set(prev).add(file));
        props.submitNewNotification(
            NotificationUtil.createMessageNotification({
                header: `Converting ${codec} to H.264…`,
                description: `"${file}" — this may take a moment.`
            })
        );

        try {
            const res = await fetch(
                `${API_URL}/api/files/${file}/transcode?dir=${encodeURIComponent(cleanDir)}`,
                { method: 'POST' }
            );
            if (res.ok) {
                const result = await res.json();
                if (result.transcoded) {
                    setFilesInfo(prev => prev.map(fi =>
                        fi.name === file ? { ...fi, hasCachedH264: true } : fi
                    ));
                    props.submitNewNotification(
                        NotificationUtil.createMessageNotification({
                            header: 'Conversion complete',
                            description: `"${file}" is now ready to play.`
                        })
                    );
                } else {
                    props.submitNewNotification(
                        NotificationUtil.createMessageNotification({
                            header: 'Conversion failed',
                            description: `Could not convert "${file}" to H.264.`
                        })
                    );
                }
            }
        } catch (e) {
            console.error('Transcoding request failed', e);
        }
        setConvertingFiles(prev => { const next = new Set(prev); next.delete(file); return next; });
    };

    const handleFileClick = async (file: string) => {
        if (file === props.currentFileName) return;
        const info = filesInfo.find(fi => fi.name === file);

        // Block playback if file is not playable
        if (!isFilePlayable(info)) return;

        const cleanDir = directory.replace(/\/+$/, '') || '/';
        const videoUrl = `${API_URL}/api/files/${file}?dir=${encodeURIComponent(cleanDir)}`;

        // Create new image data and replace the current one
        const newImage = ImageDataUtil.createImageDataFromUrl(videoUrl, file);
        (newImage as any).videoPath = cleanDir;
        props.updateImageData([newImage]);
        props.updateActiveImageIndex(0);

        // Check if this file has a CSV
        if (info?.hasCsv) {
            try {
                const res = await fetch(`${API_URL}/api/files/${file}/csv?dir=${encodeURIComponent(cleanDir)}`);
                if (res.ok) {
                    const csvText = await res.text();
                    const rows = CSVImporter.parseCSV(csvText);
                    const importer = new CSVImporter([]);
                    const { imagesData: updatedImages, labelNames } = importer.applyLabels([newImage], rows);
                    props.updateLabelNames(labelNames);
                    props.updateImageData(updatedImages);
                }
            } catch {}
        } else {
            // No CSV — show label setup popup
            props.updateActivePopupType(PopupWindowType.INSERT_LABEL_NAMES);
        }
    };

    // Close suggestions on outside click
    useEffect(() => {
        const handler = (e: MouseEvent) => {
            if (suggestionsRef.current && !suggestionsRef.current.contains(e.target as Node) &&
                inputRef.current && !inputRef.current.contains(e.target as Node)) {
                setShowSuggestions(false);
            }
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, []);

    const getFileExt = (filename: string) => {
        const dot = filename.lastIndexOf('.');
        return dot > 0 ? filename.substring(dot) : '';
    };

    const getFileBase = (filename: string) => {
        const dot = filename.lastIndexOf('.');
        return dot > 0 ? filename.substring(0, dot) : filename;
    };

    if (!props.isOpen) {
        return (
            <div
                className="FileBrowser collapsed"
                onClick={props.onToggleOpen}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        props.onToggleOpen();
                    }
                }}
                aria-label="Expand file browser"
            >
                <span className="ToggleIcon">&raquo;</span>
            </div>
        );
    }

    return (
        <div className={classNames('FileBrowser', { empty: files.length === 0 })}>
            <div className="BrowserHeader">
                <div className="HeaderCopy">
                    <span className="HeaderLabel">Files</span>
                    <span className="HeaderSubtext">Drag edge to resize</span>
                </div>
                <span className="FileCount">{files.length}</span>
                <button
                    type='button'
                    className="CollapseBtn"
                    onClick={props.onToggleOpen}
                    aria-label="Collapse file browser"
                >
                    &laquo;
                </button>
            </div>
            <div className="DirSection">
                <div className="DirInputWrapper">
                    <span className="Prompt">&gt;</span>
                    <input
                        ref={inputRef}
                        type="text"
                        value={directory}
                        onChange={e => handleDirectoryChange(e.target.value)}
                        onFocus={() => suggestions.length > 0 ? setShowSuggestions(true) : fetchSuggestions(directory)}
                        onKeyDown={handleKeyDown}
                        placeholder="/path/to/videos"
                        autoComplete="off"
                        spellCheck={false}
                    />
                    <button type='button' className="LoadBtn" onClick={loadFiles} disabled={!directory} aria-label="Load files">
                        &#8629;
                    </button>
                </div>
                {showSuggestions && suggestions.length > 0 && (
                    <div className="SuggestionsDropdown" ref={suggestionsRef}>
                        {suggestions.map((dir, i) => (
                            <div
                                key={dir}
                                className={`SuggestionItem ${i === selectedSuggestion ? 'active' : ''}`}
                                onMouseDown={() => selectSuggestion(dir)}
                                onMouseEnter={() => setSelectedSuggestion(i)}
                            >
                                {dir.split('/').filter(Boolean).pop() || '/'}
                            </div>
                        ))}
                    </div>
                )}
            </div>
            <div className="FileList">
                {loading && <div className="LoadingIndicator">Loading...</div>}
                {!loading && files.length === 0 && (
                    <div className="EmptyState">
                        <div className="EmptyStateTitle">No files loaded</div>
                        <div className="EmptyStateText">Choose a server folder to populate this browser.</div>
                    </div>
                )}
                {!loading && files.map(file => {
                    const info = filesInfo.find(fi => fi.name === file);
                    const playable = isFilePlayable(info);
                    const converting = convertingFiles.has(file);
                    return (
                        <div
                            key={file}
                            className={classNames('FileItem', {
                                active: file === props.currentFileName,
                                unplayable: !playable && !converting,
                            })}
                            onClick={() => handleFileClick(file)}
                        >
                            <div className="FileMain">
                                <span className="FileBase">{getFileBase(file)}</span>
                                <span className="FileExt">{getFileExt(file)}</span>
                            </div>
                            <div className="FileMeta">
                                {playable ? (
                                    <span className="CodecBadge ready" title="Browser-compatible — click to play">
                                        {info?.hasCachedH264 ? 'H264' : info?.codec?.toUpperCase() || 'H264'}
                                    </span>
                                ) : converting ? (
                                    <span className="CodecBadge converting" title="Converting to H.264…">
                                        <span className="ConvertSpinner" />
                                        Converting
                                    </span>
                                ) : (
                                    <button
                                        type='button'
                                        className="CodecBadge convert-btn"
                                        title={`${(info?.codec || '').toUpperCase()} — click to convert to H.264`}
                                        onClick={(e) => handleConvert(file, e)}
                                    >
                                        Convert
                                    </button>
                                )}
                                {info?.hasCsv && (
                                    <span className="CsvBadge" title="Has annotation CSV">CSV</span>
                                )}
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

const mapStateToProps = (state: AppState) => ({
    videoDirectory: state.general.videoDirectory,
    videoFiles: state.general.videoFiles,
    currentFileName: state.labels.imagesData[state.labels.activeImageIndex]?.fileData?.name || '',
    labelNames: state.labels.labels
});

const mapDispatchToProps = {
    updateVideoDirectory,
    updateVideoFiles,
    addImageData,
    updateActiveImageIndex,
    updateImageData,
    updateLabelNames,
    updateActivePopupType,
    submitNewNotification
};

export default connect(mapStateToProps, mapDispatchToProps)(FileBrowser);
