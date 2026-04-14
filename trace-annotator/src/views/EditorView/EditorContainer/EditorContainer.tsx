import React, {useState, useRef, useEffect, useMemo, useCallback} from 'react';
import {connect} from 'react-redux';
import classNames from 'classnames';
import {ISize} from '../../../interfaces/ISize';
import {Settings} from '../../../settings/Settings';
import {AppState} from '../../../store';
import {ImageData} from '../../../store/labels/types';
import './EditorContainer.scss';
import Editor from '../Editor/Editor';
import EditorBottomNavigationBar from '../EditorBottomNavigationBar/EditorBottomNavigationBar';
import BehaviorShortcutsBar from '../BehaviorShortcutsBar/BehaviorShortcutsBar';
import FileBrowser from '../FileBrowser/FileBrowser';

interface IProps {
    windowSize: ISize;
    activeImageIndex: number;
    imagesData: ImageData[];
    zoom: number;
}

const FILE_BROWSER_WIDTH_STORAGE_KEY = 'trace:file-browser-width';
const FILE_BROWSER_COLLAPSED_STORAGE_KEY = 'trace:file-browser-collapsed';

const getStoredBrowserWidth = () => {
    if (typeof window === 'undefined') return Settings.FILE_BROWSER_WIDTH_PX;
    const storedValue = Number(window.localStorage.getItem(FILE_BROWSER_WIDTH_STORAGE_KEY));
    return Number.isFinite(storedValue) ? storedValue : Settings.FILE_BROWSER_WIDTH_PX;
};

const getStoredBrowserCollapsedState = () => {
    if (typeof window === 'undefined') return false;
    return window.localStorage.getItem(FILE_BROWSER_COLLAPSED_STORAGE_KEY) === 'true';
};

const EditorContainer: React.FC<IProps> = (
    {
        windowSize,
        activeImageIndex,
        imagesData,
        zoom,
    }) => {
    const [videoState, setVideoState] = useState({
        currentTime: 0,
        duration: 0,
        isPlaying: false,
        frameRate: 30
    });
    const [fileBrowserWidth, setFileBrowserWidth] = useState<number>(getStoredBrowserWidth);
    const [isFileBrowserOpen, setIsFileBrowserOpen] = useState<boolean>(() => !getStoredBrowserCollapsedState());
    const [isResizingFileBrowser, setIsResizingFileBrowser] = useState<boolean>(false);
    const [mainContentSize, setMainContentSize] = useState<ISize>(null);

    const editorRef = useRef<any>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const mainContentRef = useRef<HTMLDivElement>(null);
    const resizeStateRef = useRef<{
        startX: number;
        startWidth: number;
    } | null>(null);

    const getMaxFileBrowserWidth = useCallback((containerWidth: number) => {
        if (!containerWidth) return Settings.FILE_BROWSER_WIDTH_PX;

        const preferredWorkspaceWidth = Math.min(
            1080,
            720 + Math.max(0, zoom - 1) * 120
        );

        return Math.max(
            Settings.FILE_BROWSER_MIN_WIDTH_PX,
            Math.min(
                Settings.FILE_BROWSER_MAX_WIDTH_PX,
                containerWidth - preferredWorkspaceWidth
            )
        );
    }, [zoom]);

    const effectiveFileBrowserWidth = useMemo(() => {
        const containerWidth = containerRef.current?.clientWidth || windowSize?.width || 0;
        const maxWidth = getMaxFileBrowserWidth(containerWidth);

        return Math.max(
            Settings.FILE_BROWSER_MIN_WIDTH_PX,
            Math.min(fileBrowserWidth, maxWidth)
        );
    }, [fileBrowserWidth, getMaxFileBrowserWidth, windowSize]);

    const calculateEditorSize = (): ISize => {
        if (mainContentSize) {
            return {
                width: mainContentSize.width,
                height: mainContentSize.height - Settings.BEHAVIOR_BAR_HEIGHT_PX
                    - Settings.EDITOR_BOTTOM_NAVIGATION_BAR_HEIGHT_PX,
            };
        }

        if (windowSize) {
            return {
                width: windowSize.width - (isFileBrowserOpen ? effectiveFileBrowserWidth : Settings.FILE_BROWSER_COLLAPSED_WIDTH_PX)
                    - Settings.FILE_BROWSER_RESIZE_HANDLE_WIDTH_PX,
                height: windowSize.height - Settings.TOP_NAVIGATION_BAR_HEIGHT_PX
                    - Settings.BEHAVIOR_BAR_HEIGHT_PX
                    - Settings.EDITOR_BOTTOM_NAVIGATION_BAR_HEIGHT_PX,
            };
        }

        return null;
    };

    useEffect(() => {
        window.localStorage.setItem(FILE_BROWSER_WIDTH_STORAGE_KEY, String(effectiveFileBrowserWidth));
    }, [effectiveFileBrowserWidth]);

    useEffect(() => {
        window.localStorage.setItem(FILE_BROWSER_COLLAPSED_STORAGE_KEY, String(!isFileBrowserOpen));
    }, [isFileBrowserOpen]);

    useEffect(() => {
        if (isFileBrowserOpen && !isResizingFileBrowser && effectiveFileBrowserWidth !== fileBrowserWidth) {
            setFileBrowserWidth(effectiveFileBrowserWidth);
        }
    }, [effectiveFileBrowserWidth, fileBrowserWidth, isFileBrowserOpen, isResizingFileBrowser]);

    useEffect(() => {
        const element = mainContentRef.current;
        if (!element) return;

        const observer = new ResizeObserver((entries) => {
            const entry = entries[0];
            if (!entry) return;

            setMainContentSize({
                width: entry.contentRect.width,
                height: entry.contentRect.height
            });
        });

        observer.observe(element);
        return () => observer.disconnect();
    }, []);

    useEffect(() => {
        if (!isResizingFileBrowser) return;

        const handlePointerMove = (event: PointerEvent) => {
            const resizeState = resizeStateRef.current;
            const containerWidth = containerRef.current?.clientWidth || windowSize?.width || 0;
            if (!resizeState || !containerWidth) return;

            const nextWidth = resizeState.startWidth + (resizeState.startX - event.clientX);
            const maxWidth = getMaxFileBrowserWidth(containerWidth);

            setFileBrowserWidth(
                Math.max(
                    Settings.FILE_BROWSER_MIN_WIDTH_PX,
                    Math.min(nextWidth, maxWidth)
                )
            );
        };

        const finishResize = () => {
            resizeStateRef.current = null;
            setIsResizingFileBrowser(false);
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        };

        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        window.addEventListener('pointermove', handlePointerMove);
        window.addEventListener('pointerup', finishResize);
        window.addEventListener('pointercancel', finishResize);

        return () => {
            window.removeEventListener('pointermove', handlePointerMove);
            window.removeEventListener('pointerup', finishResize);
            window.removeEventListener('pointercancel', finishResize);
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        };
    }, [getMaxFileBrowserWidth, isResizingFileBrowser, windowSize]);

    const handleVideoStateChange = (state) => {
        setVideoState(state);
    };

    const handleTogglePlay = () => {
        if (editorRef.current && editorRef.current.togglePlay) {
            editorRef.current.togglePlay();
        }
    };

    const handleStepFrame = (direction: 'forward' | 'backward') => {
        if (editorRef.current && editorRef.current.stepFrame) {
            editorRef.current.stepFrame(direction);
        }
    };

    const handleStepSeconds = (direction: 'forward' | 'backward') => {
        if (editorRef.current && editorRef.current.stepSeconds) {
            editorRef.current.stepSeconds(direction);
        }
    };

    const handleJumpToTime = (timestamp: string) => {
        if (editorRef.current && editorRef.current.jumpToTime) {
            editorRef.current.jumpToTime(timestamp);
        }
    };

    const handleFileBrowserResizeStart = (event: React.PointerEvent<HTMLDivElement>) => {
        if (!isFileBrowserOpen) return;

        resizeStateRef.current = {
            startX: event.clientX,
            startWidth: effectiveFileBrowserWidth
        };
        setIsResizingFileBrowser(true);
    };

    const handleResetFileBrowserWidth = () => {
        setFileBrowserWidth(Settings.FILE_BROWSER_WIDTH_PX);
        setIsFileBrowserOpen(true);
    };

    const hasActiveFile = imagesData.length > 0 && imagesData[activeImageIndex];

    return (
        <div
            className={classNames('EditorContainer', {
                resizing: isResizingFileBrowser,
                collapsed: !isFileBrowserOpen
            })}
            ref={containerRef}
            style={{
                ['--file-browser-width' as string]: `${effectiveFileBrowserWidth}px`,
                ['--file-browser-collapsed-width' as string]: `${Settings.FILE_BROWSER_COLLAPSED_WIDTH_PX}px`,
                ['--file-browser-resize-handle-width' as string]: `${Settings.FILE_BROWSER_RESIZE_HANDLE_WIDTH_PX}px`,
                ['--editor-zoom-scale' as string]: zoom.toString()
            }}
        >
            <div className='MainContent' ref={mainContentRef}>
                {hasActiveFile ? (
                    <>
                        <div className='VideoSection'>
                            <Editor
                                ref={editorRef}
                                size={calculateEditorSize()}
                                imageData={imagesData[activeImageIndex]}
                                key='editor'
                                onVideoStateChange={handleVideoStateChange}
                            />
                        </div>
                        <BehaviorShortcutsBar
                            currentTime={videoState.currentTime}
                            frameRate={videoState.frameRate}
                        />
                        <EditorBottomNavigationBar
                            size={calculateEditorSize()}
                            key='editor-bottom-navigation-bar'
                            currentTime={videoState.currentTime}
                            duration={videoState.duration}
                            isPlaying={videoState.isPlaying}
                            frameRate={videoState.frameRate}
                            onTogglePlay={handleTogglePlay}
                            onStepFrame={handleStepFrame}
                            onStepSeconds={handleStepSeconds}
                            onJumpToTime={(time) => handleJumpToTime(time.toString())}
                        />
                    </>
                ) : (
                    <div className='EmptyEditorState'>
                        <div className='EmptyEditorContent'>
                            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
                                <polygon points="5 3 19 12 5 21 5 3"/>
                            </svg>
                            <h3>Select a video file</h3>
                            <p>Choose a video from the file browser on the right to begin annotating.</p>
                        </div>
                    </div>
                )}
            </div>
            {isFileBrowserOpen && (
                <div
                    className='FileBrowserResizeHandle'
                    onPointerDown={handleFileBrowserResizeStart}
                    onDoubleClick={handleResetFileBrowserWidth}
                    role='separator'
                    aria-label='Resize file browser'
                    aria-orientation='vertical'
                    aria-valuenow={effectiveFileBrowserWidth}
                    aria-valuemin={Settings.FILE_BROWSER_MIN_WIDTH_PX}
                    aria-valuemax={Settings.FILE_BROWSER_MAX_WIDTH_PX}
                    tabIndex={-1}
                >
                    <span />
                </div>
            )}
            <FileBrowser
                isOpen={isFileBrowserOpen}
                onToggleOpen={() => setIsFileBrowserOpen((prev) => !prev)}
            />
        </div>
    );
};

const mapStateToProps = (state: AppState) => ({
    windowSize: state.general.windowSize,
    activeImageIndex: state.labels.activeImageIndex,
    imagesData: state.labels.imagesData,
    zoom: state.general.zoom
});

export default connect(
    mapStateToProps
)(EditorContainer);
