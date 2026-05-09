import React, {
    ChangeEvent,
    DragEvent,
    PropsWithChildren,
    useMemo,
    useRef,
    useState
} from 'react';
import './ImagesDropZone.scss';
import {connect} from 'react-redux';
import {AppState} from '../../../store';
import {ProjectType} from '../../../data/enums/ProjectType';
import {updateProjectData, updateVideoDirectory, updateVideoFiles} from '../../../store/general/actionCreators';
import {ProjectData} from '../../../store/general/types';
import {API_URL} from '../../../config';
import PathPicker from '../../Common/PathPicker/PathPicker';

interface IProps {
    updateProjectDataAction: (projectData: ProjectData) => any;
    updateVideoDirectoryAction: (dir: string) => any;
    updateVideoFilesAction: (files: string[]) => any;
    projectData: ProjectData;
}

type IntakeMode = 'server' | 'local';
type BusyState = 'idle' | 'scanning' | 'uploading';

const SUPPORTED_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];
const SUPPORTED_LABEL_EXTENSIONS = ['.csv'];
const SUPPORTED_INTAKE_EXTENSIONS = [...SUPPORTED_VIDEO_EXTENSIONS, ...SUPPORTED_LABEL_EXTENSIONS];

const ImagesDropZone: React.FC<IProps> = (props: PropsWithChildren<IProps>) => {
    const [intakeMode, setIntakeMode] = useState<IntakeMode>('server');
    const [busyState, setBusyState] = useState<BusyState>('idle');
    const [error, setError] = useState<string>('');
    const [statusMessage, setStatusMessage] = useState<string>('');
    const [directory, setDirectory] = useState<string>('');
    const [uploadDirectory, setUploadDirectory] = useState<string>('');
    const [localFiles, setLocalFiles] = useState<File[]>([]);
    const [dragActive, setDragActive] = useState<boolean>(false);

    const localFileInputRef = useRef<HTMLInputElement>(null);

    const enterProject = (targetDirectory: string, files: string[]) => {
        props.updateVideoDirectoryAction(targetDirectory);
        props.updateVideoFilesAction(files);
        props.updateProjectDataAction({...props.projectData, type: ProjectType.VIDEO});
    };

    const listDirectoryFiles = async (targetDirectory: string) => {
        const cleanDir = targetDirectory.replace(/\/+$/, '') || '/';
        const res = await fetch(`${API_URL}/api/files?dir=${encodeURIComponent(cleanDir)}`, {method: 'GET'});

        if (!res.ok) {
            throw new Error(`Server responded with ${res.status}: ${res.statusText}`);
        }

        const data = await res.json();
        return {
            cleanDir,
            files: data.files || []
        };
    };

    const loadServerFiles = async () => {
        if (!directory) {
            setError('Choose a server directory first.');
            return;
        }

        setBusyState('scanning');
        setError('');
        setStatusMessage('');

        try {
            const {cleanDir, files} = await listDirectoryFiles(directory);
            if (files.length === 0) {
                setError('No supported video files were found in this directory.');
                return;
            }
            enterProject(cleanDir, files);
        } catch (err: any) {
            console.error('Failed to fetch files', err);
            setError(`Cannot load files: ${err.message}`);
        } finally {
            setBusyState('idle');
        }
    };

    const uploadLocalFiles = async () => {
        const videoCount = localFiles.filter((file) => isSupportedVideoFile(file.name)).length;

        if (localFiles.length === 0 || videoCount === 0) {
            setError('Select at least one local video, with any CSV companions, before uploading.');
            return;
        }

        if (!uploadDirectory) {
            setError('Choose the server destination for the upload.');
            return;
        }

        const cleanDir = uploadDirectory.replace(/\/+$/, '') || '/';
        const formData = new FormData();
        formData.append('destination', cleanDir);
        localFiles.forEach((file) => formData.append('files', file, file.name));

        setBusyState('uploading');
        setError('');
        setStatusMessage('');

        try {
            const res = await fetch(`${API_URL}/api/upload-videos`, {
                method: 'POST',
                body: formData
            });

            if (!res.ok) {
                let detail = `Upload failed with ${res.status}`;
                try {
                    const payload = await res.json();
                    detail = payload.detail || detail;
                } catch {
                    detail = `${detail}: ${res.statusText}`;
                }
                throw new Error(detail);
            }

            const uploadResult = await res.json();
            const {cleanDir: uploadedDir, files} = await listDirectoryFiles(uploadResult.directory || cleanDir);
            setLocalFiles([]);
            enterProject(uploadedDir, files);
        } catch (err: any) {
            console.error('Failed to upload files', err);
            setError(err.message || 'Upload failed');
        } finally {
            setBusyState('idle');
        }
    };

    const isSupportedVideoFile = (fileName: string) => {
        const lowerName = fileName.toLowerCase();
        return SUPPORTED_VIDEO_EXTENSIONS.some((ext) => lowerName.endsWith(ext));
    };

    const isSupportedLabelFile = (fileName: string) => {
        const lowerName = fileName.toLowerCase();
        return SUPPORTED_LABEL_EXTENSIONS.some((ext) => lowerName.endsWith(ext));
    };

    const isSupportedIntakeFile = (fileName: string) =>
        isSupportedVideoFile(fileName) || isSupportedLabelFile(fileName);

    const formatBytes = (bytes: number) => {
        if (bytes < 1024) return `${bytes} B`;

        const units = ['KB', 'MB', 'GB', 'TB'];
        let value = bytes;
        let unitIndex = -1;

        while (value >= 1024 && unitIndex < units.length - 1) {
            value /= 1024;
            unitIndex += 1;
        }

        return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unitIndex]}`;
    };

    const ingestLocalFiles = (filesInput: FileList | File[]) => {
        const pickedFiles = Array.from(filesInput);
        const acceptedFiles = pickedFiles.filter((file) => isSupportedIntakeFile(file.name));
        const rejectedCount = pickedFiles.length - acceptedFiles.length;
        const acceptedVideoCount = acceptedFiles.filter((file) => isSupportedVideoFile(file.name)).length;
        const acceptedCsvCount = acceptedFiles.filter((file) => isSupportedLabelFile(file.name)).length;

        setLocalFiles(acceptedFiles);
        setError('');

        if (acceptedFiles.length === 0) {
            setStatusMessage('');
            setError('Select supported video files and CSV companions to upload.');
            return;
        }

        if (acceptedVideoCount === 0) {
            setStatusMessage('');
            setError('Add at least one video file so TRACE can open the uploaded set.');
            return;
        }

        if (!uploadDirectory && directory) {
            setUploadDirectory(directory.endsWith('/') ? directory : `${directory}/`);
        }

        const totalSize = acceptedFiles.reduce((sum, file) => sum + file.size, 0);
        const message = `${acceptedVideoCount} video${acceptedVideoCount === 1 ? '' : 's'}`
            + (acceptedCsvCount > 0 ? ` + ${acceptedCsvCount} CSV companion${acceptedCsvCount === 1 ? '' : 's'}` : '')
            + ' selected';
        setStatusMessage(
            rejectedCount > 0
                ? `${message}. ${rejectedCount} unsupported file${rejectedCount === 1 ? '' : 's'} ignored.`
                : `${message}, ${formatBytes(totalSize)} ready for upload.`
        );
    };

    const handleLocalFileChange = (event: ChangeEvent<HTMLInputElement>) => {
        if (event.target.files?.length) {
            ingestLocalFiles(event.target.files);
            event.target.value = '';
        }
    };

    const handleDrop = (event: DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        event.stopPropagation();
        setDragActive(false);
        if (event.dataTransfer.files?.length) {
            ingestLocalFiles(event.dataTransfer.files);
        }
    };

    const localFileSummary = useMemo(() => {
        const totalSize = localFiles.reduce((sum, file) => sum + file.size, 0);
        return {
            count: localFiles.length,
            totalSize: formatBytes(totalSize),
            videoCount: localFiles.filter((file) => isSupportedVideoFile(file.name)).length,
            csvCount: localFiles.filter((file) => isSupportedLabelFile(file.name)).length
        };
    }, [localFiles]);

    const isBusy = busyState !== 'idle';
    const busyLabel = busyState === 'uploading' ? 'Uploading videos to the server...' : 'Scanning directory...';

    return (
        <div className='ImagesDropZone'>
            <div className='ModeSwitch'>
                <button
                    className={`ModeButton ${intakeMode === 'server' ? 'active' : ''}`}
                    onClick={() => {
                        setIntakeMode('server');
                        setError('');
                        setStatusMessage('');
                    }}
                    type='button'
                >
                    <span className='ModeKicker'>01</span>
                    <span className='ModeLabel'>Server Folder</span>
                </button>

                <button
                    className={`ModeButton ${intakeMode === 'local' ? 'active' : ''}`}
                    onClick={() => {
                        setIntakeMode('local');
                        setError('');
                        setStatusMessage('');
                    }}
                    type='button'
                >
                    <span className='ModeKicker'>02</span>
                    <span className='ModeLabel'>Local Upload</span>
                </button>
            </div>

            <div className='LeadBlock'>
                <div className='LeadTitle'>
                    {intakeMode === 'server'
                        ? 'Open videos already stored on the lab server.'
                        : 'Bring videos from your computer, then place them on the server.'}
                </div>
                <div className='LeadText'>
                    {intakeMode === 'server'
                        ? 'Type a server path or use autocomplete, then load the folder into the annotation workspace.'
                        : 'Browser files cannot be annotated in place. TRACE uploads them first so the video and CSV outputs stay together on the server.'}
                </div>
            </div>

            {intakeMode === 'server' && (
                <div className='IntakePanel'>
                    <div className='Section'>
                        <div className='SectionLabel'>
                            <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                                <path d="M1.5 2.5h5l1.5 2h6.5v9h-13z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" fill="none"/>
                            </svg>
                            Server Directory
                        </div>

                        <div className='PathInputRow'>
                            <PathPicker
                                value={directory}
                                onChange={setDirectory}
                                onSubmit={loadServerFiles}
                                placeholder='/path/to/videos'
                                storageKey='annotate-server'
                                previewExtensions='.mp4,.avi,.mov,.mkv,.webm,.csv'
                            />
                            <button className='LoadBtn' onClick={loadServerFiles} type='button' disabled={!directory}>
                                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                                    <path d="M8 3v7m0 0l-3-3m3 3l3-3M3 13h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                                </svg>
                                Load Folder
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {intakeMode === 'local' && (
                <div className='IntakePanel local'>
                    <div
                        className={`DropSurface ${dragActive ? 'dragging' : ''}`}
                        onDragOver={(event) => {
                            event.preventDefault();
                            event.stopPropagation();
                            setDragActive(true);
                        }}
                        onDragLeave={(event) => {
                            event.preventDefault();
                            event.stopPropagation();
                            setDragActive(false);
                        }}
                        onDrop={handleDrop}
                    >
                        <input
                            ref={localFileInputRef}
                            type='file'
                            multiple
                            accept={SUPPORTED_INTAKE_EXTENSIONS.join(',')}
                            onChange={handleLocalFileChange}
                            hidden
                        />

                        <div className='DropSurfaceInner'>
                            <div className='DropIcon'>
                                <svg width="22" height="22" viewBox="0 0 16 16" fill="none">
                                    <path d="M8 2.5v7m0 0L5.5 7m2.5 2.5L10.5 7M2.5 12.5h11" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
                                </svg>
                            </div>
                            <div className='DropTitle'>Drop local video and CSV files here.</div>
                            <div className='DropText'>Supported formats: {SUPPORTED_INTAKE_EXTENSIONS.join(', ')}</div>
                            <button
                                className='GhostButton'
                                onClick={() => localFileInputRef.current?.click()}
                                type='button'
                            >
                                Select Local Files
                            </button>
                        </div>
                    </div>

                    <div className='UploadPrompt'>
                        <div className='PromptTitle'>Upload destination</div>
                        <div className='PromptText'>
                            Type an existing server folder or a new absolute path. TRACE creates the destination if it does not exist yet.
                        </div>
                    </div>

                    <div className='PathInputRow'>
                        <PathPicker
                            value={uploadDirectory}
                            onChange={setUploadDirectory}
                            onSubmit={localFiles.length > 0 ? uploadLocalFiles : undefined}
                            placeholder='/srv/trace/session-01'
                            storageKey='annotate-upload'
                        />
                        <button
                            className='LoadBtn'
                            onClick={uploadLocalFiles}
                            type='button'
                            disabled={!uploadDirectory || localFiles.length === 0}
                        >
                            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                                <path d="M8 3v7m0 0l-3-3m3 3l3-3M3 13h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                            Upload And Open
                        </button>
                    </div>

                    {localFiles.length > 0 && (
                        <div className='SelectedFilesPanel'>
                            <div className='SectionLabel'>
                                <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                                    <path d="M3 2.5h6l3 3v8H3z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round"/>
                                    <path d="M9 2.5v3h3" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round"/>
                                </svg>
                                Local Files
                                <span className='CountBadge'>{localFileSummary.count}</span>
                            </div>
                            <div className='UploadMeta'>
                                <span>{localFileSummary.totalSize}</span>
                                <span>
                                    {localFileSummary.videoCount} video{localFileSummary.videoCount === 1 ? '' : 's'}
                                    {localFileSummary.csvCount > 0
                                        ? ` + ${localFileSummary.csvCount} CSV companion${localFileSummary.csvCount === 1 ? '' : 's'}`
                                        : ''}
                                </span>
                            </div>
                            <div className='LocalFilesList'>
                                {localFiles.slice(0, 5).map((file) => (
                                    <div className='LocalFileItem' key={`${file.name}-${file.size}`}>
                                        <span className='LocalFileName'>{file.name}</span>
                                        <span className={`LocalFileKind ${isSupportedLabelFile(file.name) ? 'csv' : 'video'}`}>
                                            {isSupportedLabelFile(file.name) ? 'csv' : 'video'}
                                        </span>
                                        <span className='LocalFileSize'>{formatBytes(file.size)}</span>
                                    </div>
                                ))}
                                {localFiles.length > 5 && (
                                    <div className='LocalFileOverflow'>
                                        +{localFiles.length - 5} more file{localFiles.length - 5 === 1 ? '' : 's'}
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {isBusy && (
                <div className='LoadingState'>
                    <div className='LoadingDots'>
                        <span /><span /><span />
                    </div>
                    {busyLabel}
                </div>
            )}

            {statusMessage && !error && !isBusy && (
                <div className='InfoState'>
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                        <circle cx="8" cy="8" r="6.5" stroke="currentColor" strokeWidth="1.2"/>
                        <path d="M8 7.1v3.4M8 4.8h.01" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
                    </svg>
                    {statusMessage}
                </div>
            )}

            {error && (
                <div className='ErrorState'>
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                        <circle cx="8" cy="8" r="6.5" stroke="currentColor" strokeWidth="1.2"/>
                        <path d="M8 5v3.5M8 10.5v.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
                    </svg>
                    {error}
                </div>
            )}
        </div>
    );
};

const mapDispatchToProps = {
    updateProjectDataAction: updateProjectData,
    updateVideoDirectoryAction: updateVideoDirectory,
    updateVideoFilesAction: updateVideoFiles
};

const mapStateToProps = (state: AppState) => ({
    projectData: state.general.projectData
});

export default connect(
    mapStateToProps,
    mapDispatchToProps
)(ImagesDropZone);
