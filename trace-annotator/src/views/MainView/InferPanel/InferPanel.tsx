import React, { useEffect, useState } from 'react';
import './InferPanel.scss';
import PathPicker from '../../Common/PathPicker/PathPicker';
import LogViewer from '../../Common/LogViewer/LogViewer';
import { API_URL } from '../../../config';
import { useJobRunner } from '../../../hooks/useJobRunner';

interface ModelInfo {
    path: string;
    label: string;
    config_path: string | null;
    classes: string[];
    num_classes: number;
    size_mb: number;
}

interface Detection {
    segment: [number, number];
    label: string;
    score: number;
}

interface PredictionArtifact {
    predictions?: Record<string, Detection[]>;
    threshold?: number;
    annotated_videos?: Record<string, string>;
}

const DEFAULT_THRESHOLD = 0.3;

function parseThresholdInput(input: HTMLInputElement): number {
    const value = input.valueAsNumber;
    if (!Number.isFinite(value)) return DEFAULT_THRESHOLD;
    return Math.min(1, Math.max(0, value));
}

const InferPanel: React.FC = () => {
    const [modelPath, setModelPath] = useState('');
    const [inputPath, setInputPath] = useState('');
    const [annotatedVideo, setAnnotatedVideo] = useState(false);
    const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
    const [models, setModels] = useState<ModelInfo[]>([]);
    const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null);
    const [predictions, setPredictions] = useState<Record<string, Detection[]> | null>(null);
    const [annotatedVideos, setAnnotatedVideos] = useState<Record<string, string> | null>(null);
    const [error, setError] = useState('');

    const runner = useJobRunner();

    const isBusy = runner.status === 'running' || runner.status === 'submitting';

    useEffect(() => {
        fetch(`${API_URL}/api/models`)
            .then((res) => res.json())
            .then(setModels)
            .catch(() => {});
    }, []);

    useEffect(() => {
        const match = models.find((m) => modelPath && m.path === modelPath);
        setSelectedModel(match || null);
    }, [modelPath, models]);

    // Fetch predictions when job completes
    useEffect(() => {
        if (runner.status === 'completed') {
            runner.fetchArtifact<PredictionArtifact>('predictions.json')
                .then((data) => {
                    if (data?.predictions) setPredictions(data.predictions);
                    setAnnotatedVideos(data?.annotated_videos || null);
                });
        } else if (runner.status === 'failed') {
            setError(runner.error || 'Inference failed.');
        }
    }, [runner.status]);

    const handleRun = async () => {
        if (!modelPath) { setError('Select a model.'); return; }
        if (!inputPath) { setError('Select input video(s).'); return; }
        setError('');
        setPredictions(null);
        setAnnotatedVideos(null);

        try {
            await runner.submit('infer', {
                model_dir: modelPath,
                input: inputPath,
                annotated_video: annotatedVideo,
                threshold,
            });
        } catch (err: any) {
            setError(err.message);
        }
    };

    return (
        <div className='InferPanel'>
            <div className='PanelHeader'>
                <h2>Run Inference</h2>
                <p>Run a trained model on new videos to detect behaviors.</p>
            </div>

            <div className='FormSection'>
                <label className='FieldLabel'>Model</label>
                {models.length > 0 && (
                    <div className='ModelList'>
                        {models.map((m) => (
                            <button
                                key={m.path}
                                className={`ModelCard ${selectedModel?.path === m.path ? 'active' : ''}`}
                                onClick={() => { setModelPath(m.path); setSelectedModel(m); }}
                                type='button'
                            >
                                <div className='ModelName'>{m.label}</div>
                                <div className='ModelMeta'>
                                    {m.num_classes} classes &middot; {m.size_mb} MB
                                </div>
                            </button>
                        ))}
                    </div>
                )}
                <PathPicker
                    value={modelPath}
                    onChange={setModelPath}
                    placeholder='/path/to/model'
                    disabled={isBusy}
                />
                {selectedModel && (
                    <div className='ModelDetail'>
                        Classes: {selectedModel.classes.join(', ')}
                    </div>
                )}
            </div>

            <div className='FormSection'>
                <label className='FieldLabel'>Input Video(s)</label>
                <PathPicker
                    value={inputPath}
                    onChange={setInputPath}
                    placeholder='/path/to/video.mp4 or /path/to/folder'
                    mode='file'
                    extensions='.mp4,.avi,.mov,.mkv,.webm'
                    disabled={isBusy}
                />
            </div>

            <div className='FormSection'>
                <label className='FieldLabel'>Prediction Output</label>
                <div className='OutputControls'>
                    <label className='ToggleOption'>
                        <input
                            type='checkbox'
                            checked={annotatedVideo}
                            onChange={(e) => setAnnotatedVideo(e.target.checked)}
                            disabled={isBusy}
                        />
                        <span className='ToggleTrack'><span className='ToggleThumb' /></span>
                        <span className='ToggleText'>Annotated video</span>
                    </label>
                    <div className='ThresholdControl'>
                        <div className='ThresholdTop'>
                            <span>Threshold</span>
                        </div>
                        <input
                            type='number'
                            min='0'
                            max='1'
                            step='0.01'
                            value={threshold}
                            onChange={(e) => setThreshold(parseThresholdInput(e.currentTarget))}
                            disabled={isBusy}
                            aria-label='Prediction threshold'
                        />
                    </div>
                </div>
            </div>

            <button
                className='ActionBtn primary'
                onClick={handleRun}
                disabled={!modelPath || !inputPath || isBusy}
                type='button'
            >
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <path d="M5 3l8 5-8 5z" fill="currentColor" />
                </svg>
                Run Inference
            </button>

            {runner.jobId && <LogViewer jobId={runner.jobId} onCancel={() => runner.cancel()} />}

            {predictions && (
                <div className='ResultsCard'>
                    <div className='ResultsTitle'>
                        Predictions
                        <span className='ResultsCount'>
                            {Object.values(predictions).reduce((s, d) => s + d.length, 0)} detections
                        </span>
                    </div>
                    {annotatedVideos && (
                        <div className='AnnotatedOutputs'>
                            {Object.entries(annotatedVideos).map(([video, filename]) => (
                                <div key={video} className='AnnotatedOutput'>
                                    <span>{video}</span>
                                    <a
                                        href={runner.jobId ? `${API_URL}/api/jobs/${runner.jobId}/artifacts/${filename}` : undefined}
                                        target='_blank'
                                        rel='noreferrer'
                                    >
                                        <code>{filename}</code>
                                    </a>
                                </div>
                            ))}
                        </div>
                    )}
                    {Object.entries(predictions).map(([video, dets]) => (
                        <div key={video} className='VideoResult'>
                            <div className='VideoName'>{video}</div>
                            <div className='DetList'>
                                {dets.slice(0, 10).map((det, i) => (
                                    <div key={i} className='DetRow'>
                                        <span className='DetTime'>
                                            {det.segment[0].toFixed(1)}s - {det.segment[1].toFixed(1)}s
                                        </span>
                                        <span className='DetLabel'>{det.label}</span>
                                        <span className='DetScore'>
                                            {(det.score * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                ))}
                                {dets.length > 10 && (
                                    <div className='DetOverflow'>+{dets.length - 10} more</div>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {error && (
                <div className='ErrorState'>
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                        <circle cx="8" cy="8" r="6.5" stroke="currentColor" strokeWidth="1.2" />
                        <path d="M8 5v3.5M8 10.5v.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
                    </svg>
                    {error}
                </div>
            )}
        </div>
    );
};

export default InferPanel;
