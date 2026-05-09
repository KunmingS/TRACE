import React, { useEffect, useState } from 'react';
import './TestPanel.scss';
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

interface PrepResult {
    model_dir: string;
    dataset_json: string;
    classmap_path: string;
}

const TestPanel: React.FC = () => {
    const [datasetPath, setDatasetPath] = useState('');
    const [modelPath, setModelPath] = useState('');
    const [models, setModels] = useState<ModelInfo[]>([]);
    const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null);
    const [metrics, setMetrics] = useState<Record<string, any> | null>(null);
    const [error, setError] = useState('');

    const prepRunner = useJobRunner();
    const testRunner = useJobRunner();

    const isBusy = prepRunner.status === 'running' || prepRunner.status === 'submitting'
        || testRunner.status === 'running' || testRunner.status === 'submitting';

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

    // Fetch metrics when test completes
    useEffect(() => {
        if (testRunner.status === 'completed') {
            testRunner.fetchArtifact<Record<string, any>>('metrics.json').then((m) => {
                if (m) setMetrics(m);
            });
        } else if (testRunner.status === 'failed') {
            setError(testRunner.error || 'Evaluation failed.');
        }
    }, [testRunner.status]);

    useEffect(() => {
        if (prepRunner.status === 'failed') {
            setError(prepRunner.error || 'Dataset preparation failed.');
        }
    }, [prepRunner.status]);

    const handleSelectModel = (model: ModelInfo) => {
        setModelPath(model.path);
        setSelectedModel(model);
    };

    const handleRun = async () => {
        if (!datasetPath) { setError('Enter a dataset path.'); return; }
        if (!modelPath && !selectedModel) { setError('Select or enter a model path.'); return; }
        setError('');
        setMetrics(null);

        try {
            // First prepare dataset
            await prepRunner.submit('prep', { work_dir: datasetPath });

            // Wait for prep to complete
            await new Promise<void>((resolve, reject) => {
                const check = setInterval(async () => {
                    const res = await fetch(`${API_URL}/api/jobs/${prepRunner.jobId}`);
                    const j = await res.json();
                    if (j.status === 'completed') { clearInterval(check); resolve(); }
                    else if (j.status === 'failed' || j.status === 'cancelled') {
                        clearInterval(check); reject(new Error(`Prep ${j.status}`));
                    }
                }, 2000);
            });

            // Get prep result
            const prepArtifact = await prepRunner.fetchArtifact<PrepResult>('prep_result.json');

            const body: Record<string, any> = {
                model_dir: modelPath,
            };
            if (prepArtifact) {
                body.dataset_dir = prepArtifact.model_dir;
                body.annotation_path = prepArtifact.dataset_json;
            }

            await testRunner.submit('test', body);
        } catch (err: any) {
            setError(err.message);
        }
    };

    return (
        <div className='TestPanel'>
            <div className='PanelHeader'>
                <h2>Evaluate Model</h2>
                <p>Run evaluation on a dataset with a trained model to measure detection accuracy.</p>
            </div>

            <div className='FormSection'>
                <label className='FieldLabel'>Dataset Path</label>
                <PathPicker
                    value={datasetPath}
                    onChange={setDatasetPath}
                    placeholder='/path/to/dataset'
                    disabled={isBusy}
                />
            </div>

            <div className='FormSection'>
                <label className='FieldLabel'>Model</label>
                {models.length > 0 && (
                    <div className='ModelList'>
                        {models.map((m) => (
                            <button
                                key={m.path}
                                className={`ModelCard ${selectedModel?.path === m.path ? 'active' : ''}`}
                                onClick={() => handleSelectModel(m)}
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
                    placeholder='/path/to/model (contains best.pth + classmap.txt)'
                    disabled={isBusy}
                />
                {selectedModel && (
                    <div className='ModelDetail'>
                        Classes: {selectedModel.classes.join(', ')}
                    </div>
                )}
            </div>

            <button
                className='ActionBtn primary'
                onClick={handleRun}
                disabled={!datasetPath || !modelPath || isBusy}
                type='button'
            >
                Run Evaluation
            </button>

            {(prepRunner.jobId && prepRunner.status === 'running') && (
                <LogViewer jobId={prepRunner.jobId} onCancel={() => prepRunner.cancel()} />
            )}
            {testRunner.jobId && (
                <LogViewer jobId={testRunner.jobId} onCancel={() => testRunner.cancel()} />
            )}

            {metrics && (
                <div className='MetricsCard'>
                    <div className='MetricsTitle'>Results</div>
                    {metrics.average_mAP != null && (
                        <div className='MetricRow primary'>
                            <span>Average mAP</span>
                            <span className='MetricValue'>
                                {(metrics.average_mAP * 100).toFixed(2)}%
                            </span>
                        </div>
                    )}
                    {metrics.mAP != null && !metrics.average_mAP && (
                        <div className='MetricRow primary'>
                            <span>mAP</span>
                            <span className='MetricValue'>
                                {(metrics.mAP * 100).toFixed(2)}%
                            </span>
                        </div>
                    )}
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

export default TestPanel;
