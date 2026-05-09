import React, { useEffect, useState } from 'react';
import './TrainPanel.scss';
import PathPicker from '../../Common/PathPicker/PathPicker';
import LogViewer from '../../Common/LogViewer/LogViewer';
import { API_URL } from '../../../config';
import { useJobRunner } from '../../../hooks/useJobRunner';

interface PrepResult {
    model_dir: string;
    dataset_json: string;
    classmap_path: string;
}

const TrainPanel: React.FC = () => {
    const [datasetPath, setDatasetPath] = useState('');
    const [selectedStems, setSelectedStems] = useState<string[]>([]);
    const [explicitPairs, setExplicitPairs] = useState<string[]>([]);
    const [modelSize, setModelSize] = useState<'small' | 'large'>('small');
    const [configs, setConfigs] = useState<{ path: string; name: string }[]>([]);
    const [prepResult, setPrepResult] = useState<PrepResult | null>(null);
    const [error, setError] = useState('');

    const prepRunner = useJobRunner();
    const trainRunner = useJobRunner();

    const isBusy = prepRunner.status === 'running' || prepRunner.status === 'submitting'
        || trainRunner.status === 'running' || trainRunner.status === 'submitting';

    useEffect(() => {
        fetch(`${API_URL}/api/configs`)
            .then((res) => res.json())
            .then(setConfigs)
            .catch(() => {});
    }, []);

    // Watch for prep completion to fetch artifact
    useEffect(() => {
        if (prepRunner.status === 'completed') {
            prepRunner.fetchArtifact<PrepResult>('prep_result.json').then((result) => {
                if (result) setPrepResult(result);
            });
        } else if (prepRunner.status === 'failed') {
            setError(prepRunner.error || 'Dataset preparation failed.');
        }
    }, [prepRunner.status]);

    // Watch for train failure
    useEffect(() => {
        if (trainRunner.status === 'failed') {
            setError(trainRunner.error || 'Training failed.');
        }
    }, [trainRunner.status]);

    const getConfigPath = () => {
        const target = modelSize;
        const match = configs.find((c) => c.name === target);
        return match?.path || `configs/${target}.py`;
    };

    const handlePrepare = async () => {
        if (!datasetPath) {
            setError('Enter a dataset path first.');
            return;
        }
        if (explicitPairs.length < 1) {
            setError('Pick at least one video/CSV pair.');
            return;
        }
        setError('');
        setPrepResult(null);
        trainRunner.reset();

        try {
            await prepRunner.submit('prep', {
                work_dir: datasetPath,
                explicit_pairs: explicitPairs,
            });
        } catch (err: any) {
            setError(err.message);
        }
    };

    const handleTrain = async () => {
        if (!prepResult) {
            setError('Prepare the dataset first.');
            return;
        }
        setError('');

        try {
            await trainRunner.submit('train', {
                config_path: getConfigPath(),
                model_dir: prepResult.model_dir,
                dataset_dir: prepResult.model_dir,
                annotation_path: prepResult.dataset_json,
                class_map: prepResult.classmap_path,
                explicit_pairs: explicitPairs,
            });
        } catch (err: any) {
            setError(err.message);
        }
    };

    const activeJobId = trainRunner.jobId || (prepRunner.status === 'running' ? prepRunner.jobId : null);

    return (
        <div className='TrainPanel'>
            <div className='PanelHeader'>
                <h2>Train Model</h2>
                <p>Prepare a dataset from raw videos and train a temporal action detection model.</p>
            </div>

            <div className='FormSection'>
                <label className='FieldLabel'>
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                        <path d="M1.5 2.5h5l1.5 2h6.5v9h-13z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
                    </svg>
                    Dataset Path
                </label>
                <div className='FieldRow'>
                    <PathPicker
                        value={datasetPath}
                        onChange={(folder) => {
                            setDatasetPath(folder);
                            setSelectedStems([]);
                            setExplicitPairs([]);
                        }}
                        mode='multiPair'
                        selectedStems={selectedStems}
                        onSelectedStemsChange={setSelectedStems}
                        onSelectedPairsChange={setExplicitPairs}
                        requireSourceVideo={true}
                        placeholder='/path/to/dataset (videos + CSVs)'
                        disabled={isBusy}
                    />
                    <button
                        className='ActionBtn secondary'
                        onClick={handlePrepare}
                        disabled={!datasetPath || explicitPairs.length < 1 || isBusy}
                        type='button'
                    >
                        Prepare
                    </button>
                </div>
                {prepResult && (
                    <div className='ReadyBadge'>
                        <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                            <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="1.2" />
                            <path d="M5.5 8l2 2 3.5-4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                        Dataset ready
                        <span className='ReadyDetail'>{prepResult.model_dir}</span>
                    </div>
                )}
            </div>

            {prepRunner.jobId && prepRunner.status === 'running' && (
                <LogViewer jobId={prepRunner.jobId} onCancel={() => prepRunner.cancel()} />
            )}

            <div className='FormSection'>
                <label className='FieldLabel'>
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                        <rect x="2" y="3" width="12" height="10" rx="1.5" stroke="currentColor" strokeWidth="1.2" />
                        <path d="M5 7h6M5 9.5h4" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
                    </svg>
                    Model Size
                </label>
                <div className='SizeToggle'>
                    <button
                        className={`SizeBtn ${modelSize === 'small' ? 'active' : ''}`}
                        onClick={() => setModelSize('small')}
                        type='button'
                    >
                        Small
                    </button>
                    <button
                        className={`SizeBtn ${modelSize === 'large' ? 'active' : ''}`}
                        onClick={() => setModelSize('large')}
                        type='button'
                    >
                        Large
                    </button>
                </div>
            </div>

            <button
                className='ActionBtn primary'
                onClick={handleTrain}
                disabled={!prepResult || isBusy}
                type='button'
            >
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <path d="M5 3l8 5-8 5z" fill="currentColor" />
                </svg>
                Start Training
            </button>

            {trainRunner.jobId && <LogViewer jobId={trainRunner.jobId} onCancel={() => trainRunner.cancel()} />}

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

export default TrainPanel;
