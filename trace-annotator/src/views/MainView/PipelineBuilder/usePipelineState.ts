import { useCallback, useEffect, useRef, useState } from 'react';
import { API_URL } from '../../../config';

export type PipelineStepId = 'prep' | 'train' | 'test' | 'infer';
export type StepStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'skipped';

export interface StepState {
    status: StepStatus;
    jobId: string | null;
    error: string | null;
}

export interface PipelineConfig {
    steps: {
        train: boolean;
        test: boolean;
        infer: boolean;
    };
    datasetPath: string;
    testDatasetPath: string; // separate test dataset (falls back to datasetPath)
    modelSize: 'small' | 'large';
    modelPath: string;      // for test/infer without train
    inputPath: string;      // for infer
}

export interface ResolvedModel {
    checkpoint_path: string;
    class_map_path: string;
    config_path: string;
    classes: string[];
}

export interface PipelineState {
    config: PipelineConfig;
    setConfig: (update: Partial<PipelineConfig>) => void;
    toggleStep: (step: 'train' | 'test' | 'infer') => void;

    pipelineStatus: 'idle' | 'running' | 'completed' | 'failed' | 'cancelled';
    stepStates: Record<PipelineStepId, StepState>;
    activeJobId: string | null;
    currentStep: PipelineStepId | null;

    run: () => Promise<void>;
    cancel: () => Promise<void>;
    reset: () => void;

    // Results
    metrics: Record<string, any> | null;
    predictions: Record<string, any[]> | null;

    // Validation
    canRun: boolean;
    validationError: string | null;
}

const POLL_MS = 2500;

const initialStepState: StepState = { status: 'pending', jobId: null, error: null };

function makeInitialSteps(): Record<PipelineStepId, StepState> {
    return {
        prep: { ...initialStepState },
        train: { ...initialStepState },
        test: { ...initialStepState },
        infer: { ...initialStepState },
    };
}

async function submitJob(type: string, body: Record<string, any>): Promise<string> {
    const res = await fetch(`${API_URL}/api/jobs/${type}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Job submission failed (${res.status})`);
    }
    const job = await res.json();
    return job.job_id;
}

async function waitForJob(jobId: string, signal: AbortSignal): Promise<'completed' | 'failed' | 'cancelled'> {
    return new Promise((resolve, reject) => {
        const poll = setInterval(async () => {
            if (signal.aborted) {
                clearInterval(poll);
                reject(new DOMException('Aborted', 'AbortError'));
                return;
            }
            try {
                const res = await fetch(`${API_URL}/api/jobs/${jobId}`);
                if (!res.ok) return;
                const job = await res.json();
                if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') {
                    clearInterval(poll);
                    resolve(job.status);
                }
            } catch {
                // keep polling
            }
        }, POLL_MS);

        signal.addEventListener('abort', () => {
            clearInterval(poll);
            reject(new DOMException('Aborted', 'AbortError'));
        });
    });
}

async function fetchArtifact<T>(jobId: string, filename: string): Promise<T | null> {
    try {
        const res = await fetch(`${API_URL}/api/jobs/${jobId}/artifacts/${filename}`);
        if (!res.ok) return null;
        return await res.json() as T;
    } catch {
        return null;
    }
}

async function resolveModel(modelPath: string): Promise<ResolvedModel> {
    const res = await fetch(`${API_URL}/api/resolve-model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_path: modelPath }),
    });
    if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Cannot resolve model at ${modelPath}`);
    }
    return await res.json();
}

async function fetchConfigs(): Promise<{ path: string; name: string }[]> {
    try {
        const res = await fetch(`${API_URL}/api/configs`);
        if (!res.ok) return [];
        return await res.json();
    } catch {
        return [];
    }
}

export function usePipelineState(): PipelineState {
    const [config, setConfigState] = useState<PipelineConfig>({
        steps: { train: true, test: true, infer: false },
        datasetPath: '',
        testDatasetPath: '',
        modelSize: 'small',
        modelPath: '',
        inputPath: '',
    });

    const [pipelineStatus, setPipelineStatus] = useState<PipelineState['pipelineStatus']>('idle');
    const [stepStates, setStepStates] = useState<Record<PipelineStepId, StepState>>(makeInitialSteps);
    const [currentStep, setCurrentStep] = useState<PipelineStepId | null>(null);
    const [metrics, setMetrics] = useState<Record<string, any> | null>(null);
    const [predictions, setPredictions] = useState<Record<string, any[]> | null>(null);

    const abortRef = useRef<AbortController | null>(null);
    const configsRef = useRef<{ path: string; name: string }[]>([]);

    // Fetch configs on mount
    useEffect(() => {
        fetchConfigs().then((c) => { configsRef.current = c; });
    }, []);

    const setConfig = useCallback((update: Partial<PipelineConfig>) => {
        setConfigState((prev) => ({ ...prev, ...update }));
    }, []);

    const toggleStep = useCallback((step: 'train' | 'test' | 'infer') => {
        setConfigState((prev) => ({
            ...prev,
            steps: { ...prev.steps, [step]: !prev.steps[step] },
        }));
    }, []);

    const activeJobId = (() => {
        if (currentStep && stepStates[currentStep]?.jobId) {
            return stepStates[currentStep].jobId;
        }
        return null;
    })();

    // Validation
    const needsTrainDataset = config.steps.train;
    const needsTestDataset = config.steps.test;
    const effectiveTestDataset = config.testDatasetPath || config.datasetPath;
    const needsModel = !config.steps.train && (config.steps.test || config.steps.infer);
    const effectiveInputPath = config.inputPath || config.datasetPath;
    const needsInput = config.steps.infer;
    const hasAnyStep = config.steps.train || config.steps.test || config.steps.infer;

    let validationError: string | null = null;
    if (!hasAnyStep) validationError = 'Enable at least one pipeline step.';
    else if (needsTrainDataset && !config.datasetPath) validationError = 'Training dataset path is required.';
    else if (needsTestDataset && !effectiveTestDataset) validationError = 'Test dataset path is required.';
    else if (needsModel && !config.modelPath) validationError = 'Model path is required when not training.';
    else if (needsInput && !effectiveInputPath) validationError = 'Input video path is required for inference.';

    const canRun = pipelineStatus === 'idle' && !validationError;

    const updateStep = (id: PipelineStepId, update: Partial<StepState>) => {
        setStepStates((prev) => ({
            ...prev,
            [id]: { ...prev[id], ...update },
        }));
    };

    const getConfigPath = (): string => {
        const target = config.modelSize === 'small' ? 'tridet_small' : 'tridet_large';
        const match = configsRef.current.find((c) => c.name === target);
        return match?.path || `configs/tridet/${target}.py`;
    };

    const run = useCallback(async () => {
        if (!canRun) return;

        const abort = new AbortController();
        abortRef.current = abort;

        setPipelineStatus('running');
        setStepStates(makeInitialSteps());
        setMetrics(null);
        setPredictions(null);

        const expId = Math.floor(Date.now() / 1000) % 100000;
        let prepResult: { clips_dir: string; json_path: string; classmap_path: string } | null = null;
        let activeModel: ResolvedModel | null = null;

        // Mark skipped steps
        if (!config.steps.train) updateStep('train', { status: 'skipped' });
        if (!config.steps.test) updateStep('test', { status: 'skipped' });
        if (!config.steps.infer) updateStep('infer', { status: 'skipped' });

        try {
            // ── Step 1: Prep dataset (if train or test enabled) ──
            if (config.steps.train || config.steps.test) {
                setCurrentStep('prep');
                updateStep('prep', { status: 'running' });
                const prepJobId = await submitJob('prep', { dataset_path: config.datasetPath });
                updateStep('prep', { jobId: prepJobId });

                const prepStatus = await waitForJob(prepJobId, abort.signal);
                if (prepStatus !== 'completed') {
                    updateStep('prep', { status: prepStatus as StepStatus, error: `Prep ${prepStatus}` });
                    setPipelineStatus(prepStatus === 'cancelled' ? 'cancelled' : 'failed');
                    return;
                }
                updateStep('prep', { status: 'completed' });
                prepResult = await fetchArtifact(prepJobId, 'prep_result.json');
            } else {
                updateStep('prep', { status: 'skipped' });
            }

            // ── Step 2: Train ──
            if (config.steps.train) {
                setCurrentStep('train');
                updateStep('train', { status: 'running' });

                const trainBody: Record<string, any> = {
                    config_path: getConfigPath(),
                    exp_id: expId,
                };
                if (prepResult) {
                    trainBody.dataset_dir = prepResult.clips_dir;
                    trainBody.annotation_path = prepResult.json_path;
                    trainBody.class_map = prepResult.classmap_path;
                }

                const trainJobId = await submitJob('train', trainBody);
                updateStep('train', { jobId: trainJobId });

                const trainStatus = await waitForJob(trainJobId, abort.signal);
                if (trainStatus !== 'completed') {
                    updateStep('train', { status: trainStatus as StepStatus, error: `Train ${trainStatus}` });
                    setPipelineStatus(trainStatus === 'cancelled' ? 'cancelled' : 'failed');
                    return;
                }
                updateStep('train', { status: 'completed' });

                // Resolve the model published by training
                try {
                    // Refresh models list and find the top-level 'model' entry
                    const modelsRes = await fetch(`${API_URL}/api/models`);
                    const models = await modelsRes.json();
                    const topModel = models.find((m: any) => m.label === 'model');
                    if (topModel) {
                        activeModel = {
                            checkpoint_path: `${topModel.path}/best.pth`,
                            class_map_path: `${topModel.path}/classmap.txt`,
                            config_path: topModel.config_path || getConfigPath(),
                            classes: topModel.classes || [],
                        };
                    }
                } catch {
                    // Fall back to convention
                    activeModel = {
                        checkpoint_path: 'model/best.pth',
                        class_map_path: 'model/classmap.txt',
                        config_path: getConfigPath(),
                        classes: [],
                    };
                }
            }

            // If no training, resolve the user-provided model path
            if (!config.steps.train && (config.steps.test || config.steps.infer)) {
                activeModel = await resolveModel(config.modelPath);
            }

            // ── Step 3: Test ──
            if (config.steps.test && activeModel) {
                setCurrentStep('test');
                updateStep('test', { status: 'running' });

                const testBody: Record<string, any> = {
                    config_path: activeModel.config_path,
                    checkpoint: activeModel.checkpoint_path,
                    class_map: activeModel.class_map_path,
                    exp_id: expId,
                };
                if (prepResult) {
                    testBody.dataset_dir = prepResult.clips_dir;
                    testBody.annotation_path = prepResult.json_path;
                }

                const testJobId = await submitJob('test', testBody);
                updateStep('test', { jobId: testJobId });

                const testStatus = await waitForJob(testJobId, abort.signal);
                if (testStatus !== 'completed') {
                    updateStep('test', { status: testStatus as StepStatus, error: `Test ${testStatus}` });
                    setPipelineStatus(testStatus === 'cancelled' ? 'cancelled' : 'failed');
                    return;
                }
                updateStep('test', { status: 'completed' });

                const m = await fetchArtifact<Record<string, any>>(testJobId, 'metrics.json');
                if (m) setMetrics(m);
            }

            // ── Step 4: Infer ──
            if (config.steps.infer && activeModel) {
                setCurrentStep('infer');
                updateStep('infer', { status: 'running' });

                const inferJobId = await submitJob('infer', {
                    config_path: activeModel.config_path,
                    checkpoint: activeModel.checkpoint_path,
                    input: config.inputPath || config.datasetPath,
                    class_map: activeModel.class_map_path,
                    exp_id: expId,
                });
                updateStep('infer', { jobId: inferJobId });

                const inferStatus = await waitForJob(inferJobId, abort.signal);
                if (inferStatus !== 'completed') {
                    updateStep('infer', { status: inferStatus as StepStatus, error: `Infer ${inferStatus}` });
                    setPipelineStatus(inferStatus === 'cancelled' ? 'cancelled' : 'failed');
                    return;
                }
                updateStep('infer', { status: 'completed' });

                const p = await fetchArtifact<{ predictions?: Record<string, any[]> }>(inferJobId, 'predictions.json');
                if (p?.predictions) setPredictions(p.predictions);
            }

            setPipelineStatus('completed');
            setCurrentStep(null);
        } catch (err: any) {
            if (err.name === 'AbortError') {
                setPipelineStatus('cancelled');
            } else {
                setPipelineStatus('failed');
                // Update current step with error
                if (currentStep) {
                    updateStep(currentStep, { status: 'failed', error: err.message });
                }
            }
        }
    }, [canRun, config]);

    const cancel = useCallback(async () => {
        abortRef.current?.abort();
        // Cancel the active job
        const step = currentStep;
        if (step) {
            const jobId = stepStates[step]?.jobId;
            if (jobId) {
                try {
                    await fetch(`${API_URL}/api/jobs/${jobId}/cancel`, { method: 'POST' });
                } catch { /* best-effort */ }
            }
            updateStep(step, { status: 'cancelled' });
        }
        setPipelineStatus('cancelled');
    }, [currentStep, stepStates]);

    const reset = useCallback(() => {
        abortRef.current?.abort();
        setPipelineStatus('idle');
        setStepStates(makeInitialSteps());
        setCurrentStep(null);
        setMetrics(null);
        setPredictions(null);
    }, []);

    return {
        config,
        setConfig,
        toggleStep,
        pipelineStatus,
        stepStates,
        activeJobId,
        currentStep,
        run,
        cancel,
        reset,
        metrics,
        predictions,
        canRun,
        validationError,
    };
}
