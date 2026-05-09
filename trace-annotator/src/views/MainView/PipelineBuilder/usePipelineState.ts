import { useCallback, useEffect, useRef, useState } from 'react';
import { API_URL } from '../../../config';

export type PipelineStepId = 'prep' | 'train' | 'test' | 'infer';
export type StepStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'skipped';
export type TrainCacheMode = 'cached_video' | 'virtual';
export type ResourceProfileId = 'low' | 'balanced' | 'high';
export type ResourceStageId = 'train' | 'test' | 'infer';
export type TrainCacheResolution = 112 | 144 | 192 | 224;

export interface StepState {
    status: StepStatus;
    jobId: string | null;
    error: string | null;
}

export interface PairSelection {
    // Browse cursor (last folder visited in the picker). Doubles as the
    // backend's `work_dir` so the prep step's output `model_<ts>/` folder
    // lands somewhere sensible. In multi-folder train/test, pairs may live
    // outside this folder — `pairs[]` carries fully-qualified absolute paths.
    folder: string;
    stems: string[]; // single-folder mode only: pair-group keys within `folder`
    pairs: string[]; // multi-folder mode: "<absVideo>=<absCsv>"; single-folder: "video=csv" basenames
    csvByStem: Record<string, string>; // single-folder mode only: per-stem CSV override
}

export interface PipelineConfig {
    steps: {
        train: boolean;
        test: boolean;
        infer: boolean;
    };
    datasetSelection: PairSelection;       // train (and chained-test) source pairs
    testDatasetSelection: PairSelection;   // test pairs when train is off
    inputSelection: PairSelection;         // infer videos (CSV side ignored)
    modelSize: 'small' | 'large';
    modelLoadPath: string;   // test/infer when Train is disabled: model_ timestamp folder
    trainCacheMode: TrainCacheMode;
    trainCacheResolution: TrainCacheResolution;
    resources: Record<ResourceStageId, ResourceSettings>;
    inferAnnotatedVideo: boolean;
    inferThreshold: number;

    // Training schedule
    totalEpochs: number;
    valStartEpoch: number;
    valInterval: number;
    trainValRatio: number;
}

export interface PipelineCliCommand {
    argv: string[];
    command: string;
    warnings: string[];
}

export type PipelineCliCommandStatus = 'idle' | 'generating' | 'ready' | 'failed';

export interface TrainResourceProfile {
    name: 'Low' | 'Balanced' | 'High';
    id: ResourceProfileId;
    batch_size?: number;
    num_workers: number;
    decode_threads: number;
    prefetch_factor: number;
}

export interface ResourceSettings {
    profile: ResourceProfileId;
    batchSize: number;
    numWorkers: number;
    decodeThreads: number;
    prefetchFactor: number;
    advancedOpen: boolean;
}

export interface TrainingResourceEstimate {
    summary: {
        pair_count: number;
        source_mb: number;
        duration_sec: number;
        frames: number;
        clip_frames: number;
        total_windows: number;
        annotated_clip_count: number;
        annotation_count: number;
        behavior_count: number;
        coverage: number;
        max_width: number;
        max_height: number;
        cache_resolution: number;
    };
    behaviors: Array<{ name: string; count: number }>;
    cache_options: Array<{
        id: TrainCacheMode;
        label: string;
        disk_mb: number;
        ram_mb: number;
        vram_mb: number;
        recommended: boolean;
        detail: string;
    }>;
    model_options: Array<{
        id: 'small' | 'large';
        label: string;
        config_name: string;
        ram_mb: number;
        vram_mb: number;
        recommended: boolean;
        detail: string;
    }>;
    resolution_options: Array<{
        id: TrainCacheResolution;
        label: string;
        small_vram_mb: number;
        large_vram_mb: number;
        recommended: boolean;
        detail: string;
    }>;
    resource_profiles: Array<TrainResourceProfile & {
        ram_mb: number;
        vram_mb: number;
        recommended: boolean;
        detail: string;
    }>;
    recommendations: {
        cache_mode: TrainCacheMode;
        cache_reason: string;
        model_size: 'small' | 'large';
        resource_profile: ResourceProfileId;
        notes: string[];
    };
}

export const RESOURCE_PROFILES: TrainResourceProfile[] = [
    { id: 'low', name: 'Low', batch_size: 1, num_workers: 2, decode_threads: 1, prefetch_factor: 2 },
    { id: 'balanced', name: 'Balanced', batch_size: 4, num_workers: 4, decode_threads: 2, prefetch_factor: 2 },
    { id: 'high', name: 'High', batch_size: 8, num_workers: 8, decode_threads: 2, prefetch_factor: 2 },
];

export function profileById(id: ResourceProfileId): TrainResourceProfile {
    return RESOURCE_PROFILES.find((p) => p.id === id) || RESOURCE_PROFILES[1];
}

export function resourceSettingsFromProfile(
    id: ResourceProfileId,
    advancedOpen = false,
): ResourceSettings {
    const profile = profileById(id);
    return {
        profile: profile.id,
        batchSize: profile.batch_size || 4,
        numWorkers: profile.num_workers,
        decodeThreads: profile.decode_threads,
        prefetchFactor: profile.prefetch_factor,
        advancedOpen,
    };
}

function intInRange(value: number, fallback: number, min: number, max: number): number {
    if (!Number.isFinite(value)) return fallback;
    return Math.min(max, Math.max(min, Math.round(value)));
}

function resourceValidationError(stage: string, settings: ResourceSettings, includeBatch: boolean): string | null {
    if (includeBatch && (!Number.isFinite(settings.batchSize) || settings.batchSize < 1)) {
        return `${stage} batch size must be at least 1.`;
    }
    if (!Number.isFinite(settings.numWorkers) || settings.numWorkers < 1) {
        return `${stage} workers must be at least 1.`;
    }
    if (!Number.isFinite(settings.decodeThreads) || settings.decodeThreads < 1) {
        return `${stage} decode threads must be at least 1.`;
    }
    if (!Number.isFinite(settings.prefetchFactor) || settings.prefetchFactor < 1) {
        return `${stage} prefetch must be at least 1.`;
    }
    return null;
}

function resourceCfgOptions(
    settings: ResourceSettings,
    splits: Array<'train' | 'val' | 'test'>,
    includeBatch = false,
): Record<string, any> {
    const fallback = profileById(settings.profile);
    const batchSize = intInRange(settings.batchSize, fallback.batch_size || 4, 1, 256);
    const numWorkers = intInRange(settings.numWorkers, fallback.num_workers, 1, 64);
    const decodeThreads = intInRange(settings.decodeThreads, fallback.decode_threads, 1, 16);
    const prefetchFactor = intInRange(settings.prefetchFactor, fallback.prefetch_factor, 1, 16);
    const options: Record<string, any> = {};
    for (const split of splits) {
        if (includeBatch) options[`solver.${split}.batch_size`] = batchSize;
        options[`solver.${split}.num_workers`] = numWorkers;
        options[`solver.${split}.prefetch_factor`] = prefetchFactor;
        options[`solver.${split}.persistent_workers`] = true;
        options[`dataset.${split}.pipeline.1.num_threads`] = decodeThreads;
    }
    return options;
}

function profileCfgOptions(
    settings: ResourceSettings,
    resolution: TrainCacheResolution,
    modelSize: 'small' | 'large',
): Record<string, any> {
    const size = [resolution, resolution];
    const options: Record<string, any> = {
        'dataset.train.pipeline.1.resize': size,
        'dataset.val.pipeline.1.resize': size,
        'dataset.test.pipeline.1.resize': size,
        'dataset.val.pipeline.4.scale': size,
        'dataset.test.pipeline.4.scale': size,
    };
    Object.assign(options, resourceCfgOptions(settings, ['train', 'val', 'test']));
    if (modelSize === 'large') {
        options['dataset.train.pipeline.5.scale'] = size;
    } else {
        options['dataset.train.pipeline.4.scale'] = size;
    }
    return options;
}

function evalResourceCfgOptions(settings: ResourceSettings): Record<string, any> {
    return resourceCfgOptions(settings, ['test'], true);
}

const EMPTY_SELECTION: PairSelection = { folder: '', stems: [], pairs: [], csvByStem: {} };

function workDirFromSelection(selection: PairSelection): string {
    const firstPair = selection.pairs[0] || '';
    const firstPairVideo = firstPair.split('=')[0] || '';
    const firstPairFolder = firstPairVideo.includes('/')
        ? firstPairVideo.substring(0, firstPairVideo.lastIndexOf('/'))
        : '';
    return selection.folder || firstPairFolder;
}

export interface ResolvedModel {
    model_dir: string;
    checkpoint_path: string;
    class_map_path: string;
    config_path: string;
    classes: string[];
}

interface SubmittedJob {
    job_id: string;
    run_id?: string | null;
}

export interface PipelineState {
    config: PipelineConfig;
    setConfig: (update: Partial<PipelineConfig> | ((prev: PipelineConfig) => Partial<PipelineConfig>)) => void;
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
    trainingEstimate: TrainingResourceEstimate | null;
    trainingEstimateStatus: 'idle' | 'loading' | 'ready' | 'error';
    trainingEstimateError: string | null;

    // Validation
    canRun: boolean;
    validationError: string | null;

    // CLI command generation
    cliCommand: PipelineCliCommand | null;
    cliCommandStatus: PipelineCliCommandStatus;
    cliCommandError: string | null;
    generateCliCommand: () => Promise<void>;
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

async function submitJob(type: string, body: Record<string, any>): Promise<SubmittedJob> {
    const res = await fetch(`${API_URL}/api/jobs/${type}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Job submission failed (${res.status})`);
    }
    return await res.json();
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
        body: JSON.stringify({ model_dir: modelPath }),
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

function selectionToSpec(selection: PairSelection) {
    return {
        folder: selection.folder,
        pairs: selection.pairs,
        stems: selection.stems,
        csv_by_stem: selection.csvByStem,
    };
}

function resourceToSpec(settings: ResourceSettings) {
    return {
        profile: settings.profile,
        batch_size: settings.batchSize,
        num_workers: settings.numWorkers,
        decode_threads: settings.decodeThreads,
        prefetch_factor: settings.prefetchFactor,
    };
}

function pipelineSpecFromConfig(config: PipelineConfig) {
    const prepCacheMode = config.steps.test ? 'cached_video' : config.trainCacheMode;
    return {
        steps: {
            train: config.steps.train,
            extra_test: config.steps.test,
            infer: config.steps.infer,
        },
        train_selection: selectionToSpec(config.datasetSelection),
        test_selection: selectionToSpec(config.testDatasetSelection),
        input_selection: selectionToSpec(config.inputSelection),
        model_size: config.modelSize,
        model_dir: config.modelLoadPath,
        cache_mode: prepCacheMode,
        cache_resolution: config.trainCacheResolution,
        resource_profile: config.resources.train.profile,
        resources: {
            train: resourceToSpec(config.resources.train),
            test: resourceToSpec(config.resources.test),
            infer: resourceToSpec(config.resources.infer),
        },
        epochs: config.totalEpochs,
        val_start_epoch: config.valStartEpoch,
        val_interval: config.valInterval,
        train_ratio: config.trainValRatio,
        annotated_video: config.inferAnnotatedVideo,
        threshold: config.inferThreshold,
    };
}

export function usePipelineState(): PipelineState {
    const [config, setConfigState] = useState<PipelineConfig>({
        steps: { train: true, test: false, infer: false },
        datasetSelection: { ...EMPTY_SELECTION },
        testDatasetSelection: { ...EMPTY_SELECTION },
        inputSelection: { ...EMPTY_SELECTION },
        modelSize: 'small',
        modelLoadPath: '',
        trainCacheMode: 'cached_video',
        trainCacheResolution: 144,
        resources: {
            train: resourceSettingsFromProfile('balanced'),
            test: resourceSettingsFromProfile('balanced'),
            infer: resourceSettingsFromProfile('balanced'),
        },
        inferAnnotatedVideo: false,
        inferThreshold: 0.3,
        totalEpochs: 100,
        valStartEpoch: 50,
        valInterval: 10,
        trainValRatio: 0.8,
    });

    const [pipelineStatus, setPipelineStatus] = useState<PipelineState['pipelineStatus']>('idle');
    const [stepStates, setStepStates] = useState<Record<PipelineStepId, StepState>>(makeInitialSteps);
    const [currentStep, setCurrentStep] = useState<PipelineStepId | null>(null);
    const [metrics, setMetrics] = useState<Record<string, any> | null>(null);
    const [predictions, setPredictions] = useState<Record<string, any[]> | null>(null);
    const [trainingEstimate, setTrainingEstimate] = useState<TrainingResourceEstimate | null>(null);
    const [trainingEstimateStatus, setTrainingEstimateStatus] = useState<PipelineState['trainingEstimateStatus']>('idle');
    const [trainingEstimateError, setTrainingEstimateError] = useState<string | null>(null);
    const [cliCommand, setCliCommand] = useState<PipelineCliCommand | null>(null);
    const [cliCommandStatus, setCliCommandStatus] = useState<PipelineCliCommandStatus>('idle');
    const [cliCommandError, setCliCommandError] = useState<string | null>(null);

    const abortRef = useRef<AbortController | null>(null);
    const configsRef = useRef<{ path: string; name: string }[]>([]);

    // Fetch configs on mount
    useEffect(() => {
        fetchConfigs().then((c) => { configsRef.current = c; });
    }, []);

    const setConfig = useCallback((update: Partial<PipelineConfig> | ((prev: PipelineConfig) => Partial<PipelineConfig>)) => {
        setCliCommand(null);
        setCliCommandStatus('idle');
        setCliCommandError(null);
        setConfigState((prev) => ({
            ...prev,
            ...(typeof update === 'function' ? update(prev) : update),
        }));
    }, []);

    const toggleStep = useCallback((step: 'train' | 'test' | 'infer') => {
        setCliCommand(null);
        setCliCommandStatus('idle');
        setCliCommandError(null);
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
    const needsTestDataset = config.steps.test && !config.steps.train;
    const needsModelLoad = !config.steps.train && (config.steps.test || config.steps.infer);
    const needsInput = config.steps.infer;
    const hasAnyStep = config.steps.train || config.steps.test || config.steps.infer;

    const trainSel = config.datasetSelection;
    const testSel = config.testDatasetSelection;
    const inferSel = config.inputSelection;

    let validationError: string | null = null;
    if (!hasAnyStep) validationError = 'Enable at least one model run step.';
    else if (needsTrainDataset && trainSel.pairs.length < 1) validationError = 'Pick at least one training pair.';
    else if (needsTestDataset && testSel.pairs.length < 1) validationError = 'Pick at least one test pair.';
    else if (needsModelLoad && !config.modelLoadPath) validationError = 'Model load folder is required when not training.';
    else if (needsInput && !inferSel.folder) validationError = 'Prediction input folder is required.';
    else if (needsInput && inferSel.stems.length < 1) validationError = 'Pick at least one inference video.';
    else if (needsInput && (!Number.isFinite(config.inferThreshold) || config.inferThreshold < 0 || config.inferThreshold > 1)) {
        validationError = 'Prediction threshold must be between 0 and 1.';
    }
    else if (config.steps.train) {
        validationError = resourceValidationError('Train', config.resources.train, false);
    }
    if (!validationError && config.steps.test) {
        validationError = resourceValidationError('Test', config.resources.test, true);
    }
    if (!validationError && config.steps.infer) {
        validationError = resourceValidationError('Prediction', config.resources.infer, true);
    }
    if (!validationError && config.steps.train) {
        if (!Number.isFinite(config.totalEpochs) || config.totalEpochs < 1) {
            validationError = 'Total epochs must be at least 1.';
        } else if (!Number.isFinite(config.valStartEpoch) || config.valStartEpoch < 0 || config.valStartEpoch >= config.totalEpochs) {
            validationError = 'Validation start epoch must be between 0 and total epochs - 1.';
        } else if (!Number.isFinite(config.valInterval) || config.valInterval < 1) {
            validationError = 'Validation interval must be at least 1.';
        } else if (!Number.isFinite(config.trainValRatio) || config.trainValRatio <= 0 || config.trainValRatio >= 1) {
            validationError = 'Train/val ratio must be between 0 and 1 (exclusive).';
        }
    }

    const canRun = pipelineStatus === 'idle' && !validationError;

    const generateCliCommand = useCallback(async () => {
        if (validationError) return;
        setCliCommandStatus('generating');
        setCliCommandError(null);
        setCliCommand(null);

        try {
            const res = await fetch(`${API_URL}/api/pipeline/command`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(pipelineSpecFromConfig(config)),
            });

            if (!res.ok) {
                const body = await res.json().catch(() => ({}));
                const raw = body.detail;
                const message = typeof raw === 'string'
                    ? raw
                    : (raw?.message || `Command generation failed (${res.status})`);
                setCliCommandStatus('failed');
                setCliCommandError(message);
                return;
            }

            const result = await res.json() as PipelineCliCommand;
            setCliCommand(result);
            setCliCommandStatus('ready');
        } catch (err: any) {
            setCliCommandStatus('failed');
            setCliCommandError(err?.message || 'Command generation failed.');
        }
    }, [config, validationError]);

    const updateStep = (id: PipelineStepId, update: Partial<StepState>) => {
        setStepStates((prev) => ({
            ...prev,
            [id]: { ...prev[id], ...update },
        }));
    };

    const getConfigPath = (): string => {
        const target = config.modelSize;
        const match = configsRef.current.find((c) => c.name === target);
        return match?.path || `configs/${target}.py`;
    };

    useEffect(() => {
        if (!config.steps.train || config.datasetSelection.pairs.length < 1) {
            setTrainingEstimate(null);
            setTrainingEstimateStatus('idle');
            setTrainingEstimateError(null);
            return undefined;
        }

        const workDir = workDirFromSelection(config.datasetSelection);
        if (!workDir) {
            setTrainingEstimate(null);
            setTrainingEstimateStatus('idle');
            setTrainingEstimateError(null);
            return undefined;
        }

        const pairs = config.datasetSelection.pairs;
        const controller = new AbortController();
        setTrainingEstimateStatus('loading');
        setTrainingEstimateError(null);

        fetch(`${API_URL}/api/training/estimate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                work_dir: workDir,
                explicit_pairs: pairs,
                clip_frames: 768,
                cache_resolution: config.trainCacheResolution,
            }),
            signal: controller.signal,
        })
            .then(async (res) => {
                if (!res.ok) {
                    const detail = await res.json().catch(() => ({}));
                    throw new Error(detail.detail || `Estimate failed (${res.status})`);
                }
                return await res.json() as TrainingResourceEstimate;
            })
            .then((estimate) => {
                setTrainingEstimate(estimate);
                setTrainingEstimateStatus('ready');
            })
            .catch((err) => {
                if (controller.signal.aborted) return;
                setTrainingEstimate(null);
                setTrainingEstimateStatus('error');
                setTrainingEstimateError(err?.message || 'Could not estimate training resources.');
            });

        return () => controller.abort();
    }, [config.steps.train, config.datasetSelection, config.trainCacheResolution]);

    const run = useCallback(async () => {
        if (!canRun) return;

        const abort = new AbortController();
        abortRef.current = abort;

        setPipelineStatus('running');
        setStepStates(makeInitialSteps());
        setMetrics(null);
        setPredictions(null);

        const runSteps = (['train', 'test', 'infer'] as const).filter((step) => config.steps[step]);
        let runId: string | null = null;
        const submitRunJob = async (type: string, body: Record<string, any>): Promise<string> => {
            const job = await submitJob(type, {
                ...body,
                ...(runId ? { run_id: runId } : {}),
                run_steps: runSteps,
            });
            runId = job.run_id || job.job_id;
            return job.job_id;
        };
        let prepResult: { model_dir: string; dataset_json: string; classmap_path: string } | null = null;
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
                const prepCacheMode = config.steps.test ? 'cached_video' : config.trainCacheMode;
                // Pick the selection to prep: train wins over test-only.
                const prepSel = config.steps.train
                    ? config.datasetSelection
                    : config.testDatasetSelection;
                // multi-folder pairs may sit outside `prepSel.folder` (the browse
                // cursor). Backend uses `work_dir` only as the default location
                // for the output `model_<ts>/` directory and to resolve any
                // relative pair specs. Fall back to the first pair's folder if
                // the user has only ever picked from one place via the manifest.
                const workDir = workDirFromSelection(prepSel);
                const prepJobId = await submitRunJob('prep', {
                    work_dir: workDir,
                    train_ratio: config.trainValRatio,
                    cache_mode: prepCacheMode,
                    cache_resolution: config.trainCacheResolution,
                    cache_workers: config.steps.test
                        ? config.resources.test.numWorkers
                        : config.resources.train.numWorkers,
                    explicit_pairs: prepSel.pairs,
                });
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

                const cfgOptions: Record<string, any> = {
                    'scheduler.max_epoch': config.totalEpochs,
                    'workflow.end_epoch': config.totalEpochs,
                    'workflow.val_start_epoch': config.valStartEpoch,
                    'workflow.val_eval_interval': config.valInterval,
                };
                Object.assign(cfgOptions, profileCfgOptions(
                    config.resources.train,
                    config.trainCacheResolution,
                    config.modelSize,
                ));

                const trainBody: Record<string, any> = {
                    config_path: getConfigPath(),
                    cfg_options: cfgOptions,
                };
                if (prepResult) {
                    trainBody.model_dir = prepResult.model_dir;
                    trainBody.dataset_dir = prepResult.model_dir;
                    trainBody.annotation_path = prepResult.dataset_json;
                    trainBody.class_map = prepResult.classmap_path;
                }

                const trainJobId = await submitRunJob('train', trainBody);
                updateStep('train', { jobId: trainJobId });

                const trainStatus = await waitForJob(trainJobId, abort.signal);
                if (trainStatus !== 'completed') {
                    updateStep('train', { status: trainStatus as StepStatus, error: `Train ${trainStatus}` });
                    setPipelineStatus(trainStatus === 'cancelled' ? 'cancelled' : 'failed');
                    return;
                }
                updateStep('train', { status: 'completed' });

                activeModel = await resolveModel(prepResult!.model_dir);
            }

            // If no training, resolve the user-provided model path
            if (!config.steps.train && (config.steps.test || config.steps.infer)) {
                activeModel = await resolveModel(config.modelLoadPath);
            }

            // ── Step 3: Test ──
            if (config.steps.test && activeModel) {
                setCurrentStep('test');
                updateStep('test', { status: 'running' });

                const testBody: Record<string, any> = {
                    model_dir: activeModel.model_dir,
                    auto_tune: false,
                    cfg_options: evalResourceCfgOptions(config.resources.test),
                };
                if (prepResult) {
                    testBody.dataset_dir = prepResult.model_dir;
                    testBody.annotation_path = prepResult.dataset_json;
                }

                const testJobId = await submitRunJob('test', testBody);
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

            // Step 4: Predict
            if (config.steps.infer && activeModel) {
                setCurrentStep('infer');
                updateStep('infer', { status: 'running' });

                const inferJobId = await submitRunJob('infer', {
                    model_dir: activeModel.model_dir,
                    input: config.inputSelection.folder,
                    included_stems: config.inputSelection.stems,
                    annotated_video: config.inferAnnotatedVideo,
                    threshold: config.inferThreshold,
                    auto_tune: false,
                    cfg_options: evalResourceCfgOptions(config.resources.infer),
                });
                updateStep('infer', { jobId: inferJobId });

                const inferStatus = await waitForJob(inferJobId, abort.signal);
                if (inferStatus !== 'completed') {
                    updateStep('infer', { status: inferStatus as StepStatus, error: `Predict ${inferStatus}` });
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
        setCliCommand(null);
        setCliCommandStatus('idle');
        setCliCommandError(null);
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
        trainingEstimate,
        trainingEstimateStatus,
        trainingEstimateError,
        canRun,
        validationError,
        cliCommand,
        cliCommandStatus,
        cliCommandError,
        generateCliCommand,
    };
}
