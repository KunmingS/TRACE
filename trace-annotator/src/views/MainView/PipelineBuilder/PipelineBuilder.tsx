import React from 'react';
import './PipelineBuilder.scss';
import PathPicker from '../../Common/PathPicker/PathPicker';
import LogViewer from '../../Common/LogViewer/LogViewer';
import PipelineResults from './PipelineResults';
import JobDashboard from './JobDashboard';
import {
    PipelineState,
    RESOURCE_PROFILES,
    ResourceProfileId,
    ResourceSettings,
    ResourceStageId,
    resourceSettingsFromProfile,
} from './usePipelineState';

interface PipelineBuilderProps {
    pipeline: PipelineState;
}

type StageId = 'train' | 'test' | 'infer';

const STAGE_ORDER: { id: StageId; num: string; label: string; tabLabel: string; summary: string }[] = [
    { id: 'train', num: '01', label: 'Train model', tabLabel: 'Train', summary: 'Fit a model from annotated video pairs.' },
    { id: 'test',  num: '02', label: 'Test model', tabLabel: 'Test', summary: 'Evaluate a model against paired videos.' },
    { id: 'infer', num: '03', label: 'Run predictions', tabLabel: 'Predict', summary: 'Run a model on new video folders.' },
];

const RESOURCE_COPY: Record<ResourceProfileId, { label: string; detail: string; note: string }> = {
    low: { label: 'Low', detail: 'Light CPU use', note: 'Shared machines' },
    balanced: { label: 'Balanced', detail: 'Recommended', note: 'Steady throughput' },
    high: { label: 'High', detail: 'More parallelism', note: 'Dedicated machines' },
};

const RESOURCE_OPTIONS = RESOURCE_PROFILES.map((profile) => ({
    ...profile,
    ...RESOURCE_COPY[profile.id],
}));

const RESOLUTION_OPTIONS = [
    { id: 112, label: '112' },
    { id: 144, label: '144' },
    { id: 192, label: '192' },
    { id: 224, label: '224' },
] as const;

function parseThresholdInput(input: HTMLInputElement): number {
    const value = input.valueAsNumber;
    if (!Number.isFinite(value)) return 0.3;
    return Math.min(1, Math.max(0, value));
}

const CACHE_OPTIONS = [
    { id: 'cached_video', label: 'Cached', detail: '144p clips' },
    { id: 'virtual', label: 'Virtual', detail: 'No clip files' },
] as const;

const MODEL_RESOURCE_FALLBACK = {
    small: { vram_mb: 160, ram_mb: 1200 },
    large: { vram_mb: 1400, ram_mb: 2200 },
} as const;

const PROFILE_RAM_FALLBACK = {
    low: 1700,
    balanced: 2300,
    high: 4100,
} as const;

const RESOLUTION_TRAIN_PEAK_FALLBACK = {
    small: { 112: 1070, 144: 1529, 192: 2430, 224: 3177 },
    large: { 112: 5376, 144: 7147, 192: 10619, 224: 13497 },
} as const;

const formatMb = (value?: number | null): string => {
    const mb = Number(value || 0);
    if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
    return `${Math.max(0, Math.round(mb))} MB`;
};

const formatDuration = (seconds?: number | null): string => {
    const total = Math.max(0, Math.round(Number(seconds || 0)));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
};

// Number input with a local string draft so the field can be cleared and
// retyped. A naive controlled <input type=number> with `Number(e.target.value)`
// turns an empty field into 0, which then prefixes whatever the user types
// next (e.g. backspace "100" → "0", type "20" → "020").
type NumberFieldProps = {
    value: number;
    onChange: (next: number) => void;
    min?: number;
    max?: number;
    step?: number;
    className?: string;
    'aria-label'?: string;
};
const NumberField: React.FC<NumberFieldProps> = ({ value, onChange, min, max, step, className, ...rest }) => {
    const [draft, setDraft] = React.useState<string>(String(value));
    const lastCommitted = React.useRef<number>(value);
    React.useEffect(() => {
        if (value !== lastCommitted.current) {
            setDraft(String(value));
            lastCommitted.current = value;
        }
    }, [value]);
    return (
        <input
            type='number'
            className={className}
            min={min}
            max={max}
            step={step}
            value={draft}
            onChange={(e) => {
                const raw = e.target.value;
                setDraft(raw);
                if (raw === '' || raw === '-' || raw === '.' || raw === '-.') return;
                const n = Number(raw);
                if (Number.isFinite(n)) {
                    lastCommitted.current = n;
                    onChange(n);
                }
            }}
            onBlur={() => {
                const n = Number(draft);
                if (draft === '' || !Number.isFinite(n)) {
                    setDraft(String(value));
                }
            }}
            {...rest}
        />
    );
};

const StageToggle: React.FC<{
    on: boolean;
    onToggle: () => void;
    label: string;
}> = ({ on, onToggle, label }) => (
    <label className='StageToggle' aria-label={label}>
        <input type='checkbox' checked={on} onChange={onToggle} />
        <span className='StageToggleTrack'><span className='StageToggleThumb' /></span>
    </label>
);

const ResourcePresetControl: React.FC<{
    settings: ResourceSettings;
    stage: ResourceStageId;
    includeBatch?: boolean;
    recommendedProfile?: ResourceProfileId;
    ramLabel?: string | null;
    onChange: (settings: ResourceSettings) => void;
}> = ({ settings, stage, includeBatch = false, recommendedProfile, ramLabel, onChange }) => {
    const stageLabel = stage === 'infer' ? 'Prediction' : stage[0].toUpperCase() + stage.slice(1);
    const updateSetting = (patch: Partial<ResourceSettings>) => onChange({ ...settings, ...patch });
    return (
        <div className='ResourceControl'>
            <div className='ChoiceStrip WorkerPresetStrip'>
                {RESOURCE_OPTIONS.map((option) => {
                    const selected = settings.profile === option.id;
                    const next = resourceSettingsFromProfile(option.id, settings.advancedOpen);
                    return (
                        <button
                            key={option.id}
                            type='button'
                            className={`Choice WorkerPreset ${selected ? 'active' : ''}`}
                            onClick={() => onChange(next)}
                            aria-pressed={selected}
                        >
                            <span className='ChoiceTitle'>
                                {option.label}
                                {recommendedProfile === option.id && <span className='RecPill'>·rec</span>}
                            </span>
                            <span className='ChoiceDetail'>{option.detail}</span>
                            <span className='ChoiceMeta'>
                                {option.note}{ramLabel && selected ? ` · ${ramLabel}` : ''}
                            </span>
                        </button>
                    );
                })}
            </div>
            <button
                type='button'
                className='AdvancedToggle'
                aria-expanded={settings.advancedOpen}
                onClick={() => updateSetting({ advancedOpen: !settings.advancedOpen })}
            >
                <span>Advanced settings</span>
                <span className='AdvancedChevron' aria-hidden>{settings.advancedOpen ? '-' : '+'}</span>
            </button>
            {settings.advancedOpen && (
                <div className='AdvancedResourceGrid'>
                    {includeBatch && (
                        <label className='AdvancedField'>
                            <span>Batch</span>
                            <NumberField
                                className='Num'
                                min={1}
                                max={256}
                                step={1}
                                value={settings.batchSize}
                                onChange={(n) => updateSetting({ batchSize: n })}
                                aria-label={`${stageLabel} batch size`}
                            />
                        </label>
                    )}
                    <label className='AdvancedField'>
                        <span>Workers</span>
                        <NumberField
                            className='Num'
                            min={1}
                            max={64}
                            step={1}
                            value={settings.numWorkers}
                            onChange={(n) => updateSetting({ numWorkers: n })}
                            aria-label={`${stageLabel} workers`}
                        />
                    </label>
                    <label className='AdvancedField'>
                        <span>Decode</span>
                        <NumberField
                            className='Num'
                            min={1}
                            max={16}
                            step={1}
                            value={settings.decodeThreads}
                            onChange={(n) => updateSetting({ decodeThreads: n })}
                            aria-label={`${stageLabel} decode threads`}
                        />
                    </label>
                    <label className='AdvancedField'>
                        <span>Prefetch</span>
                        <NumberField
                            className='Num'
                            min={1}
                            max={16}
                            step={1}
                            value={settings.prefetchFactor}
                            onChange={(n) => updateSetting({ prefetchFactor: n })}
                            aria-label={`${stageLabel} prefetch factor`}
                        />
                    </label>
                </div>
            )}
        </div>
    );
};

const PipelineBuilder: React.FC<PipelineBuilderProps> = ({ pipeline }) => {
    const {
        config,
        setConfig,
        pipelineStatus,
        stepStates,
        activeJobId,
        currentStep,
        metrics,
        predictions,
        trainingEstimate,
        trainingEstimateStatus,
        trainingEstimateError,
        canRun,
        validationError,
    } = pipeline;
    const isRunning = pipelineStatus === 'running';
    const isDone = pipelineStatus === 'completed' || pipelineStatus === 'failed' || pipelineStatus === 'cancelled';
    const isExecuting = isRunning || isDone;

    const cacheEstimateById = (id: string) => trainingEstimate?.cache_options.find((option) => option.id === id);
    const modelEstimateById = (id: 'small' | 'large') => trainingEstimate?.model_options.find((option) => option.id === id);
    const profileEstimateById = (id: string) => trainingEstimate?.resource_profiles.find((option) => option.id === id);
    const resolutionEstimateById = (id: number) => trainingEstimate?.resolution_options.find((option) => option.id === id);
    const selectedCacheEstimate = cacheEstimateById(config.trainCacheMode);
    const selectedProfileEstimate = profileEstimateById(config.resources.train.profile);
    const modelResourceFor = (size: 'small' | 'large') => {
        const estimate = modelEstimateById(size);
        return {
            vram_mb: estimate?.vram_mb ?? MODEL_RESOURCE_FALLBACK[size].vram_mb,
            ram_mb: estimate?.ram_mb ?? MODEL_RESOURCE_FALLBACK[size].ram_mb,
        };
    };
    const profileRamFor = (id: 'low' | 'balanced' | 'high') => (
        profileEstimateById(id)?.ram_mb ?? PROFILE_RAM_FALLBACK[id]
    );
    const selectedModelResource = modelResourceFor(config.modelSize);
    const trainPeakVramFor = (resolution: 112 | 144 | 192 | 224) => {
        const estimate = resolutionEstimateById(resolution);
        const estimateVram = config.modelSize === 'large'
            ? estimate?.large_vram_mb
            : estimate?.small_vram_mb;
        return estimateVram ?? RESOLUTION_TRAIN_PEAK_FALLBACK[config.modelSize][resolution];
    };
    const resolutionExtraVramFor = (resolution: 112 | 144 | 192 | 224) => Math.max(
        0,
        trainPeakVramFor(resolution) - selectedModelResource.vram_mb,
    );
    const selectedTrainPeakVramMb = selectedModelResource.vram_mb + resolutionExtraVramFor(config.trainCacheResolution);
    const selectedProfileRamMb = profileRamFor(config.resources.train.profile);
    const selectedRamMb = selectedCacheEstimate && selectedProfileEstimate
        ? selectedModelResource.ram_mb + selectedCacheEstimate.ram_mb + selectedProfileRamMb
        : null;
    const selectedDiskMb = selectedCacheEstimate ? selectedCacheEstimate.disk_mb : null;
    const behaviorPreview = trainingEstimate?.behaviors.slice(0, 8) || [];
    const [cliCopied, setCliCopied] = React.useState(false);
    const [activeStage, setActiveStage] = React.useState<StageId>('train');

    React.useEffect(() => {
        setCliCopied(false);
    }, [pipeline.cliCommand?.command]);

    React.useEffect(() => {
        if (currentStep === 'prep') {
            setActiveStage('train');
        } else if (currentStep === 'train' || currentStep === 'test' || currentStep === 'infer') {
            setActiveStage(currentStep);
        }
    }, [currentStep]);

    const copyCliCommand = async () => {
        if (!pipeline.cliCommand?.command || !navigator.clipboard) return;
        try {
            await navigator.clipboard.writeText(pipeline.cliCommand.command);
            setCliCopied(true);
        } catch {
            setCliCopied(false);
        }
    };

    const setResource = (stage: ResourceStageId, settings: ResourceSettings) => {
        setConfig((prev) => ({
            resources: {
                ...prev.resources,
                [stage]: settings,
            },
        }));
    };

    const stageStatus = (id: StageId): string => {
        if (isExecuting) {
            if (id === 'train' && stepStates.prep.status === 'running') return 'running';
            if (id === 'train' && stepStates.prep.status === 'failed') return 'failed';
            if (id === 'train' && stepStates.prep.status === 'cancelled') return 'cancelled';
            return stepStates[id].status;
        }
        return config.steps[id] ? 'enabled' : 'disabled';
    };
    const trainShowsPrep = isExecuting
        && config.steps.train
        && stepStates.prep.status === 'running';
    const activeCount = (config.steps.train ? 1 : 0) + (config.steps.test ? 1 : 0) + (config.steps.infer ? 1 : 0);
    const enabledStageSummary = STAGE_ORDER
        .filter((stage) => config.steps[stage.id])
        .map((stage) => stage.tabLabel)
        .join(' → ') || 'No steps selected';
    const focusStage = (stageId: StageId) => {
        setActiveStage(stageId);
        window.requestAnimationFrame(() => {
            document.getElementById(`stage-tab-${stageId}`)?.focus();
        });
    };
    const handleStageTabKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>, stageId: StageId) => {
        const currentIndex = STAGE_ORDER.findIndex((stage) => stage.id === stageId);
        if (currentIndex < 0) return;
        if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
            event.preventDefault();
            focusStage(STAGE_ORDER[(currentIndex + 1) % STAGE_ORDER.length].id);
        } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
            event.preventDefault();
            focusStage(STAGE_ORDER[(currentIndex + STAGE_ORDER.length - 1) % STAGE_ORDER.length].id);
        } else if (event.key === 'Home') {
            event.preventDefault();
            focusStage(STAGE_ORDER[0].id);
        } else if (event.key === 'End') {
            event.preventDefault();
            focusStage(STAGE_ORDER[STAGE_ORDER.length - 1].id);
        }
    };

    return (
        <div className='PipelineBuilder'>
            {/* ───────────────  RUN MODEL PLAN  ─────────────── */}
            <header className='PlanBar' aria-label='Run Model plan'>
                <div className='PlanBarLead'>
                    <span className='SectionMarker'>Run Model</span>
                    <span className='PlanBarSubtitle'>
                        enabled <strong>{activeCount}</strong>/3 steps · {enabledStageSummary}
                    </span>
                </div>
                <div className='PlanStages' role='tablist' aria-label='Run Model setup sections'>
                    {STAGE_ORDER.map((stage) => {
                        const status = stageStatus(stage.id);
                        const on = config.steps[stage.id];
                        const selected = activeStage === stage.id;
                        const active = isExecuting && (currentStep === stage.id || (stage.id === 'train' && currentStep === 'prep'));
                        return (
                            <div
                                key={stage.id}
                                className={`PlanStageItem status-${status} ${on ? 'on' : 'off'} ${active ? 'active' : ''} ${selected ? 'selected' : ''}`}
                            >
                                <button
                                    type='button'
                                    id={`stage-tab-${stage.id}`}
                                    className='PlanStageBtn'
                                    onClick={() => setActiveStage(stage.id)}
                                    role='tab'
                                    aria-selected={selected}
                                    tabIndex={selected ? 0 : -1}
                                    aria-controls={`stage-panel-${stage.id}`}
                                    aria-label={`${stage.label}: ${on ? 'enabled' : 'disabled'}`}
                                    onKeyDown={(event) => handleStageTabKeyDown(event, stage.id)}
                                >
                                    <span className='PlanStageGlyph' aria-hidden>
                                        {status === 'completed' && <svg width='12' height='12' viewBox='0 0 12 12'><path d='M2 6.5l2.5 2.5L10 3' stroke='currentColor' strokeWidth='2' fill='none' strokeLinecap='round' strokeLinejoin='round'/></svg>}
                                        {status === 'failed' && <svg width='12' height='12' viewBox='0 0 12 12'><path d='M3 3l6 6M9 3l-6 6' stroke='currentColor' strokeWidth='2' strokeLinecap='round'/></svg>}
                                        {status === 'running' && <span className='PlanStagePulse' />}
                                        {(status === 'enabled' || status === 'pending') && <span className='PlanStageDot on' />}
                                        {(status === 'disabled' || status === 'skipped') && <span className='PlanStageDot off' />}
                                        {status === 'cancelled' && <span className='PlanStageDot warn' />}
                                    </span>
                                    <span className='PlanStageNum'>{stage.num}</span>
                                    <span className='PlanStageName'>{stage.label}</span>
                                    <span className='PlanStageState'>
                                        {isExecuting
                                            ? (status === 'running' ? 'running' : status)
                                            : (selected ? (on ? 'open' : 'off') : (on ? 'enabled' : 'off'))}
                                    </span>
                                    <span className='PlanStageSummary'>{stage.summary}</span>
                                </button>
                                {!isExecuting && (
                                    <div className='PlanStageSwitch'>
                                        <StageToggle
                                            on={on}
                                            onToggle={() => {
                                                setActiveStage(stage.id);
                                                pipeline.toggleStep(stage.id);
                                            }}
                                            label={`${on ? 'Disable' : 'Enable'} ${stage.label}`}
                                        />
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>
            </header>

            {!isExecuting && (
                <>
                    {/* ───────────────  STAGE PANELS  ─────────────── */}
                    <div className='StagePanels'>
                        {/* ── TRAIN PANEL ── */}
                        <section
                            id='stage-panel-train'
                            className={`StageCol ${config.steps.train ? 'on' : 'off'}`}
                            role='tabpanel'
                            aria-labelledby='stage-tab-train'
                            hidden={activeStage !== 'train'}
                        >
                            <header className='StageColHead'>
                                <span className='StageColMark'>§ 01</span>
                                <span className='StageColTitle'>Train model</span>
                                <span className='StageColRule' aria-hidden />
                            </header>
                            {!config.steps.train && (
                                <p className='StageColRationale'>
                                    Training is disabled. Enable it from the Train model tab above to fit a model on annotated videos.
                                </p>
                            )}
                            {config.steps.train && (
                                <div className='StageColBody'>
                                    <div className='Row'>
                                        <div className='RowKey'>
                                            <label className='K'>Dataset Pairs</label>
                                            <span className='KHint'>Browse any folder and tick pairs — selections accumulate. Click a row&apos;s CSV chip to swap annotation files.</span>
                                        </div>
                                        <div className='RowVal'>
                                            <PathPicker
                                                value={config.datasetSelection.folder}
                                                onChange={(folder) => setConfig((prev) => ({ datasetSelection: { ...prev.datasetSelection, folder } }))}
                                                mode='multiPair'
                                                multiFolder={true}
                                                selectedPairs={config.datasetSelection.pairs}
                                                onSelectedPairsChange={(pairs) => setConfig((prev) => ({ datasetSelection: { ...prev.datasetSelection, pairs } }))}
                                                requireSourceVideo={true}
                                                placeholder='/path/to/dataset'
                                                storageKey='pipeline-train-dataset'
                                            />
                                        </div>
                                    </div>

                                    <div className='Row'>
                                        <div className='RowKey'><label className='K'>Model Size</label></div>
                                        <div className='RowVal'>
                                            <div className='ChoiceStrip ModelStrip'>
                                                {(['small', 'large'] as const).map((size) => {
                                                    const estimate = modelEstimateById(size);
                                                    const res = modelResourceFor(size);
                                                    return (
                                                        <button
                                                            key={size}
                                                            type='button'
                                                            className={`Choice ${config.modelSize === size ? 'active' : ''}`}
                                                            onClick={() => setConfig({ modelSize: size })}
                                                            aria-pressed={config.modelSize === size}
                                                        >
                                                            <span className='ChoiceTitle'>
                                                                {size === 'small' ? 'Small' : 'Large'}
                                                                {estimate?.recommended && <span className='RecPill'>·rec</span>}
                                                            </span>
                                                            <span className='ChoiceDetail'>
                                                                {formatMb(res.vram_mb)} VRAM
                                                            </span>
                                                        </button>
                                                    );
                                                })}
                                            </div>
                                        </div>
                                    </div>

                                    <div className='Row'>
                                        <div className='RowKey'><label className='K'>Video Clip</label></div>
                                        <div className='RowVal'>
                                            <div className='ChoiceStrip'>
                                                {CACHE_OPTIONS.map((option) => {
                                                    const estimate = cacheEstimateById(option.id);
                                                    return (
                                                        <button
                                                            key={option.id}
                                                            type='button'
                                                            className={`Choice ${config.trainCacheMode === option.id ? 'active' : ''}`}
                                                            onClick={() => setConfig({ trainCacheMode: option.id })}
                                                            aria-pressed={config.trainCacheMode === option.id}
                                                        >
                                                            <span className='ChoiceTitle'>
                                                                {option.label}
                                                                {estimate?.recommended && <span className='RecPill'>·rec</span>}
                                                            </span>
                                                            <span className='ChoiceDetail'>
                                                                {estimate
                                                                    ? `+${formatMb(estimate.disk_mb)} disk · +${formatMb(estimate.ram_mb)} RAM`
                                                                    : option.detail}
                                                            </span>
                                                        </button>
                                                    );
                                                })}
                                            </div>
                                        </div>
                                    </div>

                                    <div className='Row'>
                                        <div className='RowKey'><label className='K'>Resolution</label></div>
                                        <div className='RowVal'>
                                            <div className='ChoiceStrip Compact'>
                                                {RESOLUTION_OPTIONS.map((option) => {
                                                    const estimate = resolutionEstimateById(option.id);
                                                    return (
                                                        <button
                                                            key={option.id}
                                                            type='button'
                                                            className={`Choice Square ${config.trainCacheResolution === option.id ? 'active' : ''}`}
                                                            onClick={() => setConfig({ trainCacheResolution: option.id })}
                                                            aria-pressed={config.trainCacheResolution === option.id}
                                                        >
                                                            <span className='ChoiceTitle'>
                                                                {option.label}
                                                                {estimate?.recommended && <span className='RecPill'>·dft</span>}
                                                            </span>
                                                            <span className='ChoiceDetail'>+{formatMb(resolutionExtraVramFor(option.id))}</span>
                                                        </button>
                                                    );
                                                })}
                                            </div>
                                        </div>
                                    </div>

                                    <div className='Row'>
                                        <div className='RowKey'>
                                            <label className='K'>Resource Preset</label>
                                            <span className='KHint'>Choose how much CPU and RAM the loader may use.</span>
                                        </div>
                                        <div className='RowVal'>
                                            <ResourcePresetControl
                                                settings={config.resources.train}
                                                stage='train'
                                                recommendedProfile={trainingEstimate?.recommendations.resource_profile}
                                                ramLabel={`~${formatMb(profileRamFor(config.resources.train.profile))} loader RAM`}
                                                onChange={(settings) => setResource('train', settings)}
                                            />
                                        </div>
                                    </div>

                                    <div className='Divider' aria-hidden />

                                    <div className='Schedule'>
                                        <span className='ScheduleLabel'>Schedule</span>
                                        <div className='ScheduleFields'>
                                            <label className='SchedField'>
                                                <span>Epochs</span>
                                                <NumberField className='Num' min={1} step={1} value={config.totalEpochs} onChange={(n) => setConfig({ totalEpochs: n })} aria-label='Total epochs' />
                                            </label>
                                            <label className='SchedField'>
                                                <span>Val Start</span>
                                                <NumberField className='Num' min={0} step={1} value={config.valStartEpoch} onChange={(n) => setConfig({ valStartEpoch: n })} aria-label='Validation start epoch' />
                                            </label>
                                            <label className='SchedField'>
                                                <span>Val Interval</span>
                                                <NumberField className='Num' min={1} step={1} value={config.valInterval} onChange={(n) => setConfig({ valInterval: n })} aria-label='Validation interval (epochs)' />
                                            </label>
                                            <label className='SchedField'>
                                                <span>T/V Ratio</span>
                                                <NumberField className='Num' min={0.05} max={0.95} step={0.05} value={config.trainValRatio} onChange={(n) => setConfig({ trainValRatio: n })} aria-label='Train/val split ratio' />
                                            </label>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </section>

                        {/* ── TEST PANEL ── */}
                        <section
                            id='stage-panel-test'
                            className={`StageCol ${config.steps.test ? 'on' : 'off'}`}
                            role='tabpanel'
                            aria-labelledby='stage-tab-test'
                            hidden={activeStage !== 'test'}
                        >
                            <header className='StageColHead'>
                                <span className='StageColMark'>§ 02</span>
                                <span className='StageColTitle'>Test model</span>
                                <span className='StageColRule' aria-hidden />
                            </header>
                            {!config.steps.test && (
                                <p className='StageColRationale'>
                                    Training already includes an automatic test pass and prints metrics. Enable it from the Test model tab above for a separate evaluation dataset or to re-test a saved model.
                                </p>
                            )}
                            {config.steps.test && (
                                <div className='StageColBody'>
                                    {config.steps.train ? (
                                        <div className='Row'>
                                            <div className='RowKey'>
                                                <label className='K'>Test Dataset Pairs</label>
                                                <span className='KHint'>Train pairs are reused for test (single prep). Pick pairs in the Train column.</span>
                                            </div>
                                        </div>
                                    ) : (
                                        <div className='Row'>
                                            <div className='RowKey'>
                                                <label className='K'>Test Dataset Pairs</label>
                                                <span className='KHint'>Browse any folder and tick pairs. Click a row&apos;s CSV chip to swap annotation files.</span>
                                            </div>
                                            <div className='RowVal'>
                                                <PathPicker
                                                    value={config.testDatasetSelection.folder}
                                                    onChange={(folder) => setConfig((prev) => ({ testDatasetSelection: { ...prev.testDatasetSelection, folder } }))}
                                                    mode='multiPair'
                                                    multiFolder={true}
                                                    selectedPairs={config.testDatasetSelection.pairs}
                                                    onSelectedPairsChange={(pairs) => setConfig((prev) => ({ testDatasetSelection: { ...prev.testDatasetSelection, pairs } }))}
                                                    requireSourceVideo={true}
                                                    placeholder='/path/to/test-dataset'
                                                    storageKey='pipeline-test-dataset'
                                                />
                                            </div>
                                        </div>
                                    )}
                                    <div className='Row'>
                                        <div className='RowKey'>
                                            <label className='K'>Model Folder</label>
                                            <span className='KHint'>{config.steps.train ? 'Train output is used automatically.' : 'Folder named model_YYYYMMDD_HHMMSS'}</span>
                                        </div>
                                        <div className='RowVal'>
                                            <PathPicker
                                                value={config.modelLoadPath}
                                                onChange={(v) => setConfig({ modelLoadPath: v })}
                                                placeholder={config.steps.train ? '(uses Train output)' : '/path/to/model_YYYYMMDD_HHMMSS'}
                                                disabled={config.steps.train}
                                                storageKey='pipeline-model-load'
                                            />
                                        </div>
                                    </div>
                                    <div className='Row'>
                                        <div className='RowKey'>
                                            <label className='K'>Resource Preset</label>
                                            <span className='KHint'>No auto-tune. Pick the evaluation resource level.</span>
                                        </div>
                                        <div className='RowVal'>
                                            <ResourcePresetControl
                                                settings={config.resources.test}
                                                stage='test'
                                                includeBatch
                                                onChange={(settings) => setResource('test', settings)}
                                            />
                                        </div>
                                    </div>
                                </div>
                            )}
                        </section>

                        {/* ── PREDICT PANEL ── */}
                        <section
                            id='stage-panel-infer'
                            className={`StageCol ${config.steps.infer ? 'on' : 'off'}`}
                            role='tabpanel'
                            aria-labelledby='stage-tab-infer'
                            hidden={activeStage !== 'infer'}
                        >
                            <header className='StageColHead'>
                                <span className='StageColMark'>§ 03</span>
                                <span className='StageColTitle'>Run predictions</span>
                                <span className='StageColRule' aria-hidden />
                            </header>
                            {!config.steps.infer && (
                                <p className='StageColRationale'>
                                    Predictions are disabled. Enable it from the Run predictions tab above to run a trained or supplied model on new videos.
                                </p>
                            )}
                            {config.steps.infer && (
                                <div className='StageColBody'>
                                    <div className='Row'>
                                        <div className='RowKey'>
                                            <label className='K'>Input Videos</label>
                                            <span className='KHint'>Pick a folder, then check videos to run on. CSV files (if any) are ignored.</span>
                                        </div>
                                        <div className='RowVal'>
                                            <PathPicker
                                                value={config.inputSelection.folder}
                                                onChange={(folder) => setConfig((prev) => ({ inputSelection: { ...prev.inputSelection, folder, stems: [], pairs: [], csvByStem: {} } }))}
                                                mode='multiPair'
                                                selectedStems={config.inputSelection.stems}
                                                onSelectedStemsChange={(stems) => setConfig((prev) => ({ inputSelection: { ...prev.inputSelection, stems } }))}
                                                requireSourceVideo={false}
                                                placeholder='/path/to/videos'
                                                storageKey='pipeline-input'
                                            />
                                        </div>
                                    </div>
                                    <div className='Row'>
                                        <div className='RowKey'>
                                            <label className='K'>Model Folder</label>
                                            <span className='KHint'>{config.steps.train ? 'Train output is used automatically.' : 'Folder named model_YYYYMMDD_HHMMSS'}</span>
                                        </div>
                                        <div className='RowVal'>
                                            <PathPicker
                                                value={config.modelLoadPath}
                                                onChange={(v) => setConfig({ modelLoadPath: v })}
                                                placeholder={config.steps.train ? '(uses Train output)' : '/path/to/model_YYYYMMDD_HHMMSS'}
                                                disabled={config.steps.train}
                                                storageKey='pipeline-model-load'
                                            />
                                        </div>
                                    </div>
                                    <div className='Row'>
                                        <div className='RowKey'>
                                            <label className='K'>Resource Preset</label>
                                            <span className='KHint'>No auto-tune. Pick the prediction resource level.</span>
                                        </div>
                                        <div className='RowVal'>
                                            <ResourcePresetControl
                                                settings={config.resources.infer}
                                                stage='infer'
                                                includeBatch
                                                onChange={(settings) => setResource('infer', settings)}
                                            />
                                        </div>
                                    </div>
                                    <div className='Row'>
                                        <div className='RowKey'>
                                            <label className='K'>Prediction Output</label>
                                            <span className='KHint'>Annotated video render and detection-confidence threshold.</span>
                                        </div>
                                        <div className='RowVal'>
                                            <div className='InferOutputGrid'>
                                                <label className='InlineToggle'>
                                                    <input
                                                        type='checkbox'
                                                        checked={config.inferAnnotatedVideo}
                                                        onChange={(e) => setConfig({ inferAnnotatedVideo: e.target.checked })}
                                                    />
                                                    <span className='InlineToggleTrack'><span className='InlineToggleThumb' /></span>
                                                    <span className='InlineToggleLabel'>Annotated video</span>
                                                </label>
                                                <label className='ThresholdInline'>
                                                    <span className='ThresholdLabel'>Threshold</span>
                                                    <input
                                                        type='number'
                                                        min='0'
                                                        max='1'
                                                        step='0.01'
                                                        value={config.inferThreshold}
                                                        onChange={(e) => setConfig({ inferThreshold: parseThresholdInput(e.currentTarget) })}
                                                        aria-label='Prediction threshold'
                                                    />
                                                </label>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </section>
                    </div>

                    {/* ───────────────  RESOURCE STRIP  ─────────────── */}
                    {config.steps.train && (
                        <div className='ResourceStrip' aria-live='polite'>
                            <span className='StripLead'>RESOURCE BUDGET</span>
                            <span className='StripItem'>
                                <span className='StripK'>VRAM</span>
                                <strong>~{formatMb(selectedTrainPeakVramMb)}</strong>
                            </span>
                            <span className='StripDot' aria-hidden />
                            <span className='StripItem'>
                                <span className='StripK'>RAM</span>
                                <strong>{selectedRamMb ? `~${formatMb(selectedRamMb)}` : 'pick pairs'}</strong>
                            </span>
                            <span className='StripDot' aria-hidden />
                            <span className='StripItem'>
                                <span className='StripK'>DISK</span>
                                <strong>{selectedDiskMb !== null ? `~${formatMb(selectedDiskMb)}` : 'pick pairs'}</strong>
                            </span>
                            {trainingEstimate && (
                                <>
                                    <span className='StripPipe' aria-hidden />
                                    <span className='StripItem'>
                                        <span className='StripK'>PAIRS</span>
                                        <strong>{trainingEstimate.summary.pair_count}</strong>
                                    </span>
                                    <span className='StripDot' aria-hidden />
                                    <span className='StripItem'>
                                        <span className='StripK'>SOURCE</span>
                                        <strong>{formatMb(trainingEstimate.summary.source_mb)}</strong>
                                    </span>
                                    <span className='StripDot' aria-hidden />
                                    <span className='StripItem'>
                                        <span className='StripK'>DURATION</span>
                                        <strong>{formatDuration(trainingEstimate.summary.duration_sec)}</strong>
                                    </span>
                                    <span className='StripDot' aria-hidden />
                                    <span className='StripItem'>
                                        <span className='StripK'>CLIPS</span>
                                        <strong>{trainingEstimate.summary.annotated_clip_count}</strong>
                                    </span>
                                </>
                            )}
                        </div>
                    )}

                    {/* ── Estimate notes / behaviors ── */}
                    {config.steps.train && trainingEstimateStatus !== 'idle' && (
                        <div className={`EstimateNote status-${trainingEstimateStatus}`}>
                            {trainingEstimateStatus === 'loading' && (
                                <span className='EstimateLoading'>Estimating selected videos…</span>
                            )}
                            {trainingEstimateStatus === 'error' && (
                                <span className='EstimateError'>{trainingEstimateError}</span>
                            )}
                            {trainingEstimateStatus === 'ready' && trainingEstimate && (
                                <>
                                    <span className='EstimateText'>{trainingEstimate.recommendations.cache_reason}</span>
                                    {behaviorPreview.length > 0 && (
                                        <span className='BehaviorRow'>
                                            <span className='BehaviorLabel'>annotations · {trainingEstimate.summary.annotation_count}</span>
                                            {behaviorPreview.map((behavior) => (
                                                <span className='BehaviorChip' key={behavior.name} title={behavior.name}>
                                                    <strong>{behavior.count}</strong>
                                                    {behavior.name}
                                                </span>
                                            ))}
                                            {trainingEstimate.behaviors.length > behaviorPreview.length && (
                                                <span className='BehaviorChip muted'>
                                                    +{trainingEstimate.behaviors.length - behaviorPreview.length}
                                                </span>
                                            )}
                                        </span>
                                    )}
                                </>
                            )}
                        </div>
                    )}

                    {/* ───────────────  RUN ROW  ─────────────── */}
                    <footer className='RunRow'>
                        {validationError && <div className='ValidationHint'>{validationError}</div>}
                        <div className='RunActions'>
                            <button
                                className='CliBtn'
                                onClick={pipeline.generateCliCommand}
                                disabled={!canRun || pipeline.cliCommandStatus === 'generating'}
                                type='button'
                                aria-label='Generate CLI command'
                            >
                                <svg width='16' height='16' viewBox='0 0 16 16' fill='none'>
                                    <path d='M2.5 4.5h11M2.5 8h11M2.5 11.5h7' stroke='currentColor' strokeWidth='1.4' strokeLinecap='round'/>
                                </svg>
                                <span>{pipeline.cliCommandStatus === 'generating' ? 'Generating…' : 'CLI command'}</span>
                            </button>
                            <button
                                className='RunBtn'
                                onClick={pipeline.run}
                                disabled={!canRun}
                                type='button'
                                aria-label='Run model'
                            >
                                <svg width='14' height='14' viewBox='0 0 16 16' fill='none'><path d='M5 3l8 5-8 5z' fill='currentColor'/></svg>
                                <span>Run model</span>
                                <span className='RunBtnArrow' aria-hidden>↗</span>
                            </button>
                        </div>
                    </footer>

                    {(pipeline.cliCommandStatus === 'ready' || pipeline.cliCommandStatus === 'failed') && (
                        <div className={`CliPanel ${pipeline.cliCommandStatus}`}>
                            {pipeline.cliCommandStatus === 'ready' && pipeline.cliCommand && (
                                <>
                                    <span className='CliPrompt'>$</span>
                                    <code>{pipeline.cliCommand.command}</code>
                                    <button className='CopyBtn' onClick={copyCliCommand} type='button'>
                                        {cliCopied ? '✓ copied' : 'copy'}
                                    </button>
                                </>
                            )}
                            {pipeline.cliCommandStatus === 'failed' && (
                                <span className='CliError'>{pipeline.cliCommandError || 'Could not generate command.'}</span>
                            )}
                        </div>
                    )}
                </>
            )}

            {isExecuting && (
                <div className='ExecutionView'>
                    {trainShowsPrep && (
                        <div className='PrepHint'>
                            <span className='PrepHintDot' />
                            <span>Preparing dataset… train will start once prep completes.</span>
                        </div>
                    )}
                    {isRunning && (
                        <div className='JobInfoBar'>
                            <div className='JobInfoLeft'>
                                {currentStep && <span className='JobStepLabel'>step <strong>{currentStep}</strong></span>}
                                {activeJobId && <span className='JobIdLabel'>job <code>{activeJobId}</code></span>}
                            </div>
                            <button className='CancelBtn' onClick={() => pipeline.cancel()} type='button' aria-label='Cancel run'>
                                <svg width='12' height='12' viewBox='0 0 14 14' fill='none'>
                                    <path d='M3 3l8 8M11 3l-8 8' stroke='currentColor' strokeWidth='1.6' strokeLinecap='round'/>
                                </svg>
                                Cancel
                            </button>
                        </div>
                    )}

                    {activeJobId && isRunning && (
                        <LogViewer jobId={activeJobId} onCancel={() => pipeline.cancel()} />
                    )}
                    {isDone && (
                        <>
                            <PipelineResults metrics={metrics} predictions={predictions} />
                            <div className='DoneActions'>
                                {pipelineStatus === 'failed' && <div className='DoneStatus error'>Run failed</div>}
                                {pipelineStatus === 'cancelled' && <div className='DoneStatus warning'>Run cancelled</div>}
                                {pipelineStatus === 'completed' && <div className='DoneStatus success'>Run completed</div>}
                                <button className='ResetBtn' onClick={pipeline.reset} type='button'>New run</button>
                            </div>
                        </>
                    )}
                </div>
            )}

            <JobDashboard />
        </div>
    );
};

export default PipelineBuilder;
