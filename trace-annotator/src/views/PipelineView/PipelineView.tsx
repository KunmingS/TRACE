import React from 'react';
import './PipelineView.scss';
import { connect } from 'react-redux';
import { updateHomeTab } from '../../store/general/actionCreators';
import { HomeTab } from '../../store/general/types';
import { usePipelineState, PipelineStepId, StepStatus } from '../MainView/PipelineBuilder/usePipelineState';
import PathPicker from '../Common/PathPicker/PathPicker';
import LogViewer from '../Common/LogViewer/LogViewer';
import PipelineResults from '../MainView/PipelineBuilder/PipelineResults';

interface IProps {
    updateHomeTab: (tab: HomeTab) => any;
}

/* ── DAG node layout (x, y center positions in the SVG coordinate space) ── */
const NODE_W = 128;
const NODE_H = 44;
const DAG_NODES: Record<PipelineStepId, { x: number; y: number; label: string }> = {
    prep:  { x: 100, y: 50,  label: 'Prep' },
    train: { x: 100, y: 130, label: 'Train' },
    test:  { x: 52,  y: 210, label: 'Test' },
    infer: { x: 148, y: 210, label: 'Infer' },
};

const DAG_EDGES: [PipelineStepId, PipelineStepId][] = [
    ['prep', 'train'],
    ['train', 'test'],
    ['train', 'infer'],
];

const STATUS_COLOR: Record<StepStatus | 'enabled' | 'disabled', string> = {
    pending: 'rgba(255,255,255,0.12)',
    running: '#3b9eff',
    completed: '#2ecc71',
    failed: '#e74c3c',
    cancelled: '#f39c12',
    skipped: 'rgba(255,255,255,0.06)',
    enabled: 'rgba(59,158,255,0.18)',
    disabled: 'rgba(255,255,255,0.04)',
};

const PipelineView: React.FC<IProps> = ({ updateHomeTab }) => {
    const pipeline = usePipelineState();
    const { config, setConfig, pipelineStatus, stepStates, activeJobId, currentStep, metrics, predictions, canRun, validationError } = pipeline;
    const isRunning = pipelineStatus === 'running';
    const isDone = pipelineStatus === 'completed' || pipelineStatus === 'failed' || pipelineStatus === 'cancelled';
    const isExecuting = isRunning || isDone;

    const goHome = () => updateHomeTab('annotate');

    /* Which steps are logically active in the current configuration */
    const stepEnabled = (id: PipelineStepId): boolean => {
        if (id === 'prep') return config.steps.train || config.steps.test;
        return config.steps[id];
    };

    /* Resolve visual status for a DAG node */
    const nodeStatus = (id: PipelineStepId): string => {
        if (isExecuting) return stepStates[id].status;
        return stepEnabled(id) ? 'enabled' : 'disabled';
    };

    /* Build edge path (straight vertical or angled) */
    const edgePath = (from: PipelineStepId, to: PipelineStepId): string => {
        const a = DAG_NODES[from];
        const b = DAG_NODES[to];
        return `M${a.x},${a.y + NODE_H / 2} L${b.x},${b.y - NODE_H / 2}`;
    };

    const edgeVisible = (from: PipelineStepId, to: PipelineStepId): boolean => {
        return stepEnabled(from) && stepEnabled(to);
    };

    /* For test/infer without train: show a "model" input edge from outside */
    const needsExternalModel = !config.steps.train && (config.steps.test || config.steps.infer);

    return (
        <div className='PipelineView'>
            {/* ── Top Bar ── */}
            <div className='PipelineTopBar'>
                <button className='BackBtn' onClick={goHome} type='button' aria-label='Back to home'>
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                        <path d="M10 3L5 8l5 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                    <img draggable={false} alt='trace' src='/trace-logo.svg' className='BackBtn__logo' />
                    Home
                </button>
                <h1 className='PageTitle'>Pipeline</h1>
                <div className='TopBarRight'>
                    {isRunning && (
                        <button className='CancelBtn' onClick={() => pipeline.cancel()} type='button' aria-label='Cancel pipeline'>
                            Cancel
                        </button>
                    )}
                    {isDone && (
                        <button className='ResetBtn' onClick={pipeline.reset} type='button' aria-label='Start new pipeline'>
                            New Pipeline
                        </button>
                    )}
                </div>
            </div>

            {/* ── Two-Column Layout ── */}
            <div className='PipelineColumns'>
                {/* ─── Left: DAG Visualization ─── */}
                <div className='DagPanel' aria-label='Pipeline execution graph'>
                    <div className='DagHeader'>Execution Graph</div>
                    <svg className='DagSvg' viewBox="0 0 200 260" preserveAspectRatio="xMidYMid meet" role='img' aria-label='Pipeline DAG showing step dependencies: Prep flows to Train, Train flows to Test and Infer'>
                        <defs>
                            <filter id="glow">
                                <feGaussianBlur stdDeviation="3" result="blur"/>
                                <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                            </filter>
                        </defs>

                        {/* Edges */}
                        {DAG_EDGES.map(([from, to]) => {
                            const visible = edgeVisible(from, to);
                            const fromDone = isExecuting && (stepStates[from].status === 'completed');
                            return (
                                <path
                                    key={`${from}-${to}`}
                                    d={edgePath(from, to)}
                                    className={`DagEdge ${visible ? '' : 'hidden'} ${fromDone ? 'done' : ''}`}
                                    fill="none"
                                />
                            );
                        })}

                        {/* "No train" edges: prep→test, prep→infer or external model edges */}
                        {!config.steps.train && config.steps.test && stepEnabled('prep') && (
                            <path
                                d={`M${DAG_NODES.prep.x},${DAG_NODES.prep.y + NODE_H/2} L${DAG_NODES.test.x},${DAG_NODES.test.y - NODE_H/2}`}
                                className='DagEdge'
                                fill="none"
                            />
                        )}
                        {!config.steps.train && config.steps.infer && stepEnabled('prep') && (
                            <path
                                d={`M${DAG_NODES.prep.x},${DAG_NODES.prep.y + NODE_H/2} L${DAG_NODES.infer.x},${DAG_NODES.infer.y - NODE_H/2}`}
                                className='DagEdge'
                                fill="none"
                            />
                        )}

                        {/* Nodes */}
                        {(Object.entries(DAG_NODES) as [PipelineStepId, typeof DAG_NODES.prep][]).map(([id, node]) => {
                            const status = nodeStatus(id);
                            const fill = STATUS_COLOR[status as keyof typeof STATUS_COLOR] || STATUS_COLOR.disabled;
                            const isActive = currentStep === id;
                            const enabled = stepEnabled(id);
                            return (
                                <g key={id} className={`DagNode ${status} ${isActive ? 'active' : ''} ${enabled ? '' : 'off'}`}>
                                    <rect
                                        x={node.x - NODE_W / 2}
                                        y={node.y - NODE_H / 2}
                                        width={NODE_W}
                                        height={NODE_H}
                                        rx={10}
                                        fill={fill}
                                        className='DagNodeRect'
                                        filter={isActive ? 'url(#glow)' : undefined}
                                    />
                                    <text
                                        x={node.x}
                                        y={node.y + 1}
                                        textAnchor="middle"
                                        dominantBaseline="central"
                                        className='DagNodeLabel'
                                    >
                                        {node.label}
                                    </text>
                                    {status === 'running' && (
                                        <circle
                                            cx={node.x + NODE_W / 2 - 14}
                                            cy={node.y}
                                            r={4}
                                            className='DagPulse'
                                        />
                                    )}
                                    {status === 'completed' && (
                                        <text
                                            x={node.x + NODE_W / 2 - 14}
                                            y={node.y + 1}
                                            textAnchor="middle"
                                            dominantBaseline="central"
                                            className='DagCheck'
                                        >&#10003;</text>
                                    )}
                                    {status === 'failed' && (
                                        <text
                                            x={node.x + NODE_W / 2 - 14}
                                            y={node.y + 1}
                                            textAnchor="middle"
                                            dominantBaseline="central"
                                            className='DagFail'
                                        >&#10007;</text>
                                    )}
                                </g>
                            );
                        })}
                    </svg>

                    {/* Legend */}
                    <div className='DagLegend'>
                        <span className='LegendItem'><span className='LegendDot enabled'/> Enabled</span>
                        <span className='LegendItem'><span className='LegendDot running'/> Running</span>
                        <span className='LegendItem'><span className='LegendDot completed'/> Done</span>
                        <span className='LegendItem'><span className='LegendDot failed'/> Failed</span>
                    </div>

                    {/* Status summary during execution */}
                    {isExecuting && (
                        <div className='DagStatus'>
                            {pipelineStatus === 'running' && <span className='StatusBadge running'>Running</span>}
                            {pipelineStatus === 'completed' && <span className='StatusBadge completed'>Completed</span>}
                            {pipelineStatus === 'failed' && <span className='StatusBadge failed'>Failed</span>}
                            {pipelineStatus === 'cancelled' && <span className='StatusBadge cancelled'>Cancelled</span>}
                        </div>
                    )}
                </div>

                {/* ─── Right: Config / Execution ─── */}
                <div className='ConfigPanel'>
                    {!isExecuting && (
                        <>
                            {/* ── Train Card ── */}
                            <div className={`TaskCard ${config.steps.train ? 'enabled' : 'disabled'}`}>
                                <div className='TaskHeader'>
                                    <label className='TaskToggle'>
                                        <input type='checkbox' checked={config.steps.train} onChange={() => pipeline.toggleStep('train')} aria-label='Enable training step' />
                                        <span className='ToggleTrack'><span className='ToggleThumb' /></span>
                                    </label>
                                    <div className='TaskInfo'>
                                        <span className='TaskName'>Train</span>
                                        <span className='TaskDesc'>Train a model on annotated videos</span>
                                    </div>
                                </div>
                                {config.steps.train && (
                                    <div className='TaskFields'>
                                        <div className='FormField'>
                                            <label className='FieldLabel'>Dataset Path</label>
                                            <span className='FieldHint'>Folder with videos + CSV annotations</span>
                                            <PathPicker value={config.datasetPath} onChange={(v) => setConfig({ datasetPath: v })} placeholder='/path/to/dataset' storageKey='pipeline-train-dataset' />
                                        </div>
                                        <div className='FormField'>
                                            <label className='FieldLabel'>Model Size</label>
                                            <div className='SizeToggle'>
                                                <button className={`SizeBtn ${config.modelSize === 'small' ? 'active' : ''}`} onClick={() => setConfig({ modelSize: 'small' })} type='button' aria-pressed={config.modelSize === 'small'}>Small</button>
                                                <button className={`SizeBtn ${config.modelSize === 'large' ? 'active' : ''}`} onClick={() => setConfig({ modelSize: 'large' })} type='button' aria-pressed={config.modelSize === 'large'}>Large</button>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* ── Test Card ── */}
                            <div className={`TaskCard ${config.steps.test ? 'enabled' : 'disabled'}`}>
                                <div className='TaskHeader'>
                                    <label className='TaskToggle'>
                                        <input type='checkbox' checked={config.steps.test} onChange={() => pipeline.toggleStep('test')} aria-label='Enable testing step' />
                                        <span className='ToggleTrack'><span className='ToggleThumb' /></span>
                                    </label>
                                    <div className='TaskInfo'>
                                        <span className='TaskName'>Test</span>
                                        <span className='TaskDesc'>Evaluate model on a test set</span>
                                    </div>
                                </div>
                                {config.steps.test && (
                                    <div className='TaskFields'>
                                        <div className='FormField'>
                                            <label className='FieldLabel'>Test Dataset Path</label>
                                            <span className='FieldHint'>Can differ from training dataset</span>
                                            <PathPicker value={config.testDatasetPath || config.datasetPath} onChange={(v) => setConfig({ testDatasetPath: v })} placeholder='/path/to/test-dataset' storageKey='pipeline-test-dataset' />
                                        </div>
                                        {!config.steps.train && (
                                            <div className='FormField'>
                                                <label className='FieldLabel'>Model Path</label>
                                                <span className='FieldHint'>Folder with best.pth + classmap.txt</span>
                                                <PathPicker value={config.modelPath} onChange={(v) => setConfig({ modelPath: v })} placeholder='/path/to/model' storageKey='pipeline-model' />
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>

                            {/* ── Infer Card ── */}
                            <div className={`TaskCard ${config.steps.infer ? 'enabled' : 'disabled'}`}>
                                <div className='TaskHeader'>
                                    <label className='TaskToggle'>
                                        <input type='checkbox' checked={config.steps.infer} onChange={() => pipeline.toggleStep('infer')} aria-label='Enable inference step' />
                                        <span className='ToggleTrack'><span className='ToggleThumb' /></span>
                                    </label>
                                    <div className='TaskInfo'>
                                        <span className='TaskName'>Inference</span>
                                        <span className='TaskDesc'>Run predictions on new videos</span>
                                    </div>
                                </div>
                                {config.steps.infer && (
                                    <div className='TaskFields'>
                                        <div className='FormField'>
                                            <label className='FieldLabel'>Input Video(s)</label>
                                            <span className='FieldHint'>A video file or folder of videos</span>
                                            <PathPicker value={config.inputPath} onChange={(v) => setConfig({ inputPath: v })} placeholder='/path/to/video.mp4' mode='file' extensions='.mp4,.avi,.mov,.mkv,.webm' storageKey='pipeline-input' />
                                        </div>
                                        {!config.steps.train && (
                                            <div className='FormField'>
                                                <label className='FieldLabel'>Model Path</label>
                                                <span className='FieldHint'>Folder with best.pth + classmap.txt</span>
                                                <PathPicker value={config.modelPath} onChange={(v) => setConfig({ modelPath: v })} placeholder='/path/to/model' storageKey='pipeline-model' />
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>

                            {/* ── Run ── */}
                            <div className='RunSection'>
                                {validationError && <div className='ValidationHint'>{validationError}</div>}
                                <button className='RunBtn' onClick={pipeline.run} disabled={!canRun} type='button' aria-label='Run pipeline'>
                                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M5 3l8 5-8 5z" fill="currentColor" /></svg>
                                    Run Pipeline
                                </button>
                            </div>
                        </>
                    )}

                    {/* ── Execution: Logs + Results ── */}
                    {isExecuting && (
                        <div className='ExecutionPanel'>
                            {activeJobId && isRunning && (
                                <LogViewer jobId={activeJobId} onCancel={() => pipeline.cancel()} />
                            )}
                            {isDone && (
                                <>
                                    <PipelineResults metrics={metrics} predictions={predictions} />
                                    <div className='DoneActions'>
                                        {pipelineStatus === 'failed' && <div className='DoneStatus error'>Pipeline failed</div>}
                                        {pipelineStatus === 'cancelled' && <div className='DoneStatus warning'>Pipeline cancelled</div>}
                                        {pipelineStatus === 'completed' && <div className='DoneStatus success'>Pipeline completed</div>}
                                    </div>
                                </>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

const mapDispatchToProps = { updateHomeTab };

export default connect(null, mapDispatchToProps)(PipelineView);
