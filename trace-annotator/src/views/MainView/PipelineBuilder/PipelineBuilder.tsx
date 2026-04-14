import React from 'react';
import './PipelineBuilder.scss';
import PathPicker from '../../Common/PathPicker/PathPicker';
import LogViewer from '../../Common/LogViewer/LogViewer';
import PipelineResults from './PipelineResults';
import JobDashboard from './JobDashboard';
import { PipelineState, PipelineStepId, StepStatus } from './usePipelineState';

interface PipelineBuilderProps {
    pipeline: PipelineState;
}

/* ── DAG layout constants (linear chain: Prep → Train → Test → Infer) ── */
const NODE_W = 108;
const NODE_H = 36;
const DAG_NODES: Record<PipelineStepId, { x: number; y: number; label: string }> = {
    prep:  { x: 88, y: 36,  label: 'Prep' },
    train: { x: 88, y: 100, label: 'Train' },
    test:  { x: 88, y: 164, label: 'Test' },
    infer: { x: 88, y: 228, label: 'Infer' },
};
const DAG_EDGES: [PipelineStepId, PipelineStepId][] = [
    ['prep', 'train'],
    ['train', 'test'],
    ['test', 'infer'],
];
const STATUS_FILL: Record<StepStatus | 'enabled' | 'disabled', string> = {
    pending: 'rgba(255,255,255,0.10)',
    running: 'rgba(59,158,255,0.22)',
    completed: 'rgba(46,204,113,0.18)',
    failed: 'rgba(231,76,60,0.18)',
    cancelled: 'rgba(243,156,18,0.18)',
    skipped: 'rgba(255,255,255,0.04)',
    enabled: 'rgba(59,158,255,0.14)',
    disabled: 'rgba(255,255,255,0.04)',
};

const PipelineBuilder: React.FC<PipelineBuilderProps> = ({ pipeline }) => {
    const { config, setConfig, pipelineStatus, stepStates, activeJobId, currentStep, metrics, predictions, canRun, validationError } = pipeline;
    const isRunning = pipelineStatus === 'running';
    const isDone = pipelineStatus === 'completed' || pipelineStatus === 'failed' || pipelineStatus === 'cancelled';
    const isExecuting = isRunning || isDone;

    /* Step logic */
    const stepEnabled = (id: PipelineStepId): boolean => {
        if (id === 'prep') return config.steps.train || config.steps.test;
        return config.steps[id];
    };
    const nodeStatus = (id: PipelineStepId): string => {
        if (isExecuting) return stepStates[id].status;
        return stepEnabled(id) ? 'enabled' : 'disabled';
    };
    const edgePath = (from: PipelineStepId, to: PipelineStepId): string => {
        const a = DAG_NODES[from], b = DAG_NODES[to];
        return `M${a.x},${a.y + NODE_H / 2} L${b.x},${b.y - NODE_H / 2}`;
    };

    return (
        <div className='PipelineBuilder'>
            <div className='PipelineColumns'>
                {/* ── Left: Compact DAG ── */}
                <div className='DagColumn'>
                    <div className='DagLabel'>Execution Graph</div>
                    <svg className='DagSvg' viewBox="0 0 176 264" preserveAspectRatio="xMidYMid meet" role='img' aria-label='Pipeline DAG: Prep to Train to Test to Infer'>
                        <defs>
                            <filter id="dagGlow">
                                <feGaussianBlur stdDeviation="2.5" result="b"/>
                                <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
                            </filter>
                        </defs>

                        {/* Edges */}
                        {DAG_EDGES.map(([from, to]) => {
                            const vis = stepEnabled(from) && stepEnabled(to);
                            const done = isExecuting && stepStates[from].status === 'completed';
                            return (
                                <path
                                    key={`${from}-${to}`}
                                    d={edgePath(from, to)}
                                    className={`DagEdge ${vis ? '' : 'hidden'} ${done ? 'done' : ''}`}
                                    fill="none"
                                />
                            );
                        })}
                        {/* Skip-train edge: Prep → Test directly when train is disabled */}
                        {!config.steps.train && config.steps.test && stepEnabled('prep') && (
                            <path d={`M${DAG_NODES.prep.x},${DAG_NODES.prep.y + NODE_H/2} L${DAG_NODES.test.x},${DAG_NODES.test.y - NODE_H/2}`} className='DagEdge' fill="none"/>
                        )}
                        {/* Skip-test edge: Train → Infer directly when test is disabled */}
                        {!config.steps.test && config.steps.train && config.steps.infer && (
                            <path d={`M${DAG_NODES.train.x},${DAG_NODES.train.y + NODE_H/2} L${DAG_NODES.infer.x},${DAG_NODES.infer.y - NODE_H/2}`} className='DagEdge' fill="none"/>
                        )}
                        {/* Skip both train+test: Prep → Infer */}
                        {!config.steps.train && !config.steps.test && config.steps.infer && stepEnabled('prep') && (
                            <path d={`M${DAG_NODES.prep.x},${DAG_NODES.prep.y + NODE_H/2} L${DAG_NODES.infer.x},${DAG_NODES.infer.y - NODE_H/2}`} className='DagEdge' fill="none"/>
                        )}

                        {/* Nodes */}
                        {(Object.entries(DAG_NODES) as [PipelineStepId, typeof DAG_NODES.prep][]).map(([id, node]) => {
                            const status = nodeStatus(id);
                            const fill = STATUS_FILL[status as keyof typeof STATUS_FILL] || STATUS_FILL.disabled;
                            const active = currentStep === id;
                            const on = stepEnabled(id);
                            return (
                                <g key={id} className={`DagNode ${status} ${active ? 'active' : ''} ${on ? '' : 'off'}`}>
                                    <rect
                                        x={node.x - NODE_W / 2} y={node.y - NODE_H / 2}
                                        width={NODE_W} height={NODE_H} rx={8}
                                        fill={fill} className='DagNodeRect'
                                        filter={active ? 'url(#dagGlow)' : undefined}
                                    />
                                    <text x={node.x} y={node.y + 1} textAnchor="middle" dominantBaseline="central" className='DagNodeLabel'>
                                        {node.label}
                                    </text>
                                    {status === 'running' && <circle cx={node.x + NODE_W/2 - 12} cy={node.y} r={3.5} className='DagPulse'/>}
                                    {status === 'completed' && <text x={node.x + NODE_W/2 - 12} y={node.y + 1} textAnchor="middle" dominantBaseline="central" className='DagCheck'>&#10003;</text>}
                                    {status === 'failed' && <text x={node.x + NODE_W/2 - 12} y={node.y + 1} textAnchor="middle" dominantBaseline="central" className='DagFail'>&#10007;</text>}
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

                    {isExecuting && (
                        <div className='DagStatus'>
                            <span className={`StatusBadge ${pipelineStatus}`}>
                                {pipelineStatus === 'running' ? 'Running' : pipelineStatus === 'completed' ? 'Completed' : pipelineStatus === 'failed' ? 'Failed' : 'Cancelled'}
                            </span>
                        </div>
                    )}
                </div>

                {/* ── Right: Config / Execution ── */}
                <div className='ConfigColumn'>
                    {!isExecuting && (
                        <>
                            {/* Train */}
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

                            {/* Test */}
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
                                            <label className='FieldLabel'>Test Dataset</label>
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

                            {/* Infer */}
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
                                            <PathPicker value={config.inputPath || config.datasetPath} onChange={(v) => setConfig({ inputPath: v })} placeholder='/path/to/video.mp4' mode='file' extensions='.mp4,.avi,.mov,.mkv,.webm' storageKey='pipeline-input' />
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

                            {/* Run */}
                            <div className='RunSection'>
                                {validationError && <div className='ValidationHint'>{validationError}</div>}
                                <button className='RunBtn' onClick={pipeline.run} disabled={!canRun} type='button' aria-label='Run pipeline'>
                                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M5 3l8 5-8 5z" fill="currentColor" /></svg>
                                    Run Pipeline
                                </button>
                            </div>
                        </>
                    )}

                    {isExecuting && (
                        <div className='ExecutionView'>
                            {/* Job info bar */}
                            {isRunning && (
                                <div className='JobInfoBar'>
                                    <div className='JobInfoLeft'>
                                        {currentStep && <span className='JobStepLabel'>Step: <strong>{currentStep}</strong></span>}
                                        {activeJobId && <span className='JobIdLabel'>Job: <code>{activeJobId}</code></span>}
                                    </div>
                                    <button className='CancelPipelineBtn' onClick={() => pipeline.cancel()} type='button' aria-label='Cancel pipeline'>
                                        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                                            <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                                        </svg>
                                        Cancel Pipeline
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
                                        {pipelineStatus === 'failed' && <div className='DoneStatus error'>Pipeline failed</div>}
                                        {pipelineStatus === 'cancelled' && <div className='DoneStatus warning'>Pipeline cancelled</div>}
                                        {pipelineStatus === 'completed' && <div className='DoneStatus success'>Pipeline completed</div>}
                                        <button className='ResetBtn' onClick={pipeline.reset} type='button'>New Pipeline</button>
                                    </div>
                                </>
                            )}
                        </div>
                    )}
                </div>
            </div>
            <JobDashboard />
        </div>
    );
};

export default PipelineBuilder;
